"""Serialise the agent-visible dataset and the sealed evaluation set."""
import json, os, hashlib
from datetime import datetime, timezone
import numpy as np
from netCDF4 import Dataset
import config as C
import fields as F
import physics as P
from generate import DIURNAL, diurnal_factor

SAT_SCALE = 2.0e-8
SAT_FILL = -32767


def _crs(ds):
    v = ds.createVariable("crs", "i4")
    v.grid_mapping_name = "lambert_azimuthal_equal_area"
    v.longitude_of_projection_origin = 10.0
    v.latitude_of_projection_origin = 52.0
    v.false_easting = 4321000.0
    v.false_northing = 3210000.0
    v.semi_major_axis = 6378137.0
    v.inverse_flattening = 298.257222101
    v.epsg_code = "EPSG:3035"
    v.comment = ("Domain coordinates x, y are metres east/north of the domain "
                 "origin. Add domain_origin_x_proj / domain_origin_y_proj to "
                 "obtain EPSG:3035 coordinates.")
    return v


def write_domain(path, gx, gy, gam, region_masks):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "Inference grid, projection and source-region definition"
        ds.schema_version = "1.0"
        ds.domain_origin_x_proj = C.X0_PROJ
        ds.domain_origin_y_proj = C.Y0_PROJ
        ds.createDimension("x", C.NX_C)
        ds.createDimension("y", C.NY_C)
        ds.createDimension("nv", 2)
        ds.createDimension("region", C.N_REGIONS)
        vx = ds.createVariable("x", "f8", ("x",)); vx[:] = gx
        vx.units = "m"; vx.long_name = "projected easting of cell centre"
        vx.bounds = "x_bnds"
        vy = ds.createVariable("y", "f8", ("y",)); vy[:] = gy
        vy.units = "m"; vy.long_name = "projected northing of cell centre"
        vy.bounds = "y_bnds"
        xb = ds.createVariable("x_bnds", "f8", ("x", "nv"))
        xb[:] = np.stack([gx - C.DX_COARSE / 2, gx + C.DX_COARSE / 2], axis=1)
        yb = ds.createVariable("y_bnds", "f8", ("y", "nv"))
        yb[:] = np.stack([gy - C.DX_COARSE / 2, gy + C.DX_COARSE / 2], axis=1)
        va = ds.createVariable("cell_area", "f8", ("y", "x"))
        va[:] = np.full((C.NY_C, C.NX_C), C.DX_COARSE ** 2)
        va.units = "m2"
        vg = ds.createVariable("grid_convergence", "f8", ("y", "x"))
        vg[:] = gam
        vg.units = "degree"
        vg.long_name = "angle from grid north to true north, positive counter-clockwise"
        vg.comment = ("Rotate east/north wind components into grid axes with "
                      "u_grid =  cos(g)*u_east + sin(g)*v_north ; "
                      "v_grid = -sin(g)*u_east + cos(g)*v_north.")
        vr = ds.createVariable("region_id", "i4", ("region",))
        vr[:] = np.arange(1, C.N_REGIONS + 1)
        vn = ds.createVariable("region_name", str, ("region",))
        for i, n in enumerate(C.REGION_NAMES):
            vn[i] = n
        vm = ds.createVariable("region_support", "i1", ("region", "y", "x"))
        vm[:] = region_masks.astype(np.int8)
        vm.long_name = "cells where the road prior of this region is non-zero"
        _crs(ds)


def write_emissions(path, road_c, fixed_c, episodes):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "Prior road NOx inventory, fixed non-road sources, time modulation"
        ds.schema_version = "1.0"
        ds.road_mass_basis = "NO2-equivalent"
        ds.fixed_mass_basis = "N"
        ds.molar_mass_no2_kg_per_mol = C.M_NO2
        ds.molar_mass_n_kg_per_mol = C.M_N
        ds.comment = ("road_prior is reported as NO2-equivalent mass; "
                      "fixed_sources is reported as mass of nitrogen. The two "
                      "fields require different molar masses to reach mol NOx.")
        ds.createDimension("x", C.NX_C)
        ds.createDimension("y", C.NY_C)
        ds.createDimension("region", C.N_REGIONS)
        ds.createDimension("episode", C.N_EPISODES)
        ds.createDimension("hour", 24)
        # mol m-2 s-1  ->  kg km-2 h-1 on the published mass basis
        road_kg = road_c * C.M_NO2 * 1.0e6 * 3600.0
        fixed_kg = fixed_c * C.M_N * 1.0e6 * 3600.0
        vr = ds.createVariable("road_prior", "f8", ("region", "y", "x"), zlib=True)
        vr[:] = road_kg
        vr.units = "kg km-2 h-1"
        vr.mass_basis = "NO2-equivalent"
        vr.long_name = "prior road-transport NOx emission flux by source region"
        vf = ds.createVariable("fixed_sources", "f8", ("y", "x"), zlib=True)
        vf[:] = fixed_kg
        vf.units = "kg km-2 h-1"
        vf.mass_basis = "N"
        vf.long_name = "prescribed non-road NOx emission flux, not adjusted"
        vt = ds.createVariable("road_time_factor", "f8", ("episode",))
        vt[:] = [e.road_factor for e in episodes]
        vt.long_name = "day-type multiplier applied to road_prior for this episode"
        vd = ds.createVariable("road_diurnal_factor", "f8", ("hour",))
        vd[:] = DIURNAL
        vd.long_name = "hour-of-day multiplier, nodes at whole UTC hours"
        vd.comment = ("Interpolate linearly in time between nodes and wrap "
                      "cyclically. Total road flux at time t is "
                      "road_prior * road_time_factor(episode) * diurnal(t).")
        vx = ds.createVariable("fixed_time_factor", "f8", ("episode",))
        vx[:] = [e.fixed_factor for e in episodes]
        vx.long_name = "multiplier applied to fixed_sources for this episode"
        ve = ds.createVariable("episode_id", "i4", ("episode",))
        ve[:] = np.arange(C.N_EPISODES)


def write_meteorology(path, met, episodes):
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "Supplied meteorology on the 12 km analysis grid"
        ds.schema_version = "1.0"
        ds.time_convention = "seconds since 1970-01-01T00:00:00Z (UTC)"
        ds.horizontal_diffusivity_m2_s = C.K_DIFF
        ds.comment = ("Winds are true east/north components. The transport wind "
                      "is the layer-thickness weighted mean over the levels "
                      "inside the boundary layer, formed on this grid at these "
                      "times, then interpolated bilinearly in space and linearly "
                      "in time to the inference grid.")
        nt = met["times"].shape[1]
        nlev = met["u_east"].shape[2]
        ds.createDimension("episode", C.N_EPISODES)
        ds.createDimension("time", nt)
        ds.createDimension("level", nlev)
        ds.createDimension("nv", 2)
        ds.createDimension("xm", C.NX_M)
        ds.createDimension("ym", C.NY_M)
        v = ds.createVariable("xm", "f8", ("xm",)); v[:] = met["x"]; v.units = "m"
        v = ds.createVariable("ym", "f8", ("ym",)); v[:] = met["y"]; v.units = "m"
        v = ds.createVariable("time", "f8", ("episode", "time"))
        v[:] = met["times"]; v.units = "s"
        v.long_name = "seconds since 1970-01-01T00:00:00Z"
        v = ds.createVariable("level_bounds", "f8", ("level", "nv"))
        v[:] = np.stack([met["z_iface"][:-1], met["z_iface"][1:]], axis=1)
        v.units = "m"; v.long_name = "layer interfaces above ground"
        for name, arr, unit, ln in [
            ("u_east", met["u_east"], "m s-1", "wind component toward true east"),
            ("v_north", met["v_north"], "m s-1", "wind component toward true north")]:
            vv = ds.createVariable(name, "f4", ("episode", "time", "level", "ym", "xm"),
                                   zlib=True, complevel=4)
            vv[:] = arr.astype(np.float32); vv.units = unit; vv.long_name = ln
        vv = ds.createVariable("blh", "f4", ("episode", "time", "ym", "xm"),
                               zlib=True, complevel=4)
        vv[:] = met["blh"].astype(np.float32); vv.units = "m"
        vv.long_name = "boundary layer height above ground"
        vv = ds.createVariable("f_no2", "f4", ("episode", "time", "ym", "xm"),
                               zlib=True, complevel=4)
        vv[:] = met["f_no2"].astype(np.float32); vv.units = "1"
        vv.long_name = "prescribed NO2 / NOx molar fraction"
        vv = ds.createVariable("photolysis_factor", "f4",
                               ("episode", "time", "ym", "xm"),
                               zlib=True, complevel=4)
        vv[:] = met["photolysis"].astype(np.float32); vv.units = "1"
        vv.long_name = "prescribed photolysis proxy driving the NOx sink"
        ve = ds.createVariable("episode_id", "i4", ("episode",))
        ve[:] = np.arange(C.N_EPISODES)
        vs = ds.createVariable("episode_start", "f8", ("episode",))
        vs[:] = [e.t0 for e in episodes]; vs.units = "s"
        vs.long_name = "start of the integration window, UTC seconds"
        vn = ds.createVariable("episode_end", "f8", ("episode",))
        vn[:] = [e.t_end for e in episodes]; vn.units = "s"
        va = ds.createVariable("analysis_start", "f8", ("episode",))
        va[:] = [e.t0 + C.SPINUP_HOURS * 3600.0 for e in episodes]; va.units = "s"
        va.long_name = "end of spin-up; no graded observation precedes this time"
        vo = ds.createVariable("overpass_time", "f8", ("episode",))
        vo[:] = [e.t_sat for e in episodes]; vo.units = "s"
        vo.long_name = "nominal satellite overpass time, also the episode reference time"
        vr = ds.createVariable("wind_regime", str, ("episode",))
        for e, ep in enumerate(episodes):
            vr[e] = C.REGIME_NAMES[ep.regime]


def write_swath(path, ep, sw, sens, value, sigma, qa, rng):
    n = len(value)
    packed = np.clip(np.round(value / SAT_SCALE), -32000, 32000)
    packed = np.where(qa < 0.05, SAT_FILL, packed).astype(np.int16)
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = f"Synthetic tropospheric NO2 column swath, episode {ep.e:02d}"
        ds.schema_version = "1.0"
        ds.episode_id = ep.e
        ds.time_convention = "seconds since 1970-01-01T00:00:00Z (UTC)"
        ds.qa_acceptance_threshold = 0.75
        ds.comment = ("Decode packed values with scale_factor and add_offset and "
                      "drop _FillValue entries before applying the QA rule. "
                      "Valid retrievals with negative values are genuine noise "
                      "realisations and must be retained.")
        ds.createDimension("obs", n)
        ds.createDimension("corner", 4)
        v = ds.createVariable("obs_id", str, ("obs",))
        for i in range(n):
            v[i] = f"SAT-E{ep.e:02d}-{i:05d}"
        v = ds.createVariable("scanline", "i4", ("obs",)); v[:] = sw["scanline"]
        v = ds.createVariable("ground_pixel", "i4", ("obs",)); v[:] = sw["ground_pixel"]
        v = ds.createVariable("time_utc", "f8", ("obs",)); v[:] = sw["time"]
        v.units = "s"
        v = ds.createVariable("centre_x", "f8", ("obs",)); v[:] = sw["centres"][:, 0]
        v.units = "m"
        v = ds.createVariable("centre_y", "f8", ("obs",)); v[:] = sw["centres"][:, 1]
        v.units = "m"
        v = ds.createVariable("corner_x", "f8", ("obs", "corner"))
        v[:] = sw["corners"][:, :, 0]; v.units = "m"
        v.comment = "footprint corners in counter-clockwise order"
        v = ds.createVariable("corner_y", "f8", ("obs", "corner"))
        v[:] = sw["corners"][:, :, 1]; v.units = "m"
        v = ds.createVariable("no2_column", "i2", ("obs",), fill_value=np.int16(SAT_FILL))
        v.set_auto_maskandscale(False)          # write the packed integers verbatim
        v[:] = packed
        v.scale_factor = SAT_SCALE
        v.add_offset = 0.0
        v.units = "mol m-2"
        v.long_name = "retrieved tropospheric NO2 column"
        v = ds.createVariable("no2_column_uncertainty", "f8", ("obs",))
        v[:] = sigma; v.units = "mol m-2"
        v.long_name = "1-sigma retrieval noise, uncorrelated between footprints"
        v = ds.createVariable("qa_value", "f8", ("obs",)); v[:] = qa
        v.long_name = "retrieval quality; accept qa_value >= 0.75"
        v = ds.createVariable("vertical_sensitivity", "f8", ("obs",)); v[:] = sens
        v.units = "1"
        v.long_name = "integral of the averaging kernel against the normalised vertical shape"


def write_stations_csv(path, stations, episodes, values, sigmas, train_eps):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["obs_id", "station_id", "episode_id", "x_m", "y_m",
                    "inlet_height_m", "site_type", "interval_start_utc",
                    "interval_end_utc", "no2_ug_m3", "sigma_ug_m3"])
        for e in train_eps:
            ep = episodes[e]
            for si, st in enumerate(stations):
                for ki, (ta, tb) in enumerate(ep.station_intervals):
                    w.writerow([f"STA-E{e:02d}-{st['station_id']}-{ki}",
                                st["station_id"], e,
                                f"{st['x']:.1f}", f"{st['y']:.1f}",
                                f"{st['inlet_height_m']:.2f}", st["site_type"],
                                f"{ta:.0f}", f"{tb:.0f}",
                                f"{values[e][si, ki]:.4f}",
                                f"{sigmas[e][si, ki]:.4f}"])


def write_queries(path, episodes, stations, swaths, sens, sat_sigma, sta_sigma,
                  held):
    nsat = sum(len(swaths[e]["corners"]) for e in held)
    nsta = len(held) * len(stations) * len(episodes[held[0]].station_intervals)
    with Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "Held-out prediction queries; measured values are withheld"
        ds.schema_version = "1.0"
        ds.time_convention = "seconds since 1970-01-01T00:00:00Z (UTC)"
        ds.comment = ("Every listed obs_id must appear exactly once in "
                      "/app/predicted_observations.csv.")
        ds.createDimension("sat_query", nsat)
        ds.createDimension("sta_query", nsta)
        ds.createDimension("corner", 4)
        sid = ds.createVariable("sat_obs_id", str, ("sat_query",))
        sep = ds.createVariable("sat_episode_id", "i4", ("sat_query",))
        stm = ds.createVariable("sat_time_utc", "f8", ("sat_query",)); stm.units = "s"
        scx = ds.createVariable("sat_centre_x", "f8", ("sat_query",)); scx.units = "m"
        scy = ds.createVariable("sat_centre_y", "f8", ("sat_query",)); scy.units = "m"
        sqx = ds.createVariable("sat_corner_x", "f8", ("sat_query", "corner")); sqx.units = "m"
        sqy = ds.createVariable("sat_corner_y", "f8", ("sat_query", "corner")); sqy.units = "m"
        svs = ds.createVariable("sat_vertical_sensitivity", "f8", ("sat_query",))
        sus = ds.createVariable("sat_uncertainty", "f8", ("sat_query",)); sus.units = "mol m-2"
        k = 0
        for e in held:
            sw = swaths[e]
            n = len(sw["corners"])
            for i in range(n):
                sid[k + i] = f"SAT-E{e:02d}-{i:05d}"
            sep[k:k + n] = e
            stm[k:k + n] = sw["time"]
            scx[k:k + n] = sw["centres"][:, 0]
            scy[k:k + n] = sw["centres"][:, 1]
            sqx[k:k + n] = sw["corners"][:, :, 0]
            sqy[k:k + n] = sw["corners"][:, :, 1]
            svs[k:k + n] = sens[e]
            sus[k:k + n] = sat_sigma[e]
            k += n
        tid = ds.createVariable("sta_obs_id", str, ("sta_query",))
        tst = ds.createVariable("sta_station_id", str, ("sta_query",))
        tep = ds.createVariable("sta_episode_id", "i4", ("sta_query",))
        tx = ds.createVariable("sta_x", "f8", ("sta_query",)); tx.units = "m"
        ty = ds.createVariable("sta_y", "f8", ("sta_query",)); ty.units = "m"
        th = ds.createVariable("sta_inlet_height_m", "f8", ("sta_query",)); th.units = "m"
        ta_ = ds.createVariable("sta_interval_start_utc", "f8", ("sta_query",)); ta_.units = "s"
        tb_ = ds.createVariable("sta_interval_end_utc", "f8", ("sta_query",)); tb_.units = "s"
        tu = ds.createVariable("sta_uncertainty", "f8", ("sta_query",)); tu.units = "ug m-3"
        j = 0
        for e in held:
            ep = episodes[e]
            for si, st in enumerate(stations):
                for ki, (t0, t1) in enumerate(ep.station_intervals):
                    tid[j] = f"STA-E{e:02d}-{st['station_id']}-{ki}"
                    tst[j] = st["station_id"]
                    tep[j] = e
                    tx[j] = st["x"]; ty[j] = st["y"]
                    th[j] = st["inlet_height_m"]
                    ta_[j] = t0; tb_[j] = t1
                    tu[j] = sta_sigma[e][si, ki]
                    j += 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_all(root, S):
    envd = os.path.join(root, "environment", "data")
    tstd = os.path.join(root, "tests", "truth")
    rng = S["rng"]
    episodes = S["episodes"]
    stations = S["stations"]
    held = list(C.HELD_OUT_EPISODES)
    train = [e for e in range(C.N_EPISODES) if e not in held]
    anomaly_t0 = datetime.strptime(C.ANOMALY_FROM_EPISODE_TIME, "%Y-%m-%dT%H:%M:%SZ") \
        .replace(tzinfo=timezone.utc).timestamp()

    masks = (S["road_c"] > 1e-13)
    write_domain(os.path.join(envd, "domain.nc"), S["gx_c"], S["gy_c"], S["gam_c"], masks)
    write_emissions(os.path.join(envd, "emissions.nc"), S["road_c"], S["fixed_c"], episodes)
    write_meteorology(os.path.join(envd, "meteorology.nc"), S["met"], episodes)

    # ---- observation noise, QA and reported values -------------------------
    sat_sigma, sat_report, sat_qa = {}, {}, {}
    for e in range(C.N_EPISODES):
        n = len(S["truth_sat"][e])
        floor = rng.uniform(*C.SAT_SIGMA_FLOOR, n)
        sig = np.hypot(floor, C.SAT_SIGMA_REL * np.abs(S["truth_sat"][e]))
        qa = np.clip(rng.beta(7.0, 1.0, n) * 1.02, 0.0, 1.0)
        cloudy = rng.random(n) < 0.16
        qa[cloudy] = rng.uniform(0.05, 0.72, cloudy.sum())
        dropped = rng.random(n) < 0.03
        qa[dropped] = 0.0
        val = S["truth_sat"][e] + rng.normal(0.0, sig)
        # cloud-affected retrievals are biased and noisier; the QA rule removes them
        val[cloudy] += rng.normal(6.0e-5, 4.0e-5, cloudy.sum())
        # Across-track row anomaly: a contiguous band of ground pixels develops
        # a gain and offset error partway through the record.  The quality flag
        # does not flag it and the reported uncertainty does not cover it.
        gp = S["swaths"][e]["ground_pixel"]
        lo, hi = C.ANOMALY_GROUND_PIXEL
        if episodes[e].t0 >= anomaly_t0:
            rows = (gp >= lo) & (gp <= hi)
            val[rows] = val[rows] * C.ANOMALY_GAIN + C.ANOMALY_OFFSET
            val[rows] += rng.normal(0.0, sig[rows] * (C.ANOMALY_NOISE_INFLATION - 1.0))
        sat_sigma[e], sat_report[e], sat_qa[e] = sig, val, qa

    sta_sigma, sta_report = {}, {}
    for e in range(C.N_EPISODES):
        base = np.array([st["sigma_ug_m3"] for st in stations])[:, None]
        floor = np.repeat(base, S["truth_sta"][e].shape[1], axis=1) \
            * rng.uniform(0.85, 1.15, S["truth_sta"][e].shape)
        sig = np.hypot(floor, C.STA_SIGMA_REL * np.abs(S["truth_sta"][e]))
        sta_sigma[e] = sig
        sta_report[e] = S["truth_sta"][e] + rng.normal(0.0, sig)

    for e in train:
        write_swath(os.path.join(envd, "swaths", f"no2_swath_ep{e:02d}.nc"),
                    episodes[e], S["swaths"][e], S["sens"][e],
                    sat_report[e], sat_sigma[e], sat_qa[e], rng)
    write_stations_csv(os.path.join(envd, "stations.csv"), stations, episodes,
                       sta_report, sta_sigma, train)
    write_queries(os.path.join(envd, "prediction_queries.nc"), episodes, stations,
                  S["swaths"], S["sens"], sat_sigma, sta_sigma, held)

    # ---- sealed evaluation set --------------------------------------------
    sat_ids, sat_truth, sat_sig, sat_reg, sat_ep = [], [], [], [], []
    for e in held:
        n = len(S["truth_sat"][e])
        sat_ids += [f"SAT-E{e:02d}-{i:05d}" for i in range(n)]
        sat_truth.append(S["truth_sat"][e])
        sat_sig.append(sat_sigma[e])
        sat_reg += [C.REGIME_NAMES[C.EPISODE_REGIME[e]]] * n
        sat_ep += [e] * n
    sta_ids, sta_truth, sta_sig, sta_reg, sta_ep = [], [], [], [], []
    for e in held:
        ep = episodes[e]
        for si, st in enumerate(stations):
            for ki in range(len(ep.station_intervals)):
                sta_ids.append(f"STA-E{e:02d}-{st['station_id']}-{ki}")
                sta_truth.append(S["truth_sta"][e][si, ki])
                sta_sig.append(sta_sigma[e][si, ki])
                sta_reg.append(C.REGIME_NAMES[C.EPISODE_REGIME[e]])
                sta_ep.append(e)

    road_mol = S["road_c"]                       # mol m-2 s-1 on the inference grid
    area = C.DX_COARSE ** 2
    per_region_base = road_mol.sum(axis=(1, 2)) * area          # mol s-1 at factor 1
    totals = {}
    for e in range(C.N_EPISODES):
        f = episodes[e].road_factor * float(diurnal_factor(episodes[e].t_sat))
        totals[e] = float((C.TRUE_SOURCE_SCALE * per_region_base).sum() * f)

    np.savez_compressed(
        os.path.join(tstd, "evaluation.npz"),
        sat_obs_id=np.array(sat_ids), sat_truth=np.concatenate(sat_truth),
        sat_sigma_meas=np.concatenate(sat_sig), sat_regime=np.array(sat_reg),
        sat_episode=np.array(sat_ep, dtype=int),
        sta_obs_id=np.array(sta_ids), sta_truth=np.array(sta_truth),
        sta_sigma_meas=np.array(sta_sig), sta_regime=np.array(sta_reg),
        sta_episode=np.array(sta_ep, dtype=int),
        repr_sat_floor=S["sigma_repr_sat"][0], repr_sat_rel=S["sigma_repr_sat"][1],
        repr_sta_floor=S["sigma_repr_sta"][0], repr_sta_rel=S["sigma_repr_sta"][1],
        true_source_scale=C.TRUE_SOURCE_SCALE,
        true_tau0_s=C.TRUE_TAU0_S,
        true_c_ref=C.TRUE_C_REF,
        true_fixed_scale=C.TRUE_FIXED_SCALE,
        true_zeta0=C.TRUE_ZETA0,
        true_wind_speed_scale=C.TRUE_WIND_SPEED_SCALE,
        true_wind_rotation_deg=C.TRUE_WIND_ROTATION_DEG,
        true_background=C.TRUE_BACKGROUND,
        per_region_base_mol_s=per_region_base,
        true_total_mol_s=np.array([totals[e] for e in range(C.N_EPISODES)]),
        held_out_episodes=np.array(held, dtype=int),
    )

    # ---- manifest ----------------------------------------------------------
    files = {}
    for dirpath, _, names in os.walk(envd):
        for nm in sorted(names):
            p = os.path.join(dirpath, nm)
            rel = os.path.relpath(p, envd)
            if rel == "data_manifest.json":
                continue
            files[rel] = dict(sha256=sha256(p), bytes=os.path.getsize(p))
    manifest = dict(
        schema_version="1.0",
        dataset="no2-emission-inversion synthetic pilot",
        data_nature="synthetic; generated by an independent seeded forward model",
        time_convention="seconds since 1970-01-01T00:00:00Z (UTC), no leap seconds",
        inference_grid=dict(nx=C.NX_C, ny=C.NY_C, dx_m=C.DX_COARSE,
                            lx_m=C.LX, ly_m=C.LY),
        n_episodes=C.N_EPISODES, n_regions=C.N_REGIONS, n_stations=C.N_STATIONS,
        training_episodes=train, query_episodes=held,
        constants=dict(molar_mass_no2_kg_per_mol=C.M_NO2,
                       molar_mass_n_kg_per_mol=C.M_N,
                       horizontal_diffusivity_m2_s=C.K_DIFF),
        representation_error=dict(
            model=("sigma_representation**2 = floor**2 + (relative * value)**2, "
                   "where value is the observed column or concentration when "
                   "fitting and the withheld value when grading"),
            satellite_floor_mol_m2=S["sigma_repr_sat"][0],
            satellite_relative=S["sigma_repr_sat"][1],
            station_floor_ug_m3=S["sigma_repr_sta"][0],
            station_relative=S["sigma_repr_sta"][1],
            method=("least-squares fit of the squared difference between the "
                    "generating model and the same physical state evaluated on "
                    "the 4 km inference grid, against a constant plus a term "
                    "proportional to the squared signal, over all episodes")),
        parameter_bounds={k: list(v) for k, v in C.BOUNDS.items()},
        file_count=len(files),
        files=files,
    )
    with open(os.path.join(envd, "data_manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("manifest written with", len(files), "entries")
