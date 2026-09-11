"""Build the hidden truth, the agent-visible dataset and the sealed evaluation set.

Run:  python3 generate.py --out <bundle-root>
"""
import argparse, json, hashlib, os, sys
import numpy as np
from netCDF4 import Dataset

import config as C
import fields as F
import physics as P

RNG = np.random.default_rng(C.MASTER_SEED)

DIURNAL = np.array([0.38, 0.28, 0.24, 0.26, 0.42, 0.83, 1.36, 1.78, 1.62, 1.30,
                    1.16, 1.14, 1.18, 1.20, 1.26, 1.44, 1.72, 1.66, 1.28, 0.98,
                    0.82, 0.70, 0.58, 0.46])


def diurnal_factor(t_epoch):
    """Piecewise-linear in UTC hour, nodes at whole hours, cyclic."""
    h = (np.asarray(t_epoch, dtype=float) % 86400.0) / 3600.0
    i0 = np.floor(h).astype(int) % 24
    i1 = (i0 + 1) % 24
    w = h - np.floor(h)
    return DIURNAL[i0] * (1 - w) + DIURNAL[i1] * w


class Episode:
    def __init__(self, e, met, rng):
        self.e = e
        self.t0 = met["times"][e][1]                       # start of the window
        self.t_end = self.t0 + C.EPISODE_HOURS * 3600.0
        self.t_sat = self.t0 + rng.uniform(6.6, 7.9) * 3600.0
        self.regime = C.EPISODE_REGIME[e]
        self.road_factor = float(np.round(rng.uniform(0.86, 1.14), 4))
        self.fixed_factor = float(np.round(rng.uniform(0.90, 1.10), 4))
        # six hourly station averaging intervals over the analysis window
        a0 = self.t0 + C.SPINUP_HOURS * 3600.0
        self.station_intervals = [(a0 + k * 3600.0, a0 + (k + 1) * 3600.0)
                                  for k in range(int(C.EPISODE_HOURS - C.SPINUP_HOURS))]


def interp_met_to_grid(met, blmean, e, gx, gy):
    """Pre-build spatial interpolation operators from the 12 km met grid."""
    ix, iy, tx, ty = P.bilinear_weights(met["x"], met["y"], gx, gy)
    return ix, iy, tx, ty


def met_at_time(met, blmean, e, t, op):
    ix, iy, tx, ty = op
    tt = met["times"][e]
    k = int(np.clip(np.searchsorted(tt, t) - 1, 0, len(tt) - 2))
    w = (t - tt[k]) / (tt[k + 1] - tt[k])
    def lerp(a):
        return (1 - w) * a[k] + w * a[k + 1]
    u = P.apply_bilinear(lerp(blmean[e, :, :, :, 0]), ix, iy, tx, ty)
    v = P.apply_bilinear(lerp(blmean[e, :, :, :, 1]), ix, iy, tx, ty)
    h = P.apply_bilinear(lerp(met["blh"][e]), ix, iy, tx, ty)
    f = P.apply_bilinear(lerp(met["f_no2"][e]), ix, iy, tx, ty)
    return u, v, h, f


def run_episode(ep, met, blmean, grid, road, fixed, params, sat_geom, stations,
                dt):
    """Integrate one episode and return the noise-free observation operators' output."""
    nx, ny, dx, gx, gy, gamma = grid
    s, tau, a_sc, delta, bcoef = params
    bg = P.background_field(bcoef, gx, gy)
    adv = P.Advector(nx, ny, dx, bg)
    op = interp_met_to_grid(met, blmean, ep.e, gx, gy)

    emis_road = np.tensordot(s, road, axes=(0, 0))        # (ny, nx) mol m-2 s-1
    c = bg.copy()

    n_st = len(stations)
    st_sum = np.zeros((n_st, len(ep.station_intervals)))
    st_cnt = np.zeros((n_st, len(ep.station_intervals)))
    st_ix = np.array([min(int(t["x"] // dx), nx - 1) for t in stations])
    st_iy = np.array([min(int(t["y"] // dx), ny - 1) for t in stations])
    st_inlet = np.array([t["inlet_height_m"] for t in stations])

    sat_value = None
    nsteps = int(round((ep.t_end - ep.t0) / dt))
    t = ep.t0
    sat_done = False
    for n in range(nsteps):
        tm = t + 0.5 * dt
        u, v, h, fno2 = met_at_time(met, blmean, ep.e, tm, op)
        ug, vg = P.corrected_grid_wind(u, v, a_sc, delta, gamma)
        ufx, vfy = P.face_velocities(ug, vg)
        emis = ep.road_factor * diurnal_factor(tm) * emis_road + ep.fixed_factor * fixed
        c_prev = c
        c = adv.step(c, ufx, vfy, emis, dt, tau)

        # satellite snapshot at the overpass time (linear in time within the step)
        if (not sat_done) and (t + dt >= ep.t_sat):
            w = (ep.t_sat - t) / dt
            c_sat = (1 - w) * c_prev + w * c
            _, _, h_s, f_s = met_at_time(met, blmean, ep.e, ep.t_sat, op)
            sat_value = sample_satellite(c_sat * f_s, sat_geom)
            sat_done = True

        # station sampling: instantaneous value at the end of the step
        for ki, (ta, tb) in enumerate(ep.station_intervals):
            if ta <= t + dt <= tb:
                zeta = st_inlet / h[st_iy, st_ix]
                val = (fno2[st_iy, st_ix] * c[st_iy, st_ix]
                       * P.phi_shape(zeta) / h[st_iy, st_ix])
                st_sum[:, ki] += val
                st_cnt[:, ki] += 1.0
        t += dt

    st_mol_m3 = st_sum / np.maximum(st_cnt, 1.0)
    st_ug = st_mol_m3 * (C.M_NO2 * 1e3) * 1e6            # mol m-3 -> ug m-3
    return sat_value, st_ug


def sample_satellite(field_fC, sat_geom):
    """Area-weighted footprint average of f*C, times the vertical sensitivity."""
    flat = field_fC.reshape(-1)
    out = np.empty(len(sat_geom["idx"]))
    for i, (idx, w) in enumerate(zip(sat_geom["idx"], sat_geom["wts"])):
        out[i] = float(np.dot(flat[idx], w))
    return out * sat_geom["sensitivity"]


def build_sat_geometry(swath, nx, ny, dx, sensitivity):
    idx, wts, cover = [], [], []
    for k in range(len(swath["corners"])):
        i, w = P.footprint_weights(swath["corners"][k], nx, ny, dx)
        full = P.polygon_area(swath["corners"][k])
        clipped = 0.0
        if len(i):
            # recover absolute overlap area for the coverage diagnostic
            cs = P.clip_polygon_to_box(swath["corners"][k], 0, nx * dx, 0, ny * dx)
            clipped = P.polygon_area(cs)
        idx.append(i)
        wts.append(w)
        cover.append(clipped / full if full > 0 else 0.0)
    return dict(idx=idx, wts=wts, sensitivity=sensitivity, coverage=np.array(cover))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    root = os.path.abspath(args.out)
    envd = os.path.join(root, "environment", "data")
    tstd = os.path.join(root, "tests", "truth")
    prvd = os.path.join(root, "authoring", "provenance")
    for d in (os.path.join(envd, "swaths"), tstd, prvd):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(C.MASTER_SEED)
    print("building static fields ...", flush=True)
    road_f = F.build_road_fields_fine()
    fixed_f = F.build_fixed_sources_fine()
    road_c = F.coarsen(road_f)
    fixed_c = F.coarsen(fixed_f)
    met = F.build_meteorology(rng)
    blmean = P.bl_mean_wind(met)
    stations = F.build_stations(rng)

    gx_f, gy_f = F.grid_centres(C.NX_F, C.NY_F, C.DX_FINE)
    gx_c, gy_c = F.grid_centres(C.NX_C, C.NY_C, C.DX_COARSE)
    gam_f = F.grid_convergence_field(C.NX_F, C.NY_F, C.DX_FINE)
    gam_c = F.grid_convergence_field(C.NX_C, C.NY_C, C.DX_COARSE)
    grid_f = (C.NX_F, C.NY_F, C.DX_FINE, gx_f, gy_f, gam_f)
    grid_c = (C.NX_C, C.NY_C, C.DX_COARSE, gx_c, gy_c, gam_c)

    episodes = [Episode(e, met, rng) for e in range(C.N_EPISODES)]

    print("building swath geometry ...", flush=True)
    swaths, geom_f, geom_c, sens = [], [], [], []
    for e, ep in enumerate(episodes):
        # every retained footprint lies wholly inside the domain, so the
        # footprint operator is unambiguous for both fitting and grading
        sw = F.build_swath(rng, e, ep.t_sat, inside_only=True)
        npix = len(sw["corners"])
        a0 = rng.uniform(0.70, 0.86, npix)
        a1 = rng.uniform(0.05, 0.22, npix)
        # vertical sensitivity = integral of A(z) * g(z) dz over the layer
        zeta = np.linspace(0.0, 1.0, 201)
        phi = P.phi_shape(zeta)
        hbar = float(met["blh"][e].mean())
        A = a0[:, None] + a1[:, None] * (zeta[None, :] * hbar / 1000.0)
        m = np.trapezoid(A * phi[None, :], zeta, axis=1)
        swaths.append(sw)
        sens.append(m)
        geom_f.append(build_sat_geometry(sw, C.NX_F, C.NY_F, C.DX_FINE, m))
        geom_c.append(build_sat_geometry(sw, C.NX_C, C.NY_C, C.DX_COARSE, m))
        print(f"  episode {e}: {npix} footprints", flush=True)

    true_params = (C.TRUE_SOURCE_SCALE, C.TRUE_LIFETIME_S, C.TRUE_WIND_SPEED_SCALE,
                   C.TRUE_WIND_ROTATION_DEG, C.TRUE_BACKGROUND)

    print("running truth on the 2 km grid ...", flush=True)
    truth_sat, truth_sta = [], []
    for e, ep in enumerate(episodes):
        sv, st = run_episode(ep, met, blmean, grid_f, road_f, fixed_f, true_params,
                             geom_f[e], stations, C.DT_FINE)
        truth_sat.append(sv)
        truth_sta.append(st)
        print(f"  episode {e}: sat mean {sv.mean():.3e} mol m-2, "
              f"station mean {st.mean():.2f} ug m-3", flush=True)

    print("running the same truth on the 4 km inference grid ...", flush=True)
    repr_sat, repr_sta = [], []
    for e, ep in enumerate(episodes):
        sv, st = run_episode(ep, met, blmean, grid_c, road_c, fixed_c, true_params,
                             geom_c[e], stations, C.DT_COARSE)
        repr_sat.append(sv)
        repr_sta.append(st)

    d_sat = np.concatenate([truth_sat[e] - repr_sat[e] for e in range(C.N_EPISODES)])
    d_sta = np.concatenate([(truth_sta[e] - repr_sta[e]).ravel() for e in range(C.N_EPISODES)])
    sigma_repr_sat = float(np.sqrt(np.mean(d_sat ** 2)))
    sigma_repr_sta = float(np.sqrt(np.mean(d_sta ** 2)))
    print(f"representation error: satellite {sigma_repr_sat:.4e} mol m-2, "
          f"station {sigma_repr_sta:.4f} ug m-3", flush=True)

    np.savez_compressed(
        os.path.join(prvd, "truth_state.npz"),
        truth_sat=np.concatenate(truth_sat),
        truth_sat_counts=np.array([len(v) for v in truth_sat]),
        truth_sta=np.array(truth_sta),
        repr_sat=np.concatenate(repr_sat),
        repr_sta=np.array(repr_sta))

    state = dict(road_f=road_f, fixed_f=fixed_f, road_c=road_c, fixed_c=fixed_c,
                 met=met, blmean=blmean, stations=stations, episodes=episodes,
                 swaths=swaths, sens=sens, truth_sat=truth_sat, truth_sta=truth_sta,
                 sigma_repr_sat=sigma_repr_sat, sigma_repr_sta=sigma_repr_sta,
                 gam_c=gam_c, gx_c=gx_c, gy_c=gy_c, geom_c=geom_c, rng=rng)
    import writers
    writers.write_all(root, state)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
