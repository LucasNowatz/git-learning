"""Fit deliberately flawed variants of the inversion and score them.

Each variant is a single broken link in the modelling chain, of the kind a
solver plausibly gets wrong.  For each one we refit every free parameter with
that flaw in place, predict the withheld episodes with the same flawed forward
model, and score the result.  This is the evidence behind the frozen thresholds.
"""
import copy, json, os, sys, time
import numpy as np
from scipy.optimize import minimize

ROOT = "/home/user/git-learning/no2-emission-inversion"
sys.path.insert(0, f"{ROOT}/solution")
sys.path.insert(0, f"{ROOT}/authoring/evidence")
import no2_inversion as M
import score_output as SC

M_NO2 = 46.0055e-3
M_N = 14.0067e-3
OUTDIR = "/tmp/baselines"


def nearest_matrix(cx, cy, nx, ny):
    from scipy.sparse import coo_matrix
    xc = cx.mean(axis=1); yc = cy.mean(axis=1)
    i = np.clip((xc // M.DX).astype(int), 0, nx - 1)
    j = np.clip((yc // M.DX).astype(int), 0, ny - 1)
    k = np.arange(len(xc))
    return coo_matrix((np.ones(len(xc)), (k, j * nx + i)), shape=(len(xc), nx * ny)).tocsr()


def build(variant):
    d = M.load_inputs()
    man = json.load(open(f"{ROOT}/environment/data/data_manifest.json"))
    sr_sat = man["representation_error"]["satellite_mol_m2"]
    sr_sta = man["representation_error"]["station_ug_m3"]
    if variant.get("mass_basis_wrong"):
        d["fixed"] = d["fixed"] * (M_N / M_NO2)
    if variant.get("no_diurnal"):
        d["diurnal"] = np.ones(24)
    if variant.get("gamma_ignored"):
        d["gamma"] = np.zeros_like(d["gamma"])
    if variant.get("gamma_sign_flip"):
        d["gamma"] = -d["gamma"]

    sat, sta = M.load_training_obs(d)
    if variant.get("no_qa"):
        sat = []
        from netCDF4 import Dataset
        sdir = f"{M.DATA}/swaths"
        for fn in sorted(os.listdir(sdir)):
            with Dataset(f"{sdir}/{fn}") as ds:
                e = int(ds.episode_id)
                col = ds["no2_column"][:]
                g = np.where(~np.ma.getmaskarray(col))[0]
                sat.append(dict(episode=e, idx=g,
                                obs_id=np.array([str(s) for s in ds["obs_id"][:]])[g],
                                value=np.asarray(col)[g],
                                sigma=ds["no2_column_uncertainty"][:].data[g],
                                cx=ds["corner_x"][:].data[g], cy=ds["corner_y"][:].data[g],
                                sens=ds["vertical_sensitivity"][:].data[g]))
    if variant.get("clip_negative"):
        for s in sat:
            s["value"] = np.clip(s["value"], 0.0, None)

    model = M.Model(d)
    if variant.get("nearest_cell"):
        M.footprint_matrix = nearest_matrix
    blocks = M.build_training_blocks(model, sat, sta, d)
    for e, b in blocks.items():
        b["sat"]["sig_tot"] = np.hypot(b["sat"]["sigma"], sr_sat)
        b["sta"]["sig_tot"] = np.hypot(b["sta"]["sigma"], sr_sta)
    return d, model, blocks


def fit(variant, label):
    t0 = time.time()
    if variant.get("first_order"):
        M.vl = lambda a, b: np.zeros_like(a)      # donor-cell upwind
    if variant.get("dt"):
        M.DT = float(variant["dt"])
    d, model, blocks = build(variant)
    eps = sorted(blocks)
    import multiprocessing as mp
    M._MODEL, M._BLOCKS = model, blocks
    M._POOL = mp.get_context("fork").Pool(4)
    seen, scout = set(), []
    for e in eps:
        if d["regime"][e] not in seen:
            seen.add(d["regime"][e]); scout.append(e)

    fix_a = variant.get("fix_a")
    fix_d = variant.get("fix_delta")
    fix_tau = variant.get("fix_tau")
    fix_s = variant.get("fix_scale")
    if fix_s is not None:
        M.S_LO = float(fix_s) - 1e-9
        M.S_HI = float(fix_s) + 1e-9

    def nl(z):
        tau = float(fix_tau) if fix_tau else float(np.exp(z[0]))
        a = float(fix_a) if fix_a else float(z[1])
        dd = float(fix_d) if fix_d is not None else float(z[2])
        return tau, a, dd

    def obj(z, which):
        tau, a, dd = nl(z)
        if not (M.TAU_LO <= tau <= M.TAU_HI and M.A_LO <= a <= M.A_HI
                and M.D_LO <= dd <= M.D_HI):
            return 1e18
        A, o, s = M.assemble(model, blocks, which, tau, a, dd)
        p, c = M.inner_fit(A, o, s)
        obj.last = (p, c, tau, a, dd)
        return c

    if fix_tau and fix_a and fix_d is not None:
        z = np.array([np.log(fix_tau), fix_a, fix_d])
        obj(z, eps)
    else:
        best = None
        for th in (1.4, 2.4, 3.6, 5.2, 7.2):
            for a in ((fix_a,) if fix_a else (0.8, 1.0, 1.2)):
                for dd in ((fix_d,) if fix_d is not None else (-15.0, -5.0, 5.0, 15.0)):
                    z = np.array([np.log(th * 3600.0), a, dd])
                    j = obj(z, scout)
                    if best is None or j < best[0]:
                        best = (j, z.copy())
        r0 = minimize(lambda z: obj(z, scout), best[1], method="Nelder-Mead",
                      options=dict(xatol=2e-3, fatol=1.0, maxfev=120, adaptive=True))
        r1 = minimize(lambda z: obj(z, eps), r0.x, method="Nelder-Mead",
                      options=dict(xatol=1e-3, fatol=0.5, maxfev=80, adaptive=True))
        obj(r1.x, eps)
    p, chi2, tau, a_sc, delta = obj.last
    s = p[:6]; c00, c10, c01 = p[6:]
    theta = np.concatenate([[1.0], s, [c00, c10, c01]])

    qsat, qsta = M.load_queries()
    out = f"{OUTDIR}/{label}"
    os.makedirs(out, exist_ok=True)
    import csv
    from netCDF4 import Dataset
    rows = []
    for e in sorted(set(qsat["episode"].tolist())):
        ms = qsat["episode"] == e; mt = qsta["episode"] == e
        W = M.footprint_matrix(qsat["cx"][ms], qsat["cy"][ms], model.nx, model.ny)
        spec = dict(ta=qsta["ta"][mt], tb=qsta["tb"][mt], x=qsta["x"][mt],
                    y=qsta["y"][mt], h=qsta["h"][mt])
        As, At = model.run(e, tau, a_sc, delta, sat_W=W, sta=spec)
        for oid, v in zip(qsat["obs_id"][ms], (As @ theta) * qsat["sens"][ms]):
            rows.append((oid, "satellite", float(v), "mol m-2"))
        for oid, v in zip(qsta["obs_id"][mt], At @ theta):
            rows.append((oid, "station", float(v), "ug m-3"))
    with open(f"{out}/predicted_observations.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["obs_id", "instrument_type", "predicted_value", "unit"])
        for r_ in rows:
            w.writerow([r_[0], r_[1], f"{r_[2]:.10e}", r_[3]])
    area = float(d["area"][0, 0])
    # totals are always reported with the *published* conventions
    dref = M.load_inputs()
    base = dref["road"].sum(axis=(1, 2)) * area
    totals = {str(e): float((s * base).sum() * float(dref["road_factor"][e])
                            * float(M.diurnal(dref["t_sat"][e], dref["diurnal"])))
              for e in range(18)}
    json.dump(dict(schema_version="1.0", region_ids=[1, 2, 3, 4, 5, 6],
                   emission_scale=[float(v) for v in s],
                   effective_lifetime_s=float(tau), wind_speed_scale=float(a_sc),
                   wind_rotation_deg=float(delta),
                   background_coefficients=[float(c00), float(c10 - c00), float(c01 - c00)],
                   background_units="mol m-2", integrated_road_mol_s=totals),
              open(f"{out}/result.json", "w"), indent=2)
    M._POOL.close(); M._POOL.join(); M._POOL = None
    M.S_LO, M.S_HI = 0.30, 2.50
    res = SC.score(out, label=label)
    res["chi2_per_obs"] = chi2 / sum(len(blocks[e]["sat"]["value"])
                                     + len(blocks[e]["sta"]["value"]) for e in eps)
    res["fit_seconds"] = round(time.time() - t0, 1)
    return res


VARIANTS = [
    ("prior_unadjusted", dict(fix_scale=1.0, fix_tau=4 * 3600.0, fix_a=1.0, fix_delta=0.0)),
    ("no_wind_correction", dict(fix_a=1.0, fix_delta=0.0)),
    ("no_rotation", dict(fix_delta=0.0)),
    ("wrong_mass_basis", dict(mass_basis_wrong=True)),
    ("no_qa_screening", dict(no_qa=True)),
    ("clipped_negatives", dict(clip_negative=True)),
    ("nearest_cell_footprint", dict(nearest_cell=True)),
    ("no_diurnal_factor", dict(no_diurnal=True)),
    ("grid_convergence_ignored", dict(gamma_ignored=True)),
    ("grid_convergence_sign_flipped", dict(gamma_sign_flip=True)),
    ("first_order_upwind", dict(first_order=True)),
    ("coarse_time_step_300s", dict(dt=300.0)),
]

if __name__ == "__main__":
    only = sys.argv[1:] or [v[0] for v in VARIANTS]
    os.makedirs(OUTDIR, exist_ok=True)
    report = {}
    path = f"{ROOT}/authoring/evidence/baseline_results.json"
    if os.path.exists(path):
        report = json.load(open(path))
    for name, spec in VARIANTS:
        if name not in only:
            continue
        print(f"=== {name} ===", flush=True)
        import importlib
        importlib.reload(M)
        try:
            report[name] = fit(spec, name)
        except Exception as exc:            # a broken variant may simply fail
            report[name] = dict(label=name, error=repr(exc))
        print(json.dumps(report[name], indent=2), flush=True)
        json.dump(report, open(path, "w"), indent=2, sort_keys=True)
