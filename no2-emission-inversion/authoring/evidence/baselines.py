"""Refit deliberately flawed variants of the inversion and score them.

Each variant breaks exactly one link of the modelling chain, refits every
remaining free parameter with that flaw in place, predicts the withheld
episodes with the same flawed model, and is scored as an agent would be.
This is the evidence behind the frozen thresholds.
"""
import json, os, sys, time
import numpy as np

ROOT = "/home/user/git-learning/no2-emission-inversion"
sys.path.insert(0, f"{ROOT}/solution")
sys.path.insert(0, f"{ROOT}/authoring/evidence")
import no2_inversion as M
import score_output as SC
from netCDF4 import Dataset
from scipy.optimize import least_squares
import csv

OUTDIR = "/tmp/baselines2"


def run_variant(name, spec):
    t0 = time.time()
    d = M.load_inputs()
    model = M.Model(d)
    blocks, _ = M.load_training(d, model)
    M._MODEL, M._BLOCKS = model, blocks
    import multiprocessing as mp
    M._POOL = mp.get_context("fork").Pool(4)
    eps = sorted(blocks)
    mask = {e: np.ones(len(blocks[e]["sat"]["value"]), bool) for e in eps}

    lo, hi = M.LO.copy(), M.HI.copy()
    z0 = np.array([1.0] * 6 + [np.log(9000.0), np.log(1.5e-4), 1.0, 0.0,
                              2.0, 0.0, 0.0, 1.0, 0.35])
    for idx, val in spec.get("freeze", {}).items():
        lo[idx] = val - 1e-9
        hi[idx] = val + 1e-9
        z0[idx] = val

    def fit(z, msk, maxfev=900):
        return least_squares(M.residuals, z, args=(eps, msk), bounds=(lo, hi),
                             method="trf", x_scale="jac", diff_step=6e-3,
                             xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=maxfev)

    r = fit(z0, mask)
    if spec.get("screen", True):
        for _ in range(3):
            flags = M.detect_row_anomaly(r.x, eps, mask)
            tot = 0
            for e in eps:
                if flags[e]:
                    drop = np.isin(blocks[e]["sat"]["gp"], flags[e]) & mask[e]
                    mask[e] = mask[e] & ~drop
                    tot += int(drop.sum())
            if tot == 0:
                break
            r = fit(r.x, mask)
    chi2 = 2 * r.cost / len(r.fun)
    P = M.unpack(r.x)

    out = os.path.join(OUTDIR, name)
    os.makedirs(out, exist_ok=True)
    old_out = M.OUT
    M.OUT = out
    try:
        M.export(d, model, P)
    finally:
        M.OUT = old_out
    M._POOL.close(); M._POOL.join(); M._POOL = None

    res = SC.score(out, label=name)
    res["chi2_per_obs"] = float(chi2)
    res["n_retained"] = int(sum(mask[e].sum() for e in eps))
    res["fit_seconds"] = round(time.time() - t0, 1)
    return res


# index map: 0-5 scales, 6 log tau0, 7 log c_ref, 8 a, 9 delta/10,
#            10-12 background, 13 fixed_scale, 14 zeta0
VARIANTS = [
    ("no_artefact_screening", dict(screen=False)),
    ("first_order_loss", dict(freeze={7: float(np.log(1.0e-3))})),
    ("fixed_sources_unscaled", dict(freeze={13: 1.0})),
    ("vertical_shape_assumed", dict(freeze={14: 0.35})),
    ("no_wind_correction", dict(freeze={8: 1.0, 9: 0.0})),
    ("prior_unadjusted", dict(freeze={i: 1.0 for i in range(6)})),
]

if __name__ == "__main__":
    only = sys.argv[1:] or [v[0] for v in VARIANTS]
    os.makedirs(OUTDIR, exist_ok=True)
    path = f"{ROOT}/authoring/evidence/baseline_results.json"
    report = json.load(open(path)) if os.path.exists(path) else {}
    for name, spec in VARIANTS:
        if name not in only:
            continue
        print(f"=== {name} ===", flush=True)
        try:
            report[name] = run_variant(name, spec)
        except Exception as exc:
            report[name] = dict(label=name, error=repr(exc))
        r = report[name]
        if "error" in r:
            print("  ERROR", r["error"][:120], flush=True)
        else:
            print("  sat %.4f sta %.4f cons %.4f chi2/n %.3f kept %d (%.0fs)" % (
                r["wrmse_sat"], r["wrmse_sta"],
                max(r["consistency_sat"], r["consistency_sta"]),
                r["chi2_per_obs"], r["n_retained"], r["fit_seconds"]), flush=True)
        json.dump(report, open(path, "w"), indent=2, sort_keys=True)
