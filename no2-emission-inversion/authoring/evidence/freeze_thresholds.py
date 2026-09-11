"""Derive tests/truth/thresholds.json from measured oracle and baseline scores.

One rule, applied everywhere, so nothing is hand-tuned to force an outcome:

    threshold = geometric mean of (measured reference score,
                                   best score reached by any variant that must fail)

A threshold therefore always sits the same factor above what a correct solution
achieves as it does below what a broken one achieves.  The script asserts that
factor is at least MIN_FACTOR on both sides before it writes anything.

Skill and regime gates are calibrated against variants whose *physics* is wrong.
The consistency gate is calibrated against variants whose *reported parameters
do not generate their own predictions*, with correct-physics variants that
merely use a different discretisation kept on the passing side.
"""
import json, sys
import numpy as np

ROOT = "/home/user/git-learning/no2-emission-inversion"
MIN_FACTOR = 1.45
REGIME_MIN_FACTOR = 1.15

# Variants whose physics is wrong. first_order_upwind and coarse_time_step_300s
# are excluded: model_spec.md already tells a solver that a first-order scheme
# and a time step above 60 s are inadequate on this grid, so the thresholds do
# not need to be tightened to catch them (both fail anyway on the station gate).
SKILL_MUST_FAIL = ["no_qa_screening", "no_rotation", "no_wind_correction",
                   "prior_unadjusted"]
CONSISTENCY_MUST_FAIL = ["no_diurnal_factor"]
# Correct physics, coarser numerics, and a footprint approximation that is
# defensible at this grid spacing: these must stay on the passing side.
# first_order_upwind is deliberately not listed: model_spec.md states that a
# first-order scheme on the 4 km grid is inadequate, so it is free to fail
# either gate.
# Correct or defensible operators whose predictions must still be recognised as
# coming from their own reported parameters.
CONSISTENCY_MUST_PASS = ["coarse_time_step_300s", "nearest_cell_footprint",
                         "grid_convergence_ignored", "grid_convergence_sign_flipped"]
TOTAL_MARGIN = 2.0


def geo(lo, hi, what):
    t = float(np.round(np.sqrt(lo * hi), 3))
    assert t / lo >= MIN_FACTOR, f"{what}: only {t/lo:.2f}x above the reference score"
    assert hi / t >= MIN_FACTOR, f"{what}: only {hi/t:.2f}x below the failing score"
    return t


def main():
    base = json.load(open(f"{ROOT}/authoring/evidence/baseline_results.json"))
    orc = json.load(open(f"{ROOT}/authoring/evidence/oracle_score.json"))
    fails = {k: base[k] for k in SKILL_MUST_FAIL if k in base and "error" not in base[k]}
    assert len(fails) == len(SKILL_MUST_FAIL), "a skill baseline is missing"

    thr_sat = geo(orc["wrmse_sat"], min(v["wrmse_sat"] for v in fails.values()), "satellite")
    thr_sta = geo(orc["wrmse_sta"], min(v["wrmse_sta"] for v in fails.values()), "station")

    # Regime gates are a safety net against a good average hiding a failed
    # regime, not a second skill gate. They carry the same relative margin over
    # the reference as the global gate, rescaled to each regime's own reference
    # level, and are only required to stay below the failing scores.
    regime = {}
    for reg, v in orc["regime"].items():
        rs = float(np.round(thr_sat * v["satellite"] / orc["wrmse_sat"], 3))
        rt = float(np.round(thr_sta * v["station"] / orc["wrmse_sta"], 3))
        fs = min(b["regime"][reg]["satellite"] for b in fails.values())
        ft = min(b["regime"][reg]["station"] for b in fails.values())
        assert fs / rs >= REGIME_MIN_FACTOR, f"{reg} satellite: only {fs/rs:.2f}x below failing"
        assert ft / rt >= REGIME_MIN_FACTOR, f"{reg} station: only {ft/rt:.2f}x below failing"
        regime[reg] = dict(satellite=rs, station=rt)

    ok = [max(orc["consistency_sat"], orc["consistency_sta"])]
    ok += [max(base[k]["consistency_sat"], base[k]["consistency_sta"])
           for k in CONSISTENCY_MUST_PASS if k in base and "error" not in base[k]]
    bad = [max(base[k]["consistency_sat"], base[k]["consistency_sta"])
           for k in CONSISTENCY_MUST_FAIL if k in base and "error" not in base[k]]
    cons = geo(max(ok), min(bad), "consistency")
    tot = float(np.round(TOTAL_MARGIN * orc["total_emission_rel_err_max"], 3))

    out = {
        "_comment": ("Frozen by authoring/evidence/freeze_thresholds.py from "
                     "measured reference and baseline scores, before any agent "
                     "run. Each global gate is the geometric mean of what the "
                     "reference solution achieves and the best score reached by "
                     "a deliberately broken variant; each regime gate is the "
                     "global gate rescaled by the reference score in that "
                     "regime. Do not edit by hand."),
        "_calibration": {
            "rule": "geometric mean of reference score and best failing score",
            "min_factor_each_side": MIN_FACTOR,
            "regime_min_factor_below_failing": REGIME_MIN_FACTOR,
            "regime_rule": ("global threshold rescaled by the reference score in "
                            "that regime"),
            "reference": {"satellite": round(orc["wrmse_sat"], 4),
                          "station": round(orc["wrmse_sta"], 4),
                          "consistency": [round(orc["consistency_sat"], 4),
                                          round(orc["consistency_sta"], 4)],
                          "total_emission_rel_err": round(orc["total_emission_rel_err_max"], 4)},
            "skill_must_fail": {k: {"satellite": round(v["wrmse_sat"], 4),
                                    "station": round(v["wrmse_sta"], 4)}
                                for k, v in fails.items()},
            "consistency_must_pass": {k: round(max(base[k]["consistency_sat"],
                                                   base[k]["consistency_sta"]), 4)
                                      for k in CONSISTENCY_MUST_PASS if k in base},
            "consistency_must_fail": {k: round(max(base[k]["consistency_sat"],
                                                   base[k]["consistency_sta"]), 4)
                                      for k in CONSISTENCY_MUST_FAIL if k in base},
            "total_emission_margin": TOTAL_MARGIN,
        },
        "bounds": {
            "emission_scale": [0.30, 2.50],
            "effective_lifetime_s": [3600.0, 28800.0],
            "wind_speed_scale": [0.70, 1.30],
            "wind_rotation_deg": [-20.0, 20.0],
        },
        "consistency_max": {"satellite": cons, "station": cons},
        "skill": {"satellite": thr_sat, "station": thr_sta},
        "regime": regime,
        "total_emission_rel_tol": tot,
    }
    with open(f"{ROOT}/tests/truth/thresholds.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("_")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
