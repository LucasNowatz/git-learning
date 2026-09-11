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

SKILL_MUST_FAIL = ["no_qa_screening", "no_rotation", "no_wind_correction"]
CONSISTENCY_MUST_FAIL = ["no_diurnal_factor"]
CONSISTENCY_MUST_PASS = ["coarse_time_step_300s", "nearest_cell_footprint",
                         "first_order_upwind"]
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

    regime = {}
    for reg, v in orc["regime"].items():
        regime[reg] = dict(
            satellite=geo(v["satellite"],
                          min(b["regime"][reg]["satellite"] for b in fails.values()),
                          f"{reg} satellite"),
            station=geo(v["station"],
                        min(b["regime"][reg]["station"] for b in fails.values()),
                        f"{reg} station"))

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
                     "run. Every value is the geometric mean of what the "
                     "reference solution achieves and the best score reached by "
                     "a deliberately broken variant. Do not edit by hand."),
        "_calibration": {
            "rule": "geometric mean of reference score and best failing score",
            "min_factor_each_side": MIN_FACTOR,
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
