"""Derive tests/truth/thresholds.json from measured reference and baseline scores.

Thresholds are set as a fixed margin above what the reference solution
achieves, and are then *validated* against the broken variants: every variant
that must fail has to be rejected by at least one gate. Setting the margin from
the reference and checking it against the broken set, rather than placing the
threshold between the two, keeps a correct solution with different numerics
safely inside even where a broken variant lands close.

The domain-integrated emission gate is deliberately absent. The identifiability
study in EVIDENCE.md shows the public data do not constrain the absolute road
total: the saturating sink lets emission amplitude, reference loss time and
saturation column trade against one another with almost no change in the fit.
Grading a quantity the data do not determine would fail correct work, so
prediction agreement carries the evidence, as the design requires.
"""
import json, sys
import numpy as np

ROOT = "/home/user/git-learning/no2-emission-inversion"

MARGIN_SAT = 1.75
MARGIN_STA = 1.45
MARGIN_CONS = 5.5

MUST_FAIL = ["no_artefact_screening", "prior_unadjusted", "no_wind_correction",
             "first_order_loss"]
# Variants that the gates do not reject. Each is a genuine identifiability
# limit rather than a hidden trap, and EVIDENCE.md says so.
KNOWN_PASSING = ["fixed_sources_unscaled", "vertical_shape_assumed"]


def main():
    base = json.load(open(f"{ROOT}/authoring/evidence/baseline_results.json"))
    orc = json.load(open(f"{ROOT}/authoring/evidence/oracle_score.json"))

    thr_sat = float(np.round(MARGIN_SAT * orc["wrmse_sat"], 3))
    thr_sta = float(np.round(MARGIN_STA * orc["wrmse_sta"], 3))
    cons = float(np.round(MARGIN_CONS * max(orc["consistency_sat"],
                                            orc["consistency_sta"]), 3))
    regime = {}
    for reg, v in orc["regime"].items():
        regime[reg] = dict(
            satellite=float(np.round(MARGIN_SAT * v["satellite"], 3)),
            station=float(np.round(MARGIN_STA * v["station"], 3)))

    def rejected(v):
        why = []
        if v["wrmse_sat"] > thr_sat:
            why.append("satellite skill")
        if v["wrmse_sta"] > thr_sta:
            why.append("station skill")
        if max(v["consistency_sat"], v["consistency_sta"]) > cons:
            why.append("consistency")
        for reg, rv in v["regime"].items():
            if rv["satellite"] > regime[reg]["satellite"] or \
               rv["station"] > regime[reg]["station"]:
                why.append(f"{reg} regime")
        return sorted(set(why))

    for k in MUST_FAIL:
        assert k in base and "error" not in base[k], f"{k} was not measured"
        why = rejected(base[k])
        assert why, f"{k} passes every gate; the thresholds are too loose"
        print(f"  {k:26s} rejected by {', '.join(why)}")
    assert not rejected(orc), "the reference solution would not pass its own gates"
    for k in KNOWN_PASSING:
        if k in base and "error" not in base[k] and rejected(base[k]):
            print(f"  note: {k} is now rejected by {', '.join(rejected(base[k]))}")

    out = {
        "_comment": ("Frozen by authoring/evidence/freeze_thresholds.py from "
                     "measured reference performance, before any agent run. "
                     "Each gate is a fixed margin above what the reference "
                     "solution achieves, validated against the broken variants "
                     "in EVIDENCE.md. Do not edit by hand."),
        "_calibration": {
            "rule": "fixed margin above the reference score, validated against broken variants",
            "margins": dict(satellite=MARGIN_SAT, station=MARGIN_STA,
                            consistency=MARGIN_CONS),
            "reference": {"satellite": round(orc["wrmse_sat"], 4),
                          "station": round(orc["wrmse_sta"], 4),
                          "consistency": [round(orc["consistency_sat"], 4),
                                          round(orc["consistency_sta"], 4)],
                          "regime": {k: {i: round(x, 4) for i, x in v.items()}
                                     for k, v in orc["regime"].items()}},
            "must_fail": {k: {"satellite": round(base[k]["wrmse_sat"], 4),
                              "station": round(base[k]["wrmse_sta"], 4),
                              "rejected_by": rejected(base[k])}
                          for k in MUST_FAIL},
            "known_passing": {k: {"satellite": round(base[k]["wrmse_sat"], 4),
                                  "station": round(base[k]["wrmse_sta"], 4)}
                              for k in KNOWN_PASSING if k in base},
            "emission_total_gate": ("omitted: the absolute road total is not "
                                    "identifiable from the public data"),
        },
        "bounds": {
            "emission_scale": [0.30, 2.50],
            "reference_loss_time_s": [1800.0, 28800.0],
            "loss_saturation_column": [1.0e-5, 1.0e-3],
            "wind_speed_scale": [0.70, 1.30],
            "wind_rotation_deg": [-20.0, 20.0],
            "fixed_source_scale": [0.40, 2.20],
            "vertical_shape_zeta0": [0.12, 0.80],
        },
        "consistency_max": {"satellite": cons, "station": cons},
        "skill": {"satellite": thr_sat, "station": thr_sta},
        "regime": regime,
    }
    with open(f"{ROOT}/tests/truth/thresholds.json", "w") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("_")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
