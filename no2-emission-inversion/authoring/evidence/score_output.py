"""Score a candidate /app output triple against the sealed evaluation set."""
import csv, json, sys, os
import numpy as np
sys.path.insert(0, "/home/user/git-learning/no2-emission-inversion/tests")
import verifier_forward as VF

ROOT = "/home/user/git-learning/no2-emission-inversion"
EV = np.load(f"{ROOT}/tests/truth/evaluation.npz", allow_pickle=False)
I = VF.read_inputs(f"{ROOT}/tests/inputs")
QSAT, QSTA = VF.read_queries(f"{ROOT}/tests/inputs")
NX, NY = 60, 45


def wrmse(p, t, s):
    return float(np.sqrt(np.mean(((p - t) / s) ** 2)))


def score(app="/app", recompute=True, label=""):
    r = json.load(open(f"{app}/result.json"))
    rid = np.asarray(r["region_ids"], int)
    order = np.argsort(rid)
    P = dict(emission_scale=np.asarray(r["emission_scale"], float)[order],
             effective_lifetime_s=float(r["effective_lifetime_s"]),
             wind_speed_scale=float(r["wind_speed_scale"]),
             wind_rotation_deg=float(r["wind_rotation_deg"]),
             background_coefficients=np.asarray(r["background_coefficients"], float))
    got = {}
    with open(f"{app}/predicted_observations.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            got[row["obs_id"].strip()] = float(row["predicted_value"])
    ps = np.array([got[o] for o in QSAT["obs_id"]])
    pt = np.array([got[o] for o in QSTA["obs_id"]])
    ss = np.hypot(EV["sat_sigma_meas"], float(EV["sigma_repr_sat"]))
    st = np.hypot(EV["sta_sigma_meas"], float(EV["sigma_repr_sta"]))
    out = dict(label=label,
               params=dict(emission_scale=P["emission_scale"].round(5).tolist(),
                           lifetime_h=round(P["effective_lifetime_s"] / 3600, 4),
                           wind_speed_scale=round(P["wind_speed_scale"], 5),
                           wind_rotation_deg=round(P["wind_rotation_deg"], 4),
                           background=P["background_coefficients"].tolist()),
               wrmse_sat=wrmse(ps, EV["sat_truth"], ss),
               wrmse_sta=wrmse(pt, EV["sta_truth"], st),
               regime={})
    for reg in sorted(set(EV["sat_regime"].tolist())):
        ms = EV["sat_regime"] == reg
        mt = EV["sta_regime"] == reg
        out["regime"][reg] = dict(
            satellite=wrmse(ps[ms], EV["sat_truth"][ms], ss[ms]),
            station=wrmse(pt[mt], EV["sta_truth"][mt], st[mt]))
    area = float(I["area"][0, 0])
    base = I["road"].sum(axis=(1, 2)) * area
    rep = r["integrated_road_mol_s"]
    rel = []
    for e in range(18):
        g = float(rep[str(e)] if str(e) in rep else rep[e])
        rel.append(abs(g - float(EV["true_total_mol_s"][e])) / float(EV["true_total_mol_s"][e]))
    out["total_emission_rel_err_max"] = float(max(rel))
    out["total_emission_rel_err_signed"] = float(
        (np.asarray(P["emission_scale"]) * base).sum()
        / (EV["true_source_scale"] * base).sum() - 1.0)
    if recompute:
        rs, rt = VF.recompute_all(I, QSAT, QSTA, P, NX, NY)
        qs = np.hypot(QSAT["sigma"], float(EV["sigma_repr_sat"]))
        qt = np.hypot(QSTA["sigma"], float(EV["sigma_repr_sta"]))
        out["consistency_sat"] = float(np.sqrt(np.mean(((ps - rs) / qs) ** 2)))
        out["consistency_sta"] = float(np.sqrt(np.mean(((pt - rt) / qt) ** 2)))
        out["verifier_forward_wrmse_sat"] = wrmse(rs, EV["sat_truth"], ss)
        out["verifier_forward_wrmse_sta"] = wrmse(rt, EV["sta_truth"], st)
    return out


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1] if len(sys.argv) > 1 else "/app",
                           label="oracle"), indent=2))
