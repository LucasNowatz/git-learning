"""Render authoring/evidence/EVIDENCE.md from the measured JSON results."""
import json, os

ROOT = "/home/user/git-learning/no2-emission-inversion"
E = f"{ROOT}/authoring/evidence"


def main():
    orc = json.load(open(f"{E}/oracle_score.json"))
    base = json.load(open(f"{E}/baseline_results.json"))
    cheat = json.load(open(f"{E}/cheat_results.json"))
    thr = json.load(open(f"{ROOT}/tests/truth/thresholds.json"))
    L = []
    a = L.append
    a("# Measured evidence\n")
    a("Everything here was measured on this bundle before any frontier-agent run.")
    a("Regenerate with `score_output.py`, `baselines.py`, `cheat_attempts.py`,")
    a("`freeze_thresholds.py` and this script.\n")

    a("## Reference solution\n")
    a("| quantity | value |")
    a("| --- | --- |")
    a(f"| held-out satellite WRMSE | {orc['wrmse_sat']:.4f} |")
    a(f"| held-out station WRMSE | {orc['wrmse_sta']:.4f} |")
    for r, v in sorted(orc["regime"].items()):
        a(f"| {r} satellite / station WRMSE | {v['satellite']:.4f} / {v['station']:.4f} |")
    a(f"| prediction consistency, satellite / station | {orc['consistency_sat']:.4f} / "
      f"{orc['consistency_sta']:.4f} |")
    a(f"| domain-integrated road emission error | {100*orc['total_emission_rel_err_signed']:+.2f} % |")
    a(f"| recovered effective loss time | {orc['params']['lifetime_h']:.3f} h "
      f"(true 4.600 h) |")
    a(f"| recovered wind speed scale | {orc['params']['wind_speed_scale']:.4f} "
      f"(true 1.1800) |")
    a(f"| recovered wind rotation | {orc['params']['wind_rotation_deg']:.3f} deg "
      f"(true -13.500 deg) |")
    a("")
    a("Individual region scale factors are recovered to between 2 and 20 percent,")
    a("which is why only their domain total is graded. Oracle wall time on four")
    a("cores is about eleven minutes.\n")

    a("## Deliberately broken variants\n")
    a("Each variant refits every free parameter with one link of the modelling")
    a("chain broken, predicts the withheld episodes with that same flawed model,")
    a("and is scored exactly as an agent would be.\n")
    a("| variant | sat WRMSE | sta WRMSE | consistency | total err | train chi2/n | verdict |")
    a("| --- | --- | --- | --- | --- | --- | --- |")
    a(f"| *reference solution* | {orc['wrmse_sat']:.3f} | {orc['wrmse_sta']:.3f} | "
      f"{max(orc['consistency_sat'], orc['consistency_sta']):.3f} | "
      f"{orc['total_emission_rel_err_max']:.3f} | 0.964 | **pass** |")
    for k, v in sorted(base.items()):
        if "error" in v:
            a(f"| {k} | - | - | - | - | - | not run ({v['error'][:40]}) |")
            continue
        c = max(v["consistency_sat"], v["consistency_sta"])
        why = []
        if v["wrmse_sat"] > thr["skill"]["satellite"]:
            why.append("satellite skill")
        if v["wrmse_sta"] > thr["skill"]["station"]:
            why.append("station skill")
        if c > thr["consistency_max"]["satellite"]:
            why.append("consistency")
        if v["total_emission_rel_err_max"] > thr["total_emission_rel_tol"]:
            why.append("emission total")
        for reg, rv in v["regime"].items():
            if rv["satellite"] > thr["regime"][reg]["satellite"] or \
               rv["station"] > thr["regime"][reg]["station"]:
                why.append(f"{reg} regime")
        verdict = ("rejected: " + ", ".join(sorted(set(why)))) if why else \
            "passes the graded gates"
        a(f"| {k} | {v['wrmse_sat']:.3f} | {v['wrmse_sta']:.3f} | {c:.3f} | "
          f"{v['total_emission_rel_err_max']:.3f} | {v.get('chi2_per_obs', 0):.3f} | {verdict} |")
    a("")
    a("Two variants deserve a note. `wrong_mass_basis` is not caught by the skill")
    a("gates, because the free parameters absorb most of a mis-scaled non-road")
    a("source term; it is caught by the exported-inventory gate, which recomputes")
    a("both unit conversions itself. `clipped_negatives` is genuinely harmless at")
    a("this signal level and is reported as such rather than claimed as a trap.\n")

    a("## Adversarial attempts\n")
    a("Each entry is the laziest output that might satisfy a weak verifier.\n")
    a("| attempt | reward | rejected by |")
    a("| --- | --- | --- |")
    for k, v in sorted(cheat.items()):
        if k.startswith("_"):
            continue
        why = v["rejected_because"].split("AssertionError:")[-1].split("Failed:")[-1]
        a(f"| {k} | {v['reward']} | {why.strip()[:110]} |")
    a(f"| *reference solution* | {cheat['_reference_solution']['reward']} | - |")
    a("")
    a("## Frozen thresholds\n")
    a("```json")
    a(json.dumps({k: v for k, v in thr.items() if not k.startswith("_")}, indent=2,
                 sort_keys=True))
    a("```")
    a("")
    a(thr["_comment"])
    open(f"{E}/EVIDENCE.md", "w").write("\n".join(L) + "\n")
    print("wrote EVIDENCE.md")


if __name__ == "__main__":
    main()
