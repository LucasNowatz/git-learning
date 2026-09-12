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
    pe = orc["param_err"]
    a(f"| domain-integrated road emission error | {100*orc['total_emission_rel_err_signed']:+.2f} % |")
    a(f"| wind rotation error | {pe['rot']:+.3f} deg |")
    a(f"| wind speed scale error | {100*pe['wind']:+.2f} % |")
    a(f"| vertical shape parameter error | {100*pe['zeta0']:+.2f} % |")
    a(f"| reference loss time error | {100*pe['tau0']:+.1f} % |")
    a(f"| saturation column error | {100*pe['c_ref']:+.1f} % |")
    a(f"| non-road scale error | {100*pe['fixed']:+.1f} % |")
    a("")
    a("The split in that table is the identifiability result. The wind")
    a("correction and the vertical shape parameter are recovered to a fraction")
    a("of a percent. The reference loss time, the saturation column and the")
    a("non-road scale are not: they lie along a degenerate direction in which")
    a("emission amplitude, loss time and saturation column trade against one")
    a("another with almost no change in the fit. That is why no gate compares")
    a("them, or the absolute road total, with the truth. Oracle wall time on")
    a("four cores is about twenty-three minutes, of which the artefact")
    a("screening loop is three passes of the full nonlinear fit.\n")

    a("## Deliberately broken variants\n")
    a("Each variant refits every free parameter with one link of the modelling")
    a("chain broken, predicts the withheld episodes with that same flawed model,")
    a("and is scored exactly as an agent would be.\n")
    a("| variant | sat WRMSE | sta WRMSE | consistency | train chi2/n | verdict |")
    a("| --- | --- | --- | --- | --- | --- |")
    a(f"| *reference solution* | {orc['wrmse_sat']:.3f} | {orc['wrmse_sta']:.3f} | "
      f"{max(orc['consistency_sat'], orc['consistency_sta']):.3f} | 0.960 | **pass** |")
    for k, v in sorted(base.items()):
        if "error" in v:
            a(f"| {k} | - | - | - | - | not run ({v['error'][:40]}) |")
            continue
        c = max(v["consistency_sat"], v["consistency_sta"])
        why = []
        if v["wrmse_sat"] > thr["skill"]["satellite"]:
            why.append("satellite skill")
        if v["wrmse_sta"] > thr["skill"]["station"]:
            why.append("station skill")
        if c > thr["consistency_max"]["satellite"]:
            why.append("consistency")
        for reg, rv in v["regime"].items():
            if rv["satellite"] > thr["regime"][reg]["satellite"] or \
               rv["station"] > thr["regime"][reg]["station"]:
                why.append(f"{reg} regime")
        verdict = ("rejected: " + ", ".join(sorted(set(why)))) if why else \
            "passes the graded gates"
        a(f"| {k} | {v['wrmse_sat']:.3f} | {v['wrmse_sta']:.3f} | {c:.3f} | "
          f"{v.get('chi2_per_obs', 0):.3f} | {verdict} |")
    a("")
    a("The training chi-square column is what the solver itself sees. It is the")
    a("only warning that anything is wrong, and it is loud for the artefact")
    a("(2.19 against 0.96) and quiet for everything else.\n")
    a("* `no_artefact_screening` is the central trap. Fitting the accepted")
    a("  retrievals as delivered costs a factor of three on held-out satellite")
    a("  skill. The specification says the quality flag is not exhaustive and")
    a("  that establishing which retrievals are fit to use is part of the")
    a("  analysis; it does not say which across-track positions or from when.")
    a("* `first_order_loss` pins the saturation column at the top of its range,")
    a("  which is what a solver that treats the sink as first order would have")
    a("  to report. It survives the global skill gates and is rejected by two of")
    a("  the three regime gates, which is what those gates are for.")
    a("* `fixed_sources_unscaled` and `vertical_shape_assumed` are **not**")
    a("  rejected. Assuming the non-road inventory is unbiased costs almost")
    a("  nothing, and assuming a plausible vertical shape costs little. Both are")
    a("  genuine identifiability limits, reported as such rather than claimed as")
    a("  traps.\n")
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
    a("There is no gate on the absolute road emission total, and none on the")
    a("individual chemistry parameters. The identifiability result above is why.\n")
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
