"""Sealed verifier for afterquery/no2-emission-inversion.

Every gate is mandatory.  The reward is 1 only when the whole module passes.
Agent artifacts are treated as untrusted input: only JSON, CSV and NetCDF are
parsed, nothing is executed, and array sizes are bounded before use.
"""
import csv, json, math, os
import numpy as np
import pytest
from netCDF4 import Dataset

import verifier_forward as VF

APP = "/app"
INPUTS = "/tests/inputs"
TRUTH = "/tests/truth"
NX, NY = 60, 45
NREG = 6
N_EPISODES = 18
MAX_BYTES = 64 * 1024 * 1024
SAT_UNITS = {"mol m-2", "mol/m2", "mol m^-2"}
STA_UNITS = {"ug m-3", "ug/m3", "ug m^-3", "µg m-3", "µg/m3"}


def _size_ok(path):
    return os.path.isfile(path) and 0 < os.path.getsize(path) <= MAX_BYTES


@pytest.fixture(scope="module")
def ctx():
    thr = json.load(open(os.path.join(TRUTH, "thresholds.json")))
    ev = np.load(os.path.join(TRUTH, "evaluation.npz"), allow_pickle=False)
    I = VF.read_inputs(INPUTS)
    qsat, qsta = VF.read_queries(INPUTS)
    return dict(thr=thr, ev=ev, I=I, qsat=qsat, qsta=qsta, state={})


# ------------------------------------------------------------------ gate 1 --
def test_result_json_schema(ctx):
    p = os.path.join(APP, "result.json")
    assert _size_ok(p), "result.json missing or empty"
    with open(p) as fh:
        r = json.load(fh)
    assert isinstance(r, dict)
    need = ["region_ids", "emission_scale", "effective_lifetime_s",
            "wind_speed_scale", "wind_rotation_deg", "background_coefficients",
            "integrated_road_mol_s"]
    for k in need:
        assert k in r, f"result.json is missing required key {k!r}"
    rid = [int(v) for v in r["region_ids"]]
    assert sorted(rid) == list(range(1, NREG + 1)), "region_ids must be 1..6"
    es = np.asarray(r["emission_scale"], dtype=float)
    assert es.shape == (NREG,), "emission_scale must hold six values"
    assert np.all(np.isfinite(es))
    for k in ("effective_lifetime_s", "wind_speed_scale", "wind_rotation_deg"):
        assert np.isfinite(float(r[k])), f"{k} must be finite"
    bg = np.asarray(r["background_coefficients"], dtype=float)
    assert bg.shape == (3,) and np.all(np.isfinite(bg))
    tot = r["integrated_road_mol_s"]
    assert isinstance(tot, dict), "integrated_road_mol_s must be a mapping"
    keys = {int(k) for k in tot}
    assert keys == set(range(N_EPISODES)), "integrated_road_mol_s must cover all 18 episodes"
    assert all(np.isfinite(float(v)) for v in tot.values())

    # region_ids may be reordered; map the reported scales back to region order
    order = np.argsort(np.asarray(rid))
    ctx["state"]["result"] = r
    ctx["state"]["scale"] = es[order]
    ctx["state"]["params"] = dict(
        emission_scale=es[order], effective_lifetime_s=float(r["effective_lifetime_s"]),
        wind_speed_scale=float(r["wind_speed_scale"]),
        wind_rotation_deg=float(r["wind_rotation_deg"]),
        background_coefficients=bg)


def test_posterior_schema(ctx):
    p = os.path.join(APP, "posterior.nc")
    assert _size_ok(p), "posterior.nc missing or empty"
    with Dataset(p) as ds:
        for v in ("corrected_road_flux", "total_source_flux", "region_id", "x", "y"):
            assert v in ds.variables, f"posterior.nc is missing variable {v!r}"
        crf = np.asarray(ds["corrected_road_flux"][:], dtype=float)
        tsf = np.asarray(ds["total_source_flux"][:], dtype=float)
        assert crf.shape == (NREG, NY, NX), f"corrected_road_flux has shape {crf.shape}"
        assert tsf.shape == (NY, NX), f"total_source_flux has shape {tsf.shape}"
        assert np.all(np.isfinite(crf)) and np.all(np.isfinite(tsf))
        rid = np.asarray(ds["region_id"][:], dtype=int)
        assert sorted(rid.tolist()) == list(range(1, NREG + 1))
        x = np.asarray(ds["x"][:], dtype=float)
        y = np.asarray(ds["y"][:], dtype=float)
        assert np.allclose(x, ctx["I"]["x"], atol=1.0), "posterior x axis mismatch"
        assert np.allclose(y, ctx["I"]["y"], atol=1.0), "posterior y axis mismatch"
        for v in ("corrected_road_flux", "total_source_flux"):
            u = getattr(ds[v], "units", "").replace("^", "").strip()
            assert u in ("mol m-2 s-1", "mol/m2/s"), f"{v} declares units {u!r}"
    order = np.argsort(rid)
    ctx["state"]["crf"] = crf[order]
    ctx["state"]["tsf"] = tsf


def test_predictions_schema(ctx):
    p = os.path.join(APP, "predicted_observations.csv")
    assert _size_ok(p), "predicted_observations.csv missing or empty"
    want_sat = set(ctx["qsat"]["obs_id"].tolist())
    want_sta = set(ctx["qsta"]["obs_id"].tolist())
    got = {}
    with open(p, newline="") as fh:
        rd = csv.DictReader(fh)
        head = {(h or "").strip().lstrip("\ufeff"): h for h in (rd.fieldnames or [])}
        need = {"obs_id", "instrument_type", "predicted_value", "unit"}
        assert need <= set(head), \
            f"header must contain {sorted(need)}, found {rd.fieldnames}"
        for i, row in enumerate(rd):
            if i > 4 * (len(want_sat) + len(want_sta)):
                pytest.fail("predicted_observations.csv has far too many rows")
            cell = lambda k: (row.get(head[k]) or "").strip()
            oid = cell("obs_id")
            assert oid not in got, f"duplicate obs_id {oid!r}"
            kind = cell("instrument_type").lower()
            unit = cell("unit").lower()
            try:
                val = float(cell("predicted_value"))
            except ValueError:
                pytest.fail(f"predicted_value for {oid!r} is not a number: "
                            f"{cell('predicted_value')!r}")
            assert math.isfinite(val), f"non-finite prediction for {oid!r}"
            if oid in want_sat:
                assert kind == "satellite", f"{oid!r} declares instrument {kind!r}"
                assert unit in SAT_UNITS, f"{oid!r} declares unit {unit!r}"
            elif oid in want_sta:
                assert kind == "station", f"{oid!r} declares instrument {kind!r}"
                assert unit in STA_UNITS, f"{oid!r} declares unit {unit!r}"
            else:
                pytest.fail(f"unknown obs_id {oid!r} in predicted_observations.csv")
            got[oid] = val
    missing = (want_sat | want_sta) - set(got)
    assert not missing, f"{len(missing)} query ids have no prediction, e.g. {sorted(missing)[:3]}"
    ctx["state"]["pred_sat"] = np.array([got[o] for o in ctx["qsat"]["obs_id"]])
    ctx["state"]["pred_sta"] = np.array([got[o] for o in ctx["qsta"]["obs_id"]])


# ------------------------------------------------------------------ gate 2 --
def test_parameters_admissible(ctx):
    P = ctx["state"]["params"]
    b = ctx["thr"]["bounds"]
    s = P["emission_scale"]
    assert np.all(s >= b["emission_scale"][0] - 1e-9) and np.all(s <= b["emission_scale"][1] + 1e-9), \
        f"emission_scale outside the published range: {s}"
    tau = P["effective_lifetime_s"]
    assert b["effective_lifetime_s"][0] - 1e-6 <= tau <= b["effective_lifetime_s"][1] + 1e-6, \
        f"effective_lifetime_s = {tau} s is outside the published range"
    a = P["wind_speed_scale"]
    assert b["wind_speed_scale"][0] - 1e-9 <= a <= b["wind_speed_scale"][1] + 1e-9
    dd = P["wind_rotation_deg"]
    assert b["wind_rotation_deg"][0] - 1e-9 <= dd <= b["wind_rotation_deg"][1] + 1e-9
    b0, bx, by = P["background_coefficients"]
    corners = [b0, b0 + bx, b0 + by, b0 + bx + by]
    assert min(corners) >= -1e-12, \
        f"the background field is negative somewhere in the domain: corners {corners}"


def test_exported_fluxes_match_parameters(ctx):
    I = ctx["I"]
    s = ctx["state"]["params"]["emission_scale"]
    expect = s[:, None, None] * I["road"]
    got = ctx["state"]["crf"]
    scale = float(np.abs(expect).max())
    err = float(np.abs(got - expect).max()) / scale
    assert err <= 2e-3, (
        "corrected_road_flux does not equal emission_scale times the supplied "
        f"prior converted to mol m-2 s-1 (max relative deviation {err:.3e}); "
        "check the NO2-equivalent mass basis of road_prior")
    exp_tot = expect.sum(axis=0) + I["fixed"]
    err2 = float(np.abs(ctx["state"]["tsf"] - exp_tot).max()) / float(np.abs(exp_tot).max())
    assert err2 <= 2e-3, (
        "total_source_flux does not equal the corrected road flux plus the "
        f"prescribed non-road sources (max relative deviation {err2:.3e}); "
        "check the nitrogen mass basis of fixed_sources")


def test_reported_totals_match_parameters(ctx):
    I = ctx["I"]
    s = ctx["state"]["params"]["emission_scale"]
    area = float(I["area"][0, 0])
    base = I["road"].sum(axis=(1, 2)) * area
    rep = ctx["state"]["result"]["integrated_road_mol_s"]
    worst = 0.0
    for e in range(N_EPISODES):
        f = float(I["road_factor"][e]) * float(VF.diurnal_at(I["tsat"][e], I["diurnal"]))
        expect = float((s * base).sum() * f)
        got = float(rep[str(e)] if str(e) in rep else rep[e])
        worst = max(worst, abs(got - expect) / expect)
    assert worst <= 5e-3, (
        f"integrated_road_mol_s is inconsistent with the reported scale factors "
        f"(worst relative deviation {worst:.3e}); it must be the domain integral "
        "of the corrected road flux at the episode reference time")


# ------------------------------------------------------------------ gate 3 --
@pytest.fixture(scope="module")
def recomputed(ctx):
    return VF.recompute_all(ctx["I"], ctx["qsat"], ctx["qsta"],
                            ctx["state"]["params"], NX, NY)


def test_predictions_consistent_with_parameters(ctx, recomputed):
    ev = ctx["ev"]
    rs, rt = recomputed
    st = np.hypot(ctx["qsat"]["sigma"], float(ev["sigma_repr_sat"]))
    tt = np.hypot(ctx["qsta"]["sigma"], float(ev["sigma_repr_sta"]))
    ds = float(np.sqrt(np.mean(((ctx["state"]["pred_sat"] - rs) / st) ** 2)))
    dt = float(np.sqrt(np.mean(((ctx["state"]["pred_sta"] - rt) / tt) ** 2)))
    lim = ctx["thr"]["consistency_max"]
    assert ds <= lim["satellite"], (
        f"submitted satellite predictions disagree with an independent forward "
        f"run of the submitted parameters (normalised RMS {ds:.3f} > {lim['satellite']})")
    assert dt <= lim["station"], (
        f"submitted station predictions disagree with an independent forward "
        f"run of the submitted parameters (normalised RMS {dt:.3f} > {lim['station']})")


# ------------------------------------------------------------- gates 4 & 5 --
def _wrmse(pred, truth, sigma):
    return float(np.sqrt(np.mean(((pred - truth) / sigma) ** 2)))


def test_held_out_skill(ctx):
    ev = ctx["ev"]
    st = np.hypot(ev["sat_sigma_meas"], float(ev["sigma_repr_sat"]))
    tt = np.hypot(ev["sta_sigma_meas"], float(ev["sigma_repr_sta"]))
    ws = _wrmse(ctx["state"]["pred_sat"], ev["sat_truth"], st)
    wt = _wrmse(ctx["state"]["pred_sta"], ev["sta_truth"], tt)
    thr = ctx["thr"]["skill"]
    ctx["state"]["wrmse"] = (ws, wt)
    assert ws <= thr["satellite"], (
        f"held-out satellite WRMSE {ws:.4f} exceeds the frozen threshold "
        f"{thr['satellite']}")
    assert wt <= thr["station"], (
        f"held-out station WRMSE {wt:.4f} exceeds the frozen threshold "
        f"{thr['station']}")


def test_regime_robustness(ctx):
    ev = ctx["ev"]
    st = np.hypot(ev["sat_sigma_meas"], float(ev["sigma_repr_sat"]))
    tt = np.hypot(ev["sta_sigma_meas"], float(ev["sigma_repr_sta"]))
    thr = ctx["thr"]["regime"]
    bad = []
    for reg in sorted(set(ev["sat_regime"].tolist())):
        ms = ev["sat_regime"] == reg
        mt = ev["sta_regime"] == reg
        ws = _wrmse(ctx["state"]["pred_sat"][ms], ev["sat_truth"][ms], st[ms])
        wt = _wrmse(ctx["state"]["pred_sta"][mt], ev["sta_truth"][mt], tt[mt])
        if ws > thr[reg]["satellite"]:
            bad.append(f"{reg} satellite WRMSE {ws:.4f} > {thr[reg]['satellite']}")
        if wt > thr[reg]["station"]:
            bad.append(f"{reg} station WRMSE {wt:.4f} > {thr[reg]['station']}")
    assert not bad, "wind regimes failing their own threshold: " + "; ".join(bad)


# ------------------------------------------------------------------ gate 6 --
def test_identifiable_emission_recovery(ctx):
    ev = ctx["ev"]
    rep = ctx["state"]["result"]["integrated_road_mol_s"]
    true = ev["true_total_mol_s"]
    tol = ctx["thr"]["total_emission_rel_tol"]
    worst, where = 0.0, -1
    for e in range(N_EPISODES):
        got = float(rep[str(e)] if str(e) in rep else rep[e])
        rel = abs(got - float(true[e])) / float(true[e])
        if rel > worst:
            worst, where = rel, e
    assert worst <= tol, (
        f"domain-integrated road NOx emission is off by {100*worst:.2f} percent "
        f"at episode {where}; the tolerated deviation is {100*tol:.2f} percent")
