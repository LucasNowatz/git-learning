"""Adversarial attempts against the sealed verifier.

Each attempt is the laziest thing that might pass.  None is ever executed by
the pipeline; this script exists so the author can show that each one is
rejected, and for which reason.  Run it with the bundle's tests/ mounted at
/tests and a writable /app.
"""
import csv, json, os, shutil, subprocess, sys, tempfile
import numpy as np
from netCDF4 import Dataset

ROOT = "/home/user/git-learning/no2-emission-inversion"
GOOD = "/tmp/oracle_out"
APP = "/app"
DATA = f"{ROOT}/environment/data"


def install(result=None, posterior=None, predictions=None, base=GOOD):
    for f in ("result.json", "posterior.nc", "predicted_observations.csv"):
        p = os.path.join(APP, f)
        if os.path.exists(p):
            os.remove(p)
    if result is not False:
        shutil.copy(f"{base}/result.json", f"{APP}/result.json")
        if callable(result):
            r = json.load(open(f"{APP}/result.json"))
            json.dump(result(r), open(f"{APP}/result.json", "w"), indent=2)
    if posterior is not False:
        shutil.copy(f"{base}/posterior.nc", f"{APP}/posterior.nc")
        if callable(posterior):
            posterior(f"{APP}/posterior.nc")
    if predictions is not False:
        shutil.copy(f"{base}/predicted_observations.csv", f"{APP}/predicted_observations.csv")
        if callable(predictions):
            rows = list(csv.DictReader(open(f"{APP}/predicted_observations.csv")))
            rows = predictions(rows)
            with open(f"{APP}/predicted_observations.csv", "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["obs_id", "instrument_type", "predicted_value", "unit"])
                for r in rows:
                    w.writerow([r["obs_id"], r["instrument_type"],
                                r["predicted_value"], r["unit"]])


def run():
    p = subprocess.run(["bash", "/tests/test.sh"], capture_output=True, text=True)
    reward = open("/logs/verifier/reward.txt").read().strip()
    first = ""
    for line in p.stdout.splitlines():
        if line.startswith("E  ") or "AssertionError" in line or line.startswith("FAILED"):
            first = line.strip()[:160]
            break
    return reward, first


# --------------------------------------------------------------- attempts ---
def zeros(rows):
    for r in rows:
        r["predicted_value"] = "0.0"
    return rows


def constant(rows):
    for r in rows:
        r["predicted_value"] = "1.1e-05" if r["instrument_type"] == "satellite" else "9.0"
    return rows


def training_mean(rows):
    vals = []
    for fn in sorted(os.listdir(f"{DATA}/swaths")):
        with Dataset(f"{DATA}/swaths/{fn}") as ds:
            c = ds["no2_column"][:]
            qa = ds["qa_value"][:].data
            g = (~np.ma.getmaskarray(c)) & (qa >= 0.75)
            vals.append(np.asarray(c)[g])
    msat = float(np.concatenate(vals).mean())
    sta = [float(r["no2_ug_m3"]) for r in csv.DictReader(open(f"{DATA}/stations.csv"))]
    msta = float(np.mean(sta))
    for r in rows:
        r["predicted_value"] = repr(msat if r["instrument_type"] == "satellite" else msta)
    return rows


def nan_values(rows):
    for i, r in enumerate(rows):
        if i % 97 == 0:
            r["predicted_value"] = "NaN"
    return rows


def drop_rows(rows):
    return rows[:-50]


def duplicate_rows(rows):
    return rows + rows[:5]


def foreign_ids(rows):
    for r in rows[:3]:
        r["obs_id"] = r["obs_id"] + "-X"
    return rows


def scales_at_bound(r):
    r["emission_scale"] = [2.5] * 6
    return r


def garbage_parameters(r):
    r["effective_lifetime_s"] = 9000.0
    r["wind_speed_scale"] = 0.9
    r["wind_rotation_deg"] = 4.0
    return r


def flip_rotation_sign(r):
    r["wind_rotation_deg"] = -float(r["wind_rotation_deg"])
    return r


def negative_background(r):
    r["background_coefficients"] = [1.0e-6, -8.0e-6, 3.0e-6]
    return r


def wrong_mass_basis_posterior(path):
    with Dataset(path, "a") as ds:
        v = ds["corrected_road_flux"]
        v[:] = np.asarray(v[:]) * (14.0067 / 46.0055)
        t = ds["total_source_flux"]
        t[:] = np.asarray(t[:]) * (14.0067 / 46.0055)


def totals_without_diurnal(r):
    with Dataset(f"{DATA}/meteorology.nc") as ds:
        tsat = np.asarray(ds["overpass_time"][:])
    with Dataset(f"{DATA}/emissions.nc") as ds:
        tab = np.asarray(ds["road_diurnal_factor"][:])
    h = (tsat % 86400.0) / 3600.0
    i = np.floor(h).astype(int) % 24
    w = h - np.floor(h)
    d = tab[i] * (1 - w) + tab[(i + 1) % 24] * w
    r["integrated_road_mol_s"] = {k: float(v) / float(d[int(k)])
                                  for k, v in r["integrated_road_mol_s"].items()}
    return r


def emission_scale_doubled(r):
    r["emission_scale"] = [min(2.5, 2.0 * v) for v in r["emission_scale"]]
    return r


ATTEMPTS = [
    ("nothing_written", dict(result=False, posterior=False, predictions=False)),
    ("zero_predictions", dict(predictions=zeros)),
    ("constant_predictions", dict(predictions=constant)),
    ("training_mean_predictions", dict(predictions=training_mean)),
    ("nan_predictions", dict(predictions=nan_values)),
    ("missing_rows", dict(predictions=drop_rows)),
    ("duplicate_rows", dict(predictions=duplicate_rows)),
    ("foreign_obs_ids", dict(predictions=foreign_ids)),
    ("no_posterior_file", dict(posterior=False)),
    ("scales_pinned_at_bound", dict(result=scales_at_bound)),
    ("garbage_nonlinear_parameters", dict(result=garbage_parameters)),
    ("reported_rotation_sign_flipped", dict(result=flip_rotation_sign)),
    ("negative_background_field", dict(result=negative_background)),
    ("posterior_on_wrong_mass_basis", dict(posterior=wrong_mass_basis_posterior)),
    ("totals_missing_diurnal_factor", dict(result=totals_without_diurnal)),
    ("emission_scale_doubled", dict(result=emission_scale_doubled)),
]

if __name__ == "__main__":
    out = {}
    for name, spec in ATTEMPTS:
        install(**spec)
        reward, why = run()
        out[name] = dict(reward=int(reward), rejected_because=why)
        print(f"{name:34s} reward={reward}  {why}", flush=True)
    install()
    reward, _ = run()
    out["_reference_solution"] = dict(reward=int(reward))
    print(f"{'reference solution':34s} reward={reward}")
    json.dump(out, open(f"{ROOT}/authoring/evidence/cheat_results.json", "w"),
              indent=2, sort_keys=True)
