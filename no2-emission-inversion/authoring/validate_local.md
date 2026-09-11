# Local validation

## With Docker and harbor (the gates the pipeline runs)

From the task directory:

    harbor run -p . -a oracle -e docker     # must print reward 1
    harbor run -p . -a nop   -e docker      # must print reward 0
    harbor check . -m anthropic/claude-opus-4-8

Time the oracle run before submitting and confirm `[agent].timeout_sec` is
several times that. On four cores the reference solution takes about eleven
minutes of compute, and the agent budget is set to 28800 s.

## Without Docker (what was run while authoring this bundle)

The verifier is plain pytest, so its gates can be exercised directly:

    ln -sfn "$PWD/tests" /tests
    ln -sfn "$PWD/environment/data" /app/data
    pip install pytest==8.4.1 pytest-json-ctrf==0.3.5 numpy==2.2.6 \
                scipy==1.15.3 netCDF4==1.7.2

    python3 solution/no2_inversion.py       # writes the three artifacts to /app
    bash /tests/test.sh                     # expect /logs/verifier/reward.txt == 1

    rm -f /app/result.json /app/posterior.nc /app/predicted_observations.csv
    bash /tests/test.sh                     # expect reward 0

    python3 authoring/evidence/cheat_attempts.py   # every attempt must score 0

Re-run `bash /tests/test.sh` on the same artifacts several times: the verifier
has no random component and must give the same reward every time.

## Rebuilding the dataset

    cd authoring/provenance/generator
    python3 generate.py --out <bundle-root>
    python3 ../refresh_manifest.py

This overwrites `environment/data/` and `tests/truth/evaluation.npz` together.
Re-run the reference solution and `freeze_thresholds.py` afterwards; thresholds
are calibrated against a specific dataset realisation.
