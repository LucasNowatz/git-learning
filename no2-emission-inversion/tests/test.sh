#!/usr/bin/env bash
# Sealed verifier entry point. Writes a reward on every path, including crashes.
set -u

mkdir -p /logs/verifier
printf '0' > /logs/verifier/reward.txt

cd /tests || exit 0

python -m pytest test_no2_inversion.py \
    -p no:cacheprovider \
    --ctrf /logs/verifier/ctrf-report.json \
    -rA --tb=short > /logs/verifier/pytest.log 2>&1
status=$?

cat /logs/verifier/pytest.log

if [ "${status}" -eq 0 ]; then
    printf '1' > /logs/verifier/reward.txt
fi

exit 0
