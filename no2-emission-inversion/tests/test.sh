#!/usr/bin/env bash
# Sealed verifier entry point.
# Every path, including a crash inside pytest, leaves a reward and a CTRF report.
set -u

REPORT=/logs/verifier/ctrf.json
mkdir -p /logs/verifier
printf '0' > /logs/verifier/reward.txt

cd /tests || exit 0

python -m pytest test_no2_inversion.py \
    -p no:cacheprovider \
    --ctrf "${REPORT}" \
    -rA --tb=short > /logs/verifier/pytest.log 2>&1
status=$?

cat /logs/verifier/pytest.log

if [ ! -s "${REPORT}" ]; then
    printf '{"results":{"tool":{"name":"pytest"},"summary":{"tests":1,"passed":0,"failed":1,"skipped":0,"pending":0,"other":0,"start":0,"stop":0},"tests":[{"name":"test_no2_inversion","status":"failed","duration":0,"message":"the verifier terminated without producing a report"}]}}' > "${REPORT}"
fi

if [ "${status}" -eq 0 ]; then
    printf '1' > /logs/verifier/reward.txt
fi

exit 0
