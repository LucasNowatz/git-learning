#!/usr/bin/env bash
# Reference solution. Runs in the agent's environment container.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd /app
python3 "${HERE}/no2_inversion.py"
ls -l /app/result.json /app/posterior.nc /app/predicted_observations.csv
