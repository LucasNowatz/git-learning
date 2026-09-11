#!/usr/bin/env bash
# Build the submission zip: task directory CONTENTS at the top level, no
# enclosing folder, no caches, no zip of this script's own output.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-/tmp/no2-emission-inversion.zip}"
cd "$HERE"
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf tests/.pytest_cache .pytest_cache
rm -f "$OUT"
zip -rq "$OUT" . -x '*.pyc' -x '*__pycache__*' -x '*.pytest_cache*' -x '.git/*'
echo "wrote $OUT"
unzip -l "$OUT" | tail -3
