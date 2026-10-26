#!/usr/bin/env bash
# run3d.sh: build the L3D CLI, then run every tests/cases3d/*.txt row against the numpy oracle.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
mkdir -p .bin .work
rm -f .bin/mojo-cv3d
if ! log=$(.venv/bin/mojo build -I src src/mojocv/cli3d.mojo -o .bin/mojo-cv3d 2>&1); then
  echo "$log" | grep -E "error" | head -20
  echo "FAIL build: mojo build"; exit 1
fi
test -x .bin/mojo-cv3d || { echo "FAIL build: no binary"; exit 1; }
PYTHONDONTWRITEBYTECODE=1 "$PY" -I tools/check3d.py tests/cases3d/*.txt
