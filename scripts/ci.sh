#!/usr/bin/env bash
# CI entry point, same on a GitHub runner and a local Ubuntu VM: install, build, then every CPU gate.
# The speed table (docs/bench.md) and the real-photo rows (tests/cases/real.txt) are run by hand, see README.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v uv >/dev/null || { echo "FAIL setup: uv not installed"; exit 1; }
uv sync || { echo "FAIL setup: uv sync"; exit 1; }
python3 -c 'import platform; print("oracle python", platform.python_version())'
# cv2 5.0 has no HOGDescriptor: the HOG rows take their oracle from cv2 4.8.1 in a second environment.
if [ ! -x .venv4/bin/python ]; then
  { uv venv --python 3.11 .venv4 && uv pip install --python .venv4/bin/python opencv-python-headless==4.8.1.78 "numpy<2"; } >/dev/null 2>&1 \
    || { echo "FAIL setup: cv2 4.8.1 environment (.venv4) for the HOG rows"; exit 1; }
fi
mkdir -p .work .bin
.venv/bin/mojo build -I src examples/quickstart.mojo -o .work/quickstart 2>.work/build.err \
  || { echo "FAIL build: examples/quickstart.mojo"; grep error .work/build.err | head -20; exit 1; }
.work/quickstart > /dev/null || { echo "FAIL example: examples/quickstart.mojo"; exit 1; }
tests/run.sh || { echo "FAIL 2d: tests/run.sh"; exit 1; }
.venv/bin/python -I tools/fastpath_check.py > .work/fastpath.log || { echo "FAIL fastpath: tools/fastpath_check.py"; tail -5 .work/fastpath.log; exit 1; }
tests/run3d.sh || { echo "FAIL 3d: tests/run3d.sh"; exit 1; }
echo "PASS ci"
