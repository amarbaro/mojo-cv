"""Malformed input must raise a mojocv error (exit 1, message on stderr), never abort, crash or hang.
  run from the repo root after tests/run.sh and tests/run3d.sh built .bin/: `.venv/bin/python -I tools/malformed_check.py`"""
import struct, subprocess, sys

W = ".work/malformed"
bad = 0


def mcv(rows, cols, ch, code, data=b""):
    return b"MCV1" + struct.pack("<IIHH", rows, cols, ch, code) + data


def put(name, data):
    open(f"{W}/{name}", "wb").write(data)
    return f"{W}/{name}"


def expect_error(label, cmd, timeout=10):
    """cmd must exit 1 with 'mojocv' in its stderr: a raised Error."""
    global bad
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        ok, why = r.returncode == 1 and b"mojocv" in r.stderr, f"rc {r.returncode}: {r.stderr[-120:]!r}"
    except subprocess.TimeoutExpired:
        ok, why = False, f"no exit in {timeout} s"
    print(("ok   " if ok else "BAD  ") + label + ("" if ok else "  " + why))
    bad += not ok


subprocess.run(["mkdir", "-p", W], check=True)
# mat.load: a header whose rows*cols*ch wraps Int64 to 0 must not allocate or run
expect_error("load: 2^31 x 2^31 x 4 header", [".bin/mojo-cv", "flip", f"{W}/o.mcv", put("ovf.mcv", mcv(1 << 31, 1 << 31, 4, 0))])
# core.peek_code: a file shorter than its 16-byte header, and an op with no input at all
expect_error("peek_code: empty .mcv", [".bin/mojo-cv", "flip", f"{W}/o.mcv", put("empty.mcv", b"")])
expect_error("peek_code: 15-byte .mcv", [".bin/mojo-cv", "flip", f"{W}/o.mcv", put("short.mcv", mcv(1, 1, 1, 0)[:15])])
expect_error("core op without an input", [".bin/mojo-cv", "flip", f"{W}/o.mcv"])
print("BAD", bad)
sys.exit(1 if bad else 0)
