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


def expect_error(label, cmd, timeout=10, msg=b"mojocv"):
    """cmd must exit 1 with msg (default 'mojocv') in its stderr: a raised Error."""
    global bad
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        ok, why = r.returncode == 1 and msg in r.stderr, f"rc {r.returncode}: {r.stderr[-120:]!r}"
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
# Rng.sample(n, k) with n < k looped forever; callers need at least 3 (affine), 4 (PnP), 5 (essential) points
pts = lambda n: mcv(n, 2, 1, 3, struct.pack("<%dd" % (2 * n), *[float(i % 7) for i in range(2 * n)]))
expect_error("estimateAffine2D with 2 points", [".bin/mojo-cv", "estimateAffine2D", f"{W}/o.mcv", put("p2a.mcv", pts(2)), put("p2b.mcv", pts(2))], timeout=5)
expect_error("estimateAffine2D with 0 points", [".bin/mojo-cv", "estimateAffine2D", f"{W}/o.mcv", put("p0a.mcv", pts(0)), put("p0b.mcv", pts(0))], timeout=5)
f64 = lambda r, c, *v: mcv(r, c, 1, 3, struct.pack("<%dd" % (r * c), *(list(v) + [1.0] * (r * c - len(v)))))
K = put("k.mcv", f64(3, 3, 500, 0, 320, 0, 500, 240, 0, 0, 1))
expect_error("solvePnPRansac with 3 points", [".bin/mojo-cv", "solvePnPRansac", f"{W}/o.mcv", put("o3.mcv", f64(3, 3)), put("i3.mcv", f64(3, 2)), K, put("d0.mcv", f64(0, 1))], timeout=5, msg=b"need at least 4 points")
expect_error("findEssentialMat with 4 points", [".bin/mojo-cv", "findEssentialMat", f"{W}/o.mcv", put("e1.mcv", f64(4, 2)), put("e2.mcv", f64(4, 2)), K], timeout=5, msg=b"need at least 5 points")
# glb accessor bounds: (count - 1) * stride must not wrap to a small number
import json
def glb(acc):
    doc = {"asset": {"version": "2.0"}, "meshes": [{"primitives": [{"attributes": {"POSITION": 0}}]}],
           "bufferViews": [{"buffer": 0, "byteLength": 16}], "accessors": [acc]}
    js = json.dumps(doc).encode(); js += b" " * (-len(js) % 4)
    return (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + 16) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", 16, 0x004E4942) + bytes(16))
for count in (1 << 62, 3 << 61, 1 << 40):  # the first two wrap (count - 1) * 12; the last is a plain too-big control
    acc = {"bufferView": 0, "componentType": 5126, "count": count, "type": "VEC3"}
    expect_error(f"glb_read accessor count {count}", [".bin/mojo-cv3d", "glb_read", f"{W}/g.mcv", put("wrap.glb", glb(acc)), f"faces={W}/gf.mcv"], timeout=10, msg=b"reads outside")
# shapes and lengths the ops index without checking (each must raise a mojocv error, not read out of range)
u8 = lambda r, c, ch=1: mcv(r, c, ch, 0, bytes((i * 7) % 256 for i in range(r * c * ch)))
f32 = lambda r, c, ch=1: mcv(r, c, ch, 1, struct.pack("<%df" % (r * c * ch), *[float(i % 5) for i in range(r * c * ch)]))
O = f"{W}/o.mcv"
expect_error("remap: mapy shape differs from mapx", [".bin/mojo-cv", "remap", O, put("rm_i.mcv", u8(4, 4)), put("rm_x.mcv", f32(4, 4)), put("rm_y.mcv", f32(2, 2))], msg=b"mojocv: remap")
expect_error("blobFromImage: 1 mean value for 3 channels", [".bin/mojo-cv", "blobFromImage", O, put("bl_i.mcv", u8(4, 4, 3)), "mean=1.0"], msg=b"mojocv: blobFromImage")
expect_error("HOGScores: svm with 10 floats", [".bin/mojo-cv", "HOGScores", O, put("hg_i.mcv", u8(128, 64)), put("hg_s.mcv", f32(1, 10))], msg=b"mojocv: HOGScores")
print("BAD", bad)
sys.exit(1 if bad else 0)
