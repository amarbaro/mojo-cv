"""Odd shapes and 1/3/4 channels through the uint8 fast paths (resize 2x, GaussianBlur 5, pyrDown) against cv2, bit exact.
  run from the repo root: `.venv/bin/python -I tools/fastpath_check.py` (after tests/run.sh has built .bin/mojo-cv)"""
import sys, numpy as np, cv2, subprocess
sys.path.insert(0, "tools"); import mcv
rng = np.random.default_rng(3)
bad = 0
for (h, w, ch) in [(40, 200, 4), (41, 203, 4), (30, 64, 4), (16, 9, 4), (50, 301, 1), (50, 301, 3)]:
    a = rng.integers(0, 256, (h, w, ch), dtype=np.uint8)
    if ch == 1: a = a[..., 0]
    mcv.save(a, ".work/c4.mcv")
    for name, args, ref in [("pyrDown", [], cv2.pyrDown(a)), ("GaussianBlur", ["ksize=5", "sigma=0"], cv2.GaussianBlur(a, (5, 5), 0))] + ([("resize", ["w=%d" % (w // 2), "h=%d" % (h // 2), "interp=1"], cv2.resize(a, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR))] if w % 2 == 0 and h % 2 == 0 else []):
        subprocess.run([".bin/mojo-cv", name, ".work/c4o.mcv", ".work/c4.mcv", *args], check=True, capture_output=True)
        o = mcv.load(".work/c4o.mcv")
        d = np.abs(o.astype(int) - ref.reshape(o.shape).astype(int)).max()
        print(name, h, w, ch, "maxdiff", d); bad += d > 0
for (h, w, ch, dw, dh) in [(70, 101, 3, 67, 41), (70, 101, 3, 150, 90), (33, 47, 1, 29, 17), (50, 301, 1, 400, 61), (40, 200, 3, 17, 3), (21, 33, 3, 33, 21), (40, 200, 4, 130, 70)]:
    a = rng.integers(0, 256, (h, w, ch), dtype=np.uint8)
    if ch == 1: a = a[..., 0]
    mcv.save(a, ".work/c4.mcv")
    subprocess.run([".bin/mojo-cv", "resize", ".work/c4o.mcv", ".work/c4.mcv", "w=%d" % dw, "h=%d" % dh, "interp=1"], check=True, capture_output=True)
    o = mcv.load(".work/c4o.mcv")
    d = np.abs(o.astype(int) - cv2.resize(a, (dw, dh), interpolation=cv2.INTER_LINEAR).reshape(o.shape).astype(int)).max()
    print("resize general", h, w, ch, "->", dh, dw, "maxdiff", d); bad += d > 1
print("BAD", bad)
sys.exit(1 if bad else 0)
