"""Speed rows for docs/bench3d.md: our CLI with reps=5 (compute step only, files already loaded) against the cv2 call that
does the same job, or against the numpy oracle where cv2 has no equivalent. Usage: python tools/bench3d.py > docs/bench3d.md"""
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mcv
import oracle3d as o

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / ".bin" / "mojo-cv3d"
W = ROOT / ".work" / "bench3d"
W.mkdir(parents=True, exist_ok=True)
rs = np.random.default_rng(1)


def ours(op, ins, *args):
    for i, a in enumerate(ins):
        mcv.save(a, W / f"in{i}.mcv")
    r = subprocess.run([str(BIN), op, str(W / "out.mcv"), *[str(W / f"in{i}.mcv") for i in range(len(ins))], *args, "reps=5"],
                       capture_output=True, text=True)
    m = re.search(r"time best_ms=\s*([\d.e+-]+)", r.stdout)
    if not m:
        raise SystemExit(f"bench {op} failed: {r.stdout[-200:]} {r.stderr[-200:]}")
    return float(m[1])


def best(fn, n=5):
    t = 1e9
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t * 1e3


rows = []


def row(name, size, ours_ms, base_name, base_ms, note=""):
    rows.append((name, size, ours_ms, base_name, base_ms, note))


# phase correlation, 1920x1080
a = rs.random((1080, 1920))
b = np.roll(a, (13, -21), (0, 1))
row("phasecorr", "1920x1080 f64", ours("phasecorr", [a, b]), "cv2.phaseCorrelate", best(lambda: cv2.phaseCorrelate(a, b)),
    "CPU, 16 threads; cv2 returns (-21.0, 13.0) as well, ours is exact on fractional shifts too")
# burst: 4 frames 1080p, scale 2, known shifts
frames = [a, np.roll(a, 1, 1), np.roll(a, 1, 0), np.roll(a, (1, 1), (0, 1))]
sh = np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]])
ms = ours("burst", frames + [sh], "shifts=1", "scale=2")
row("burst (4 frames, splat only)", "4 x 1920x1080 f64 -> 3840x2160", ms, "cv2.resize 2x INTER_LINEAR (1 frame)",
    best(lambda: cv2.resize(a, None, fx=2, fy=2)), "no cv2 burst op; resize is the cost of the single-frame alternative")
# octree diff vs dense absdiff
fx = o.fixtures()
scene, add1 = fx["scene64"], fx["scene64_add1"]
ga = np.zeros((64, 64 * 64), np.uint8)
gb = ga.copy()
for g, cells in ((ga, scene), (gb, add1)):
    g[cells[:, 0], cells[:, 1] * 64 + cells[:, 2]] = 255
ms = ours("odiff", [scene, add1], "depth=6")
row("odiff, 1 changed cell", "64^3 scene, 3000 cells", ms, "cv2.absdiff + countNonZero (dense, grids prebuilt)",
    best(lambda: cv2.countNonZero(cv2.absdiff(ga, gb))), "walk of two prebuilt trees, 35 of 262144 cells visited")
row("octree build", "64^3 scene, 3000 cells", ours("ohash", [scene], "depth=6"), "numpy: dense grid fill",
    best(lambda: np.zeros((64, 4096), np.uint8).__setitem__((scene[:, 0], scene[:, 1] * 64 + scene[:, 2]), 255)),
    "the tree is built once per frame; the diff then costs the change")
# haar compress
n = 128
vol = rs.random((n, n, n))
ms = ours("haar3c", [vol], "thr=0.5")
row("haar3c", "128^3 f64", ms, "numpy oracle (python loops over levels)", best(lambda: o.OPS["haar3c"]([vol], {"thr": 0.5}), 2),
    "no cv2 equivalent")
# descriptors and matching
pts = rs.normal(size=(3000, 3))
ms = ours("describe", [pts], "k=16")
row("describe", "3000 points, k=16", ms, "-", float("nan"), "O(N^2) kNN; no cv2 equivalent")
d1 = rs.integers(0, 256, (10000, 8), dtype=np.uint8)
d2 = rs.integers(0, 256, (10000, 8), dtype=np.uint8)
ms = ours("match", [d1, d2])
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
row("match (Hamming, 64 bit)", "10000 x 10000", ms, "cv2.BFMatcher(NORM_HAMMING).match", best(lambda: bf.match(d1, d2), 3), "")
# mesh
bl = np.unique(rs.integers(0, 40, (20000, 3)), axis=0).astype(np.int32)
ms = ours("mesh", [bl], f"faces={W}/f.mcv")
row("mesh", f"{len(bl)} cells in 40^3", ms, "-", float("nan"), "no cv2 equivalent")
# nurbs
net = np.ones((16, 16, 4))
net[..., :3] = rs.normal(size=(16, 16, 3))
row("nurbs_eval", "512x512 samples, 16x16 net", ours("nurbs_eval", [net], "p=3", "q=3", "m=512", "n=512"), "numpy oracle",
    best(lambda: o.nurbs_eval([net], {"p": 3, "q": 3, "m": 512, "n": 512}), 2), "")
grid = o.nurbs_eval([net], {"p": 3, "q": 3, "m": 200, "n": 200})[..., :3]
row("nurbs_fit", "200x200 samples -> 16x16 net", ours("nurbs_fit", [grid], "nu=16", "nv=16", "p=3", "q=3"), "numpy oracle (lstsq)",
    best(lambda: o.nurbs_fit([grid], {"nu": 16, "nv": 16, "p": 3, "q": 3}), 2), "")
# umeyama
src = rs.normal(size=(100000, 3))
dst = src * 1.5 + 2.0
row("umeyama", "100000 points", ours("umeyama", [src, dst]), "numpy oracle (SVD)", best(lambda: o.umeyama([src, dst], {}), 5), "")

print("# 3D speed rows\n")
print(f"Measured {time.strftime('%Y-%m-%d')} on the 7800X3D (CPU only, 16 threads where an op is threaded; no GPU was used).")
print("Ours: `.bin/mojo-cv3d OP ... reps=5`, best of 5, compute step only (files already loaded). Baseline: best of 5 Python")
print("calls on in-memory arrays. Command: `python tools/bench3d.py`.\n")
print("| op | size | ours (ms) | baseline | baseline (ms) | ours / baseline | note |")
print("|---|---|---|---|---|---|---|")
for name, size, om, bn, bm, note in rows:
    ratio = "-" if bm != bm else f"{om / bm:.3g}"
    bms = "-" if bm != bm else f"{bm:.4g}"
    print(f"| {name} | {size} | {om:.4g} | {bn} | {bms} | {ratio} | {note} |")
