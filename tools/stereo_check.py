"""StereoSGBM regression: disparity bytes on a seeded 320x240 pair equal the committed golden (tests/golden/sgbm.sha256,
written by the pre-rolling-rows code), and 1080p / 128 disparities stay under a peak-RSS budget.
  `.venv/bin/python -I tools/stereo_check.py`            check
  `.venv/bin/python -I tools/stereo_check.py --gold`     rewrite the golden (only for an intended output change)
  run from the repo root after tests/run.sh built .bin/mojo-cv; MOJO_CV overrides the binary."""
import hashlib, os, resource, subprocess, sys
import numpy as np
sys.path.insert(0, "tools"); import mcv

BIN = os.environ.get("MOJO_CV", ".bin/mojo-cv")
GOLD = "tests/golden/sgbm.sha256"
RSS_MB = 2048
CASES = [  # name, args
    ("b5", ["minDisp=0", "numDisp=32", "block=5"]),
    ("b3_lr_uniq", ["minDisp=0", "numDisp=32", "block=3", "lrMax=1", "uniq=10"]),
    ("b1", ["minDisp=0", "numDisp=16", "block=1"]),
    ("b7_mind", ["minDisp=2", "numDisp=16", "block=7", "P1=40", "P2=300"]),
    ("neg_mind", ["minDisp=-8", "numDisp=24", "block=5", "lrMax=0"]),
    ("d48_b11", ["minDisp=0", "numDisp=48", "block=11", "uniq=5"]),
]


def pair(h, w, seed):
    rng = np.random.default_rng(seed)
    left = rng.integers(0, 256, (h, w), dtype=np.uint8)
    disp = np.zeros((h, w), int)
    disp[h // 4: h // 2, w // 4: w // 2] = 6
    disp[h // 2:, w // 2:] = 13
    right = left.copy()
    for y in range(h):
        for x in range(w):
            if x - disp[y, x] >= 0: right[y, x - disp[y, x]] = left[y, x]
    right = np.clip(right.astype(int) + rng.integers(-3, 4, (h, w)), 0, 255).astype(np.uint8)
    return left, right


def run(args, out):
    p = subprocess.run([BIN, "StereoSGBM", out, ".work/st_l.mcv", ".work/st_r.mcv", *args], capture_output=True, timeout=120)
    if p.returncode: sys.exit("FAIL stereo: %s rc %d %s" % (args, p.returncode, p.stderr[-200:]))
    return hashlib.sha256(open(out, "rb").read()).hexdigest()


os.makedirs(".work", exist_ok=True); os.makedirs("tests/golden", exist_ok=True)
l, r = pair(240, 320, 11); mcv.save(l, ".work/st_l.mcv"); mcv.save(r, ".work/st_r.mcv")
got = {n: run(a, ".work/st_o.mcv") for n, a in CASES}
if "--gold" in sys.argv:
    open(GOLD, "w").write("".join("%s %s\n" % (n, h) for n, h in got.items())); print("golden written"); sys.exit(0)
want = dict(line.split() for line in open(GOLD))
bad = [n for n in got if got[n] != want.get(n)]
for n in got: print("ok  " if n not in bad else "BAD ", n)
rng = np.random.default_rng(5)  # 1080p / 128 disparities: memory budget
mcv.save(rng.integers(0, 256, (1080, 1920), dtype=np.uint8), ".work/st_l.mcv"); mcv.save(rng.integers(0, 256, (1080, 1920), dtype=np.uint8), ".work/st_r.mcv")
p = subprocess.run([BIN, "StereoSGBM", ".work/st_o.mcv", ".work/st_l.mcv", ".work/st_r.mcv", "minDisp=0", "numDisp=128", "block=5"], capture_output=True, timeout=300)
rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024
print(("ok  " if p.returncode == 0 and rss < RSS_MB else "BAD ") + "1080p/128 rc %d, peak RSS %d MB (budget %d)" % (p.returncode, rss, RSS_MB))
sys.exit(1 if bad or p.returncode or rss >= RSS_MB else 0)
