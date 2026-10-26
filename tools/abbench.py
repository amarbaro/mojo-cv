"""A/B timing of 1080p rows: OLD_BIN vs .bin/mojo-cv vs cv2, interleaved, min of N runs per side.
  abbench.py OLD_BIN [N] [ROWID...]   (run under `.venv/bin/python -I`)
Interleaving makes the ratios fair on a loaded box; absolute numbers still move with the load."""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import oracle_cv2 as oc

old = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 3
only = set(sys.argv[3:])
new = os.path.join(oc.ROOT, ".bin", "mojo-cv")
cases = sorted(os.path.join(oc.ROOT, "tests", "cases", f) for f in os.listdir(os.path.join(oc.ROOT, "tests", "cases")) if f.endswith(".txt"))
tmp = tempfile.mkdtemp(dir=os.path.join(oc.ROOT, ".work"))


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    return oc.time_of(r.stdout) if r.returncode == 0 else None


print("| op | row | old ms | new ms | cv2 ms | new/cv2 |\n|---|---|---:|---:|---:|---:|")
for _f, (cid, opn, ins, kv, _c) in oc.rows(cases):
    if only and cid not in only or not only and not any(i.startswith("hd") for i in ins):
        continue
    paths = [oc.fpath(i) for i in ins]
    kvs = [f"{k}={v}" for k, v in kv.items()] + ["reps=5"]
    py = oc.PY4 if opn in oc.HOG_OPS else sys.executable
    best = [None, None, None]
    for _ in range(n):
        for k, cmd in enumerate(([old, opn, os.path.join(tmp, "o"), *paths, *kvs], [new, opn, os.path.join(tmp, "n"), *paths, *kvs], [py, "-I", oc.__file__, opn, os.path.join(tmp, "c"), *paths, *kvs])):
            t = run(cmd)
            if t is not None and (best[k] is None or t < best[k]):
                best[k] = t
    f = lambda v: f"{v:.2f}" if v is not None else "n/a"
    print(f"| {opn} | {cid} | {f(best[0])} | {f(best[1])} | {f(best[2])} | {best[1] / best[2]:.2f} |" if best[1] and best[2] else f"| {opn} | {cid} | {f(best[0])} | {f(best[1])} | {f(best[2])} | |", flush=True)
