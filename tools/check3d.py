"""Differential driver: tests/cases3d/*.txt rows run through the Mojo CLI and the numpy oracle.
Row: NAME OP TOL INPUTS(comma list of fixture names) [k=v ...] [expect:KEY=VALUE ...]
TOL is a number (max abs diff) or `exact`. expect:visited_frac_lt=X checks the printed `visited=`/`dense=`.
Prints `PASS name` or `FAIL name: reason`; exit 1 on any failure."""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mcv
import oracle3d

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / ".bin" / "mojo-cv3d"
WORK = ROOT / ".work" / "run3d"


def ours(op, out, ins, args):
    r = subprocess.run([str(BIN), op, str(out), *map(str, ins), *args], capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def check(row, fx):
    name, op, tol, ins_s, *rest = row.split()
    if name in fx:
        return False, "row name collides with a fixture of the same name"
    args = [t for t in rest if not t.startswith("expect:")]
    expect = dict(t[7:].split("=") for t in rest if t.startswith("expect:"))
    kv = dict(a.split("=") for a in args)
    ins = [WORK / f"{n}.mcv" for n in ins_s.split(",") if n != "-"]
    for path in ins:
        if not path.exists() and path.name.startswith("mesh_") and path.name.endswith((".v.mcv", ".f.mcv")):
            stem = path.name.rsplit(".", 2)[0]
            code, stdout, stderr = ours("mesh", WORK / f"{stem}.v.mcv", [WORK / f"ms_{stem[5:]}.mcv"],
                                       [f"faces={WORK / (stem + '.f.mcv')}"])
            if code != 0:
                return False, f"mesh fixture failed: {(stdout + stderr).strip()[:200]}"
    out = WORK / f"{name}.ours.mcv"
    if op in oracle3d.CUSTOM:
        return oracle3d.CUSTOM[op](dict(name=name, ins=ins, args=args, expect=expect, work=WORK, mcv=mcv,
                                        run=lambda o, i, a: ours(op, o, i, a),
                                        run3=lambda p, o, i, a: ours(p, o, i, a)))
    code, stdout, stderr = ours(op, out, ins, args)
    if "raises" in expect:
        return (code != 0), ("expected a failure, got exit 0" if code == 0 else "")
    if code != 0:
        return False, f"exit {code}: {(stdout + stderr).strip()[:200]}"
    got = mcv.load(out)
    want = oracle3d.OPS[op]([mcv.load(p) for p in ins], kv)
    mcv.save(want, WORK / f"{name}.mcv")
    if "equals" in expect:
        want = mcv.load(WORK / f"{expect['equals']}.mcv")
    if "cyl" in expect:  # exact rational quarter cylinder: radius 1, normal = outward radial
        xy, nz = got[..., :2], got[..., 3:5]
        d = max(float(np.abs(np.linalg.norm(xy, axis=-1) - 1).max()), float(np.abs(xy - nz).max()), float(np.abs(got[..., 5]).max()))
        if d > 1e-12:
            return False, f"cylinder deviates by {d:g}"
        print(f"     {name}: radius and radial normal exact to {d:.1e}")
    if "crop" in expect:
        c = int(expect["crop"])
        got, want = got[c:-c, c:-c], want[c:-c, c:-c]
    if "cols" in expect:
        got, want = got[:, : int(expect["cols"])], want[:, : int(expect["cols"])]
    if got.shape != want.shape or got.dtype != want.dtype:
        return False, f"shape/dtype {got.shape} {got.dtype} vs {want.shape} {want.dtype}"
    diff = float(np.max(np.abs(got.astype(np.float64) - want.astype(np.float64)))) if got.size else 0.0
    if diff > (0.0 if tol == "exact" else float(tol)):
        return False, f"maxdiff {diff:g} > {tol}"
    if "visited_frac_lt" in expect:
        m = re.search(r"visited=\s*(\d+)\s+dense=\s*(\d+)", stdout)
        v, d = int(m[1]), int(m[2])
        if v >= float(expect["visited_frac_lt"]) * d:
            return False, f"visited {v} of {d} dense"
        print(f"     {name}: visited {v} of {d} dense cells = {100.0 * v / d:.4f}%")
    if "kept_frac_lt" in expect:
        m = re.search(r"kept=\s*(\d+)\s+of=\s*(\d+)", stdout)
        k, n = int(m[1]), int(m[2])
        if k >= float(expect["kept_frac_lt"]) * n:
            return False, f"kept {k} of {n}"
        x = mcv.load(ins[0])
        c = oracle3d.OPS["haar3f"]([x], kv)
        dropped = float((c[np.abs(c) < float(kv["thr"])] ** 2).sum())
        err = float(((x - got) ** 2).sum())
        if abs(err - dropped) > 1e-9 * max(1.0, dropped):  # Parseval: squared error = dropped energy
            return False, f"error energy {err:g} != dropped energy {dropped:g}"
        print(f"     {name}: kept {k} of {n} coefficients = {100.0 * k / n:.2f}%, squared error {err:.4g} = dropped energy")
    return True, f"maxdiff {diff:g}"


def main(files):
    WORK.mkdir(parents=True, exist_ok=True)
    fx = oracle3d.fixtures()
    for n, a in fx.items():
        mcv.save(a, WORK / f"{n}.mcv")
    bad = n_rows = 0
    for fn in files:
        for line in Path(fn).read_text().splitlines():
            line = line.split("#")[0].strip()
            if not line:
                continue
            n_rows += 1
            ok, msg = check(line, fx)
            print(("PASS " if ok else "FAIL ") + line.split()[0] + (": " + msg if msg else ""))
            bad += not ok
    print(f"{n_rows - bad}/{n_rows} cases passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
