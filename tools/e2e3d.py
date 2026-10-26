"""User-level check on a real recording: python tools/e2e3d.py VIDEO > .work/e2e3d.txt
A  burst super-resolution of a real 1080p frame from four aliased 960x540 phases (known and estimated shifts) against bilinear/bicubic
B  phase correlation of a real frame against a Fourier-shifted copy, ours vs cv2
C  a voxelized 64x36x24 clip -> static/moving octree split -> watertight mesh -> GLB (colored by time) and STEP -> Blender import
Everything goes through the .bin/mojo-cv3d CLI."""
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import mcv
import oracle3d as o

ROOT = Path(__file__).resolve().parent.parent
BIN = str(ROOT / ".bin" / "mojo-cv3d")
W = ROOT / ".work" / "e2e3d"
W.mkdir(parents=True, exist_ok=True)
video = sys.argv[1]


def run(op, out, ins, *args):
    for i, a in enumerate(ins):
        if not isinstance(a, Path):
            mcv.save(a, W / f"{op}{i}.mcv")
    paths = [a if isinstance(a, Path) else W / f"{op}{i}.mcv" for i, a in enumerate(ins)]
    r = subprocess.run([BIN, op, str(out), *map(str, paths), *args], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"FAIL {op}: {(r.stdout + r.stderr)[-300:]}")
    return r.stdout


def ffmpeg_gray(ss, w, h, n=1, fps=None):
    vf = f"scale={w}:{h}" + (f",fps={fps}" if fps else "")
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", str(ss), "-i", video, "-frames:v", str(n), "-vf", vf, "-pix_fmt", "gray",
                          "-f", "rawvideo", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float64)


def psnr(a, b, crop=4):
    d = (a[crop:-crop, crop:-crop] - b[crop:-crop, crop:-crop]) ** 2
    return 10 * np.log10(255.0 ** 2 / max(d.mean(), 1e-12))


print("# L3D user-level run\n")
# ---- A: burst
hr = ffmpeg_gray(30, 1920, 1080)[0]
lr = [hr[sy::2, sx::2].copy() for sy, sx in ((0, 0), (0, 1), (1, 0), (1, 1))]  # four decimation phases = half-pixel shifts
shifts = np.array([[0, 0], [-0.5, 0], [0, -0.5], [-0.5, -0.5]])
run("burst", W / "sr_known.mcv", lr + [shifts], "shifts=1", "scale=2")
out_k = mcv.load(W / "sr_known.mcv")
txt = run("burst", W / "sr_est.mcv", lr, "scale=2")
out_e = mcv.load(W / "sr_est.mcv")
bil = cv2.resize(lr[0], None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
cub = cv2.resize(lr[0], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
print("## A. Burst super-resolution, real 1920x1080 frame, 4 decimated 960x540 phases, scale 2\n")
print("| method | PSNR vs the original frame (dB, 4 px border cropped) |\n|---|---|")
print(f"| bilinear upsample of one frame | {psnr(bil, hr):.2f} |")
print(f"| bicubic upsample of one frame | {psnr(cub, hr):.2f} |")
print(f"| burst, shifts given | {psnr(out_k, hr):.2f} |")
print(f"| burst, shifts from phase correlation | {psnr(out_e, hr):.2f} |")
print("\nestimated shifts (dx, dy; true: (-0.5,0) (0,-0.5) (-0.5,-0.5)):")
print("\n".join("    " + l for l in txt.strip().splitlines()))
print()
# ---- B: phase correlation
fr = ffmpeg_gray(40, 1920, 1080)[0]
print("## B. Phase correlation, real frame against a Fourier-shifted copy (truth in the first column)\n")
print("| true (dx, dy) | ours | cv2.phaseCorrelate | ours with Hann | cv2 with Hann |\n|---|---|---|---|---|")
win = cv2.createHanningWindow((1920, 1080), cv2.CV_64F)
for dx, dy in ((7.0, -3.0), (2.5, 1.25), (-0.4, 0.3)):
    sh = o.fshift(fr, dy, dx)
    r0 = mcv.load(W / "pc.mcv") if False else None
    run("phasecorr", W / "pc.mcv", [fr, sh])
    p0 = mcv.load(W / "pc.mcv")[0]
    run("phasecorr", W / "pcw.mcv", [fr, sh], "window=1")
    p1 = mcv.load(W / "pcw.mcv")[0]
    c0 = cv2.phaseCorrelate(fr, sh)[0]
    c1 = cv2.phaseCorrelate(fr, sh, win)[0]
    print(f"| ({dx}, {dy}) | ({p0[0]:.3f}, {p0[1]:.3f}) | ({c0[0]:.3f}, {c0[1]:.3f}) | ({p1[0]:.3f}, {p1[1]:.3f}) | ({c1[0]:.3f}, {c1[1]:.3f}) |")
print()
# ---- C: clip -> voxels -> octree -> mesh -> GLB/STEP -> Blender
clip = ffmpeg_gray(20, 64, 36, n=24, fps=2)
print(f"## C. Voxelized clip: {len(clip)} frames of 64x36, cell (x, y, t) occupied where the pixel is brighter than the clip mean\n")
occ = clip > clip.mean()
cells = np.argwhere(occ.transpose(2, 1, 0)).astype(np.int32)  # (x, y, t)
cells = cells[np.lexsort((cells[:, 2], cells[:, 1], cells[:, 0]))]
print(f"- occupied cells: {len(cells)} of {occ.size}")
slabs = [cells[(cells[:, 2] >= 8 * k) & (cells[:, 2] < 8 * k + 8)] * [1, 1, 1] - [0, 0, 8 * k] for k in range(3)]
txt = run("static", W / "st.mcv", [s.astype(np.int32) for s in slabs], "depth=6")
st = mcv.load(W / "st.mcv")
print(f"- static/moving across three 8-frame slabs: {int(st[:, 3].sum())} static of {len(st)} cells; walk: {txt.strip()}")
a = (slabs[0] * 0 + slabs[0]).astype(np.int32)
txt = run("odiff", W / "od.mcv", [slabs[0].astype(np.int32), slabs[1].astype(np.int32)], "depth=6")
print(f"- slab 0 vs slab 1 diff: {txt.strip()}")
txt = run("mesh", W / "clip.v.mcv", [cells], f"faces={W / 'clip.f.mcv'}")
print(f"- mesh: {txt.strip()}")
v, f = mcv.load(W / "clip.v.mcv"), mcv.load(W / "clip.f.mcv")
field = v[:, 2:3].copy()  # time coordinate of every vertex
run("colormap", W / "col.mcv", [field])
run("glb", W / "clip.glb", [W / "clip.v.mcv", W / "clip.f.mcv", W / "col.mcv"])
run("step", W / "clip.stp", [W / "clip.v.mcv", W / "clip.f.mcv"])
txt = run("step_read", W / "back.v.mcv", [W / "clip.stp"], f"faces={W / 'back.f.mcv'}")
print(f"- STEP written ({(W / 'clip.stp').stat().st_size} bytes) and read back by our reader: {txt.strip()}")
bl = subprocess.run(["blender", "-b", "-P", str(ROOT / "tools" / "blender_glb_check.py"), "--", str(W / "clip.glb")], capture_output=True, text=True,
                    timeout=300).stdout
m = re.search(r"BLENDER_MESH \S+ verts=(\d+) faces=(\d+) color_attrs=(\d+)", bl)
print(f"- GLB written ({(W / 'clip.glb').stat().st_size} bytes), imported by Blender 5.2: {m[0] if m else 'NO MESH'}")
ok = m and int(m[1]) == len(v) and int(m[2]) == len(f) and int(m[3]) >= 1
print(f"- Blender vertex/face counts equal ours ({len(v)} / {len(f)}) and the color attribute is present: {'yes' if ok else 'NO'}")
sys.exit(0 if ok else 1)
