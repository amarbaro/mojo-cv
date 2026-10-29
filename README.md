# mojo-cv

[![CI](https://github.com/amarbaro/mojo-cv/actions/workflows/ci.yml/badge.svg)](https://github.com/amarbaro/mojo-cv/actions/workflows/ci.yml)

OpenCV-style computer vision written from scratch in Mojo, on the CPU: filters, colour, resize and warps, contours,
ORB, optical flow, camera calibration and pose, stereo, plus 3D helpers (octree diffs, phase correlation, NURBS fits,
GLB and STEP export). Every op is run row by row against an oracle on the same input, cv2 5.0 for the 2D ops and numpy,
brute force or a closed form for the 3D ones: 258 and 161 rows, all passing in `scripts/ci.sh`. It is a library and a command-line
tool, not a cv2 replacement: there is no Python binding and no GPU code.

```mojo
from mojocv.mat import Mat
from mojocv.imgproc import gaussian_blur, otsu, threshold

def main() raises:
    var img = Mat[DType.uint8](64, 64, 1)
    for r in range(64):
        for c in range(64):
            img.set(r, c, 0, UInt8(40) if c < 32 else UInt8(200))
    var mask = threshold(img, Float64(otsu(img)), 255.0, 0)   # cv2.THRESH_OTSU, then THRESH_BINARY
    var soft = gaussian_blur(img, 5, 0.0)                      # cv2.GaussianBlur((5, 5), 0)
    print(mask.at(0, 0, 0), mask.at(0, 63, 0), soft.at(32, 31, 0))   # 0 255 90
```

## Install

Needs Mojo 1.2 and MAX 26.7: the code uses 1.2 syntax and `max.algorithm.parallelize`, and Mojo 1.1.0 does
not finish compiling it. `pyproject.toml` pins `mojo 1.2.0` and `max 26.7.0`.

```
uv sync                                                          # Mojo, MAX, numpy, opencv-python-headless 5.0
.venv/bin/mojo build -I src examples/quickstart.mojo -o quickstart && ./quickstart
.venv/bin/mojo build -O3 -I src src/main.mojo -o .bin/mojo-cv    # the command-line tool
```

In your own code: `mojo build -I path/to/mojo-cv/src app.mojo`, then `from mojocv.imgproc import ...`.

## What you get

| Area | Contents | Checked against |
|---|---|---|
| Core and filters | flip, rotate, absdiff, bitwise, convert, GaussianBlur, Sobel, pyrDown, threshold (Otsu, triangle), integral, erode and dilate, cvtColor (gray, HSV, YCrCb, Lab, Luv) | cv2 5.0 |
| Geometry of images | resize (nearest, linear, cubic, Lanczos, area), warpAffine, remap, blobFromImage | cv2 5.0 |
| Shapes and features | connected components, Canny, findContours (with hierarchy), goodFeaturesToTrack, line and rectangle, GrabCut, HOG people detector, ORB, descriptor matching | cv2 5.0 (HOG: cv2 4.8.1) |
| Motion | cornerSubPix, pyramidal Lucas-Kanade, Farneback | cv2 5.0 |
| Calibration and pose | Rodrigues, projectPoints, undistort maps (pinhole and fisheye), solvePnP (EPnP), RANSAC, essential matrix (5-point), recoverPose, stereoRectify, semi-global stereo matching, chessboard and ArUco 4x4_50 detection | cv2 5.0 |
| 3D and time | voxel cells, hashed octree diff, Umeyama alignment, 3D Haar, 3D descriptors, TAPVid-3D metrics, phase correlation, burst super-resolution | brute force, numpy, closed forms |
| CAD and CAE | NURBS fit and evaluation, watertight voxel meshes, GLB and STEP writers and readers, stress measures | independent parsers, closed forms |

Images and arrays travel as `.mcv` files (`tools/mcv.py` reads and writes them from numpy). The API is in
[docs/api.md](docs/api.md); how each 2D op differs from cv2 is in [docs/2d.md](docs/2d.md) and the 3D design decisions
in [docs/3d.md](docs/3d.md).

## Verified

`scripts/ci.sh` builds everything and runs: 258 2D rows (cv2 5.0, cv2 4.8.1 for HOG), 161 3D rows, the odd-shape and
4-channel fast-path check, and the quickstart. Inputs are generated from fixed seeds (`scene()` in `tools/oracle_cv2.py`);
nothing is downloaded except the Python packages. Most ops are bit exact or within one grey level of cv2; the tolerance
of every row is written in the row (`tests/cases/*.txt`, `tests/cases3d/*.txt`). Where an op is not cv2's algorithm
(StereoSGBM, GrabCut, findEssentialMat's root solve, chessboard corner order) [docs/2d.md](docs/2d.md) says how it
differs.

Not in CI, run by hand: the real-photo rows (`tests/cases/real.txt`: chessboard and ArUco photos from two Kaggle
datasets, one video frame; `tools/fetch_fixtures.sh` needs the kaggle CLI and `MCV_VIDEO`), and the Blender import check
of GLB output (`tools/blender_glb_check.py`).

## Performance

CPU only, one run on a Ryzen 7 7800X3D, ours against cv2 5.0 on 1920x1080 inputs, best of 5 per row
([docs/bench.md](docs/bench.md), regenerate with `tests/run.sh --bench-all --md docs/bench.md`):

- 36 of 51 rows are faster than cv2 (examples: 8-bit Otsu 0.05 ms vs 1.07, GaussianBlur 5x5 0.23 vs 0.59, pyrDown
  0.11 vs 0.36, HOG people detection 46.7 vs 71.1 ms, goodFeaturesToTrack 3.9 vs 27.4, ORB 12.4 vs 14.7).
- 4 more are within 1.5x and 11 are slower, listed in the table: Canny (2.0x), cvtColor to HSV (4.2x) and YCrCb (2.9x),
  flip (3.8x) and others. cv2 is hand-vectorised for those.

The ratios move with machine load, so a row near 1.0 can land on either side; the table says how to run the fair
old-build/new-build/cv2 comparison (`tools/abbench.py`). 3D ops against numpy or cv2 equivalents are in
[docs/bench3d.md](docs/bench3d.md) (for example Hamming matching of 10,000 x 10,000 descriptors: 12.7 ms vs 76.5 ms for
cv2's BFMatcher).

## Limits

- Needs Mojo 1.2 and MAX 26.7 (see Install). Linux x86-64 is what CI and the numbers cover.
- No GPU path and no Python binding. The input and output of the command-line tool is `.mcv`, not PNG or JPEG.
- Not every cv2 function exists; the ones that do are the lists above. Missing and different behaviour is in
  [docs/2d.md](docs/2d.md): cvtColor to YUV, anisotropic or even erode kernels, and so on.
- The speed table is one run on one machine; several 3D ops have no cv2 equivalent to compare with.
- The real-photo rows and the Blender check are not run by CI.
