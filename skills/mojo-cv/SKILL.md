---
name: mojo-cv
description: "OpenCV-style image processing, features, calibration and 3D/CAD helpers in Mojo, checked against cv2 5.0. Use when Mojo code must blur, threshold, resize, warp, find contours or corners, match ORB features, track optical flow, solve PnP or write GLB/STEP, or when a Mojo op must agree with cv2 on the same input."
---

# mojo-cv

CPU computer vision for Mojo, with cv2 5.0 as the oracle. 258 2D rows and 161 3D rows run in `scripts/ci.sh`.

## Install

Needs Mojo 1.2 and MAX 26.7 from `pyproject.toml` (`uv sync`). Mojo 1.1.0 does not compile it.
`mojo build -I <repo>/src app.mojo`, then `from mojocv.imgproc import ...`.

## Use

```mojo
from mojocv.mat import Mat
from mojocv.imgproc import gaussian_blur, otsu, threshold

def main() raises:
    var img = Mat[DType.uint8](64, 64, 1)
    for r in range(64):
        for c in range(64):
            img.set(r, c, 0, UInt8(40) if c < 32 else UInt8(200))
    var mask = threshold(img, Float64(otsu(img)), 255.0, 0)   # cv2.THRESH_OTSU then THRESH_BINARY
    var soft = gaussian_blur(img, 5, 0.0)                      # cv2.GaussianBlur((5, 5), 0)
    print(mask.at(0, 0, 0), mask.at(0, 63, 0), soft.at(32, 31, 0))
```

The command-line tool runs one op on `.mcv` files: `mojo-cv OP OUT IN... [k=v]`
(`mojo build -O3 -I src src/main.mojo -o .bin/mojo-cv`). Op names and options: `docs/2d.md`, `docs/3d.md`.

## Rules

- `Mat[dt]` is `rows x cols x ch`, row-major, interleaved, one dtype fixed at compile time (`uint8`, `int16`,
  `int32`, `float32`, `float64`). Ops are generic over `dt`; a few take only `uint8`.
- Function arguments follow cv2's meaning, not its spelling: `flip(a, code)`, `threshold(src, thresh, maxval, kind)` with
  kind 0 binary, 1 inverse, 2 trunc, 3 tozero, 4 tozero inverse. Read the docstring before assuming a cv2 default.
- Most 2D ops are bit exact or within one grey level of cv2. StereoSGBM, GrabCut, findEssentialMat's root solve and
  chessboard corner order are not cv2's algorithm; `docs/2d.md` lists each difference with its measured agreement.
- `Mat.at` is bounds-checked and slow; hot loops take `m.ptr()`. A `Mat` that only a raw pointer outlives is freed at its
  last named use: keep it with `_ = m.rows`.
- Parallel regions use 8 workers (`core.par`); 16 workers cost about 215 us per region, 8 cost about 8 us.
- HOG exists only as the cv2 4.x people detector (`hog_detect`); its oracle runs under cv2 4.8.1 (`.venv4`).
- A new op needs case rows in `tests/cases/*.txt` (2D, against cv2) or `tests/cases3d/*.txt` (3D, against an oracle in
  `tools/oracle3d.py`); a row is never loosened to make it pass.
