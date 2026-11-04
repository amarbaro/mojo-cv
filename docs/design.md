# Design

## Layout
- `src/mojocv/`: one package, one module per area. `mat.mojo` (the `Mat[dt]` container and the `.mcv` file format), `core.mojo` (Args, `par`,
  element-wise ops), `imgproc.mojo` (filters, colour, resize, warps, contours, HOG), `features.mojo`, `video.mojo`, `calib3d.mojo`
  (calibration, pose, stereo), and the 3D side (`desc3d`, `octree`, `mesh`, `nurbs`, `align`, `phasecorr`, `glb`, `step`).
- Two entry points sit outside the package: `src/main.mojo` (the `mojo-cv` tool) and `tools/cli3d.mojo` (`mojo-cv3d`). A package cannot hold a
  `main()` (`mojo precompile` refuses it on Mojo 1.2 and on the 1.3 nightly), so `src/mojocv/cli3d.mojo` exports `dispatch3d` and nothing else.
- `Mat[dt]` owns a `List[Scalar[dt]]` of `rows * cols * ch` values, row-major, channels interleaved. Ops return a new `Mat`.
  Images and arrays move between processes as `.mcv` files (16-byte header, raw little-endian data; `tools/mcv.py` reads and writes them).

## Why MAX is a dependency
`max.algorithm.parallelize` is the only thing taken from MAX (`core.mojo`, `desc3d.mojo`, `phasecorr.mojo`). `par()` wraps it with the worker
count measured on this CPU: 8 workers on the 8 physical cores (a dispatch costs about 8 us), 16 only for regions that run for milliseconds (with
16 every call waits about 215 us for a sleeping SMT sibling). The pin is `max==26.7.0` next to `mojo==1.2.0`; the library is CPU only, no GPU code.

## Memory and speed
- Hot loops work on raw pointers into the `List` storage with SIMD blocks, and the scalar tail handles what the block cannot. A block must
  stay inside its row: the cvtColor 32-byte loads and stores need 11 pixels (19 for the 16-pixel blocks) left, which `tools/fastpath_check.py`
  checks at every width 1..40.
- StereoSGBM keeps one uint16 cost volume and one int32 sum volume; each of the 8 aggregation paths holds two rolling rows. 1080p with 128
  disparities peaks at 1.5 GB (`tools/stereo_check.py` budgets 2 GB); the first version kept 11 int32 volumes (11.7 GB by arithmetic).
  Block sizes above 16 raise because the uint16 sum would overflow.
- Timings against cv2 are in [bench.md](bench.md) and [bench3d.md](bench3d.md); rerun them with `tests/run.sh --bench`.

## Errors
Every input a user can get wrong raises `mojocv: ...` and the tool exits 1: files shorter than their header, headers whose size does not match the
data, missing arguments, mismatched shapes, too few points for a RANSAC sample, glb accessors outside their buffer. `tools/malformed_check.py`
runs each of these against the built tools and fails on an abort, a hang or an exit 0.

## Checked against
cv2 5.0 for the 2D ops (cv2 4.8.1 for the HOG rows: 5.0 dropped `HOGDescriptor`), numpy, brute force or a closed form for the 3D ones.
Inputs are generated from fixed seeds in `tools/oracle_cv2.py` and `tools/oracle3d.py`. Python lives only in `tools/`, as subprocess oracles.

## Rejected
No Python binding (the tools talk through `.mcv` files) and no GPU path: the aim is exactness against the oracle, not speed records.
Differences from cv2 that remain are listed per op in [2d.md](2d.md).
