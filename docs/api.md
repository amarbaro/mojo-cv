# API

Two ways in. Both are checked by the same case rows (`tests/cases/*.txt`, `tests/cases3d/*.txt`), which run the CLI
against cv2 5.0, a numpy oracle, or an exact closed form.

## Library

Build with `mojo build -I path/to/mojo-cv/src app.mojo`, then `from mojocv.<module> import ...`. A runnable example is
`examples/quickstart.mojo`, built and run by `scripts/ci.sh`.

### Mat

`mojocv.mat.Mat[dt]` owns a `rows x cols x ch` array, row-major and interleaved (cv2's layout), `dt` fixed at compile
time (`uint8`, `int16`, `int32`, `float32`, `float64`).

| Call | Behaviour |
|---|---|
| `Mat[dt](rows, cols, ch=1)` | zero-filled array |
| `m.at(r, c, k=0)`, `m.set(r, c, k, v)` | bounds-checked element access |
| `m.ptr()`, `m.size()` | raw pointer to the data, element count |
| `load[dt](path)`, `save(m, path)` | the `.mcv` file format: `"MCV1"`, u32 rows, u32 cols, u16 ch, u16 dtype code, raw little-endian data. `tools/mcv.py` reads and writes the same files from numpy |

### 2D ops

Same meaning as the cv2 function of the same name; the exactness of each is in `docs/2d.md`.

| Module | Functions |
|---|---|
| `core` | `flip(a, code)`, `rotate(a, code)`, `absdiff(a, b)`, `bitwise[op](a, b)`, `convert[s, d](a, alpha, beta)` |
| `imgproc` | `gaussian_blur(src, ksize, sigma)`, `sobel`, `pyr_down`, `threshold(src, thresh, maxval, kind)`, `otsu`, `triangle`, `integral`, `morph[is_min](src, shape, k, iters)`, `cvt_color(src, code)`, `resize(src, w, h, interp)`, `warp_affine`, `remap`, `blob`, `components`, `canny`, `find_contours`, `good_features`, `draw_line`, `draw_rect`, `hog_detect`, `group_rects`, `grab_cut` |
| `features` | `orb_detect(img, nfeatures, fast_t)`, `knn2(q, t)`, `ratio_test(k, ratio)` |
| `video` | `corner_sub_pix`, `lk_flow` (pyramidal Lucas-Kanade), `farneback` |
| `calib3d` | `rodrigues_vec`, `project_points`, `undistort_points`, `init_undistort_rectify_map`, `estimate_affine`, `solve_pnp`, `solve_pnp_ransac`, `find_essential`, `recover_pose`, `stereo_rectify`, `stereo_sgm`, `find_chessboard`, `aruco_detect` (4x4_50) |

### 3D, time and CAD

Checked against a brute-force Mojo oracle, numpy, or a closed form; details in `docs/3d.md`.

| Module | Functions |
|---|---|
| `voxel`, `octree` | `voxelize`, `unvoxelize`, `occupied`; Zobrist-hashed octree with `walk` (diff, static/moving split) |
| `align` | `umeyama(src, dst, with_scale)` |
| `wavelet` | `haar3[inverse](v)`, `compress(v, thr)` |
| `desc3d` | `describe(p, k)`, `match_desc(a, b)` |
| `tapvid` | `tapvid(...)`: APD, OA, AJ |
| `phasecorr`, `burst` | `phase_corr(a, b, window)`, `register`, `burst` |
| `nurbs`, `mesh` | `fit_surface`, `eval_surface`; `voxel_mesh(cells)` |
| `glb`, `step` | `write_glb`, `read_glb`; `write_step_mesh`, `write_step_surface`, `read_step` |
| `cae` | `stress_measures`, `colormap` |

## Command line

`mojo-cv OP OUT IN... [k=v]` runs one op on `.mcv` files (build: `mojo build -O3 -I src src/main.mojo -o .bin/mojo-cv`).
Several outputs go to `OUT`, `OUT.1`, `OUT.2`, ... `reps=N` repeats the compute and prints `TIME best_ms`. The op names
and options are the tables in `docs/2d.md` and `docs/3d.md`; an unknown op or option exits non-zero with
`FAIL main: unknown op NAME`.

## Case rows

`id op IN... [k=v...] => cmp[, cmp...]`, one comparison per output file: `exact`, `max:T`, `mean:T`, `frac:T:F`, `iou:F`,
`pts:T:F`, `cnt:R`, `orb:T:F:H`. The tolerance is in the row and a row is never weakened to pass. The 3D rows are
documented in `tools/check3d.py`.
