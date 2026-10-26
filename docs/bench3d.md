# 3D speed rows

Measured 2026-10-26 on the 7800X3D (CPU only, 16 threads where an op is threaded; no GPU was used).
Ours: `.bin/mojo-cv3d OP ... reps=5`, best of 5, compute step only (files already loaded). Baseline: best of 5 Python
calls on in-memory arrays. Command: `python tools/bench3d.py`.

| op | size | ours (ms) | baseline | baseline (ms) | ours / baseline | note |
|---|---|---|---|---|---|---|
| phasecorr | 1920x1080 f64 | 41.34 | cv2.phaseCorrelate | 62.19 | 0.665 | CPU, 16 threads; cv2 returns (-21.0, 13.0) as well, ours is exact on fractional shifts too |
| burst (4 frames, splat only) | 4 x 1920x1080 f64 -> 3840x2160 | 125.5 | cv2.resize 2x INTER_LINEAR (1 frame) | 1.9 | 66 | no cv2 burst op; resize is the cost of the single-frame alternative |
| odiff, 1 changed cell | 64^3 scene, 3000 cells | 0.00306 | cv2.absdiff + countNonZero (dense, grids prebuilt) | 0.01338 | 0.229 | walk of two prebuilt trees, 35 of 262144 cells visited |
| octree build | 64^3 scene, 3000 cells | 0.2043 | numpy: dense grid fill | 0.02126 | 9.61 | the tree is built once per frame; the diff then costs the change |
| haar3c | 128^3 f64 | 70.22 | numpy oracle (python loops over levels) | 157 | 0.447 | no cv2 equivalent |
| describe | 3000 points, k=16 | 64.32 | - | - | - | O(N^2) kNN; no cv2 equivalent |
| match (Hamming, 64 bit) | 10000 x 10000 | 14.65 | cv2.BFMatcher(NORM_HAMMING).match | 82.16 | 0.178 |  |
| mesh | 17177 cells in 40^3 | 132.6 | - | - | - | no cv2 equivalent |
| nurbs_eval | 512x512 samples, 16x16 net | 111.8 | numpy oracle | 1684 | 0.0664 |  |
| nurbs_fit | 200x200 samples -> 16x16 net | 1.562 | numpy oracle (lstsq) | 2.669 | 0.585 |  |
| umeyama | 100000 points | 2.15 | numpy oracle (SVD) | 4.386 | 0.49 |  |
