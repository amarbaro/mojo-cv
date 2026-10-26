# Geometry and sampler notes

The 3D oracle now creates missing derived mesh inputs before GLB and STEP rows,
including STEP reader rows run without mesh cases. Previously a fresh worktree
failed in `check_glb` before reaching the mesh cases because alphabetical case
order put GLB first. `tests/run3d.sh` on a fresh `.work/run3d` passes.

EPnP replaces both pose initialization paths with PCA control points, normalized
image equations, linearized inter-control distances and Gauss-Newton beta
refinement. Planar inputs use three control points. Pixel reprojection still uses
the existing damped Gauss-Newton refinement (LM). RANSAC now samples four points.
Two new cv2 rows cover four non-coplanar points and noisy full-set refinement.
Primary derivation: https://vincentlepetit.github.io/files/papers/comp_lepetit_ijcv08.pdf

Five-point RANSAC solves the nine trace cubics and determinant constraint in a
four-dimensional epipolar null space. Elimination gives multiplication by z on a
ten-dimensional quotient basis. Real eigenvalues come from Hessenberg shifted QR;
reconstructed matrices must satisfy the cubic constraints before scoring. This is
an action-matrix implementation of the five-point equations, not Nister's exact
univariate-polynomial elimination. Source: Nister 2004, DOI 10.1109/TPAMI.2004.17.
New six-correspondence and clean-cloud rows pass against cv2 at 1e-5 and 1e-6;
existing noisy tolerances are unchanged. RANSAC best time on ess_ransac increased
from 2.285 ms to 12.577 ms (20 repeats); this correctness item misses cv2 speed.
The accepted variant is Stewenius, Engels & Nister (2006), cited in `docs/sources.md`.
`ess_five_roots` returns all six real matrices from an exactly five-point problem,
matching cv2 within 6.2e-12; its remaining four algebraic roots are complex and
excluded. Five-point outputs are canonicalized by Frobenius norm, largest-entry
sign, then their first entry for deterministic set comparison.

Rectification supports alpha -1 and [0, 1] with border sampling and interpolated
inner/outer rectangle scales. Vertical baselines use the y principal-point
difference in Q. Four new cv2 rows cover alpha 0/1 and vertical -1/0.5; max P/Q
error is 3.2e-5. Default rectification (1000 repeats): 0.02 s / 13736 KB before,
0.02 s / 13796 KB after; best compute 0.00777 -> 0.00822 ms.

INTER_AREA (resize interp=4) uses exact overlap weights: each destination pixel averages the source
pixels by the area they cover, in both directions. It matches cv2 within 1 for every shrink in both
axes, for pure x or pure y changes, for exact 2x enlargement (bit exact) and for the 1.9 x / 1.5 x
enlargement row. cv2 itself departs from exact area weights when the output is enlarged in y: on the
270-row fixture, h = 300 (scale 0.9) differs from the exact weights by 216-233 grey levels for every
width, h = 200 or 400 differ by 52-57 for some widths. Those sizes are not rows; the cv2 behaviour
there is not an average over the covered area, so matching it would mean copying its quirk.

Thick lines (line thick=N, rectangle thick=N) follow cv2's ThickLine: a filled rectangle around the
segment (16.16 fixed-point vertices, outline by Line2, scanline fill between the two active edges)
plus a filled Bresenham circle at each end; a rectangle is a closed polyline with a cap at the end of
each side. Exact against cv2 on 300 random lines inside the image (thickness 2-12, every slope) and
on the rows. When a polygon edge leaves the image, a few boundary pixels in the first rows next to the
clip differ by one (99.9% of values equal on the clipped row); two guesses at what cv2 does there
(no clip, and an alternative start phase) made it worse, so it is left as measured.
