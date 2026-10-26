# 2D speed rows: mojo-cv vs cv2 5.0

Method: `tests/run.sh --bench-all --md docs/bench.md`. Each row runs the same op in one resident process with `reps=5`
and keeps the best wall time of the compute only (file load and save excluded), ours via `.bin/mojo-cv`, cv2 via
`tools/oracle_cv2.py` (numpy in, cv2 call, best of 5). Both sides use all threads of the 7800X3D; CPU only, no GPU.
Inputs are the same `.mcv` fixtures (`hd*` = 1920x1080 synthetic scenes from `scene()` in `tools/oracle_cv2.py`, the rest
smaller). Target: ours <= cv2. **Misses are listed as measured; nothing is hidden.** The HOG rows time cv2 4.8.1
(cv2 5.0 has no HOG).

1080p rows: 36 of 51 faster than cv2, 4 more within 1.5x, the rest slower (ratio column).

Slower than cv2 in this run: canny_hdg, cvt_hsv_hd, cvt_lab_hd, cvt_swap_hd, cvt_ycrcb_hd, erode_hd5, flip_x_hd, remap_lin_hdg, resize_cub_hdg, resize_nn_hd, transpose_hd.

- **Parallel dispatch**: `max.algorithm.parallelize` with the default 16 workers costs about 215 us per region (best of
  50 trivial regions); with 8 workers (the physical cores) it costs about 8 us, so `core.par` uses 8 and takes 16 only for
  regions that run for milliseconds. Allocating and zero-filling a 2 MB `Mat` is 18 us.
- **Read the ratios with care.** One run of this table is one sample on a shared desktop: a row near 1.0 can land on either
  side, and a loaded run of the same binary moves a row by up to a factor of two. `tools/abbench.py OLD_BIN` runs an old
  build, the current build and cv2 back to back per row, which is the fair comparison when the machine is not quiet.
- Point-set ops (projectPoints, estimateAffine2D, stereoRectify) take microseconds on both sides; 0.00 ms entries are
  below the timer's resolution, not a result.

## 1920x1080 rows

| op | row | ours ms | cv2 ms | ours/cv2 |
|---|---|---:|---:|---:|
| flip | flip_x_hd | 0.66 | 0.17 | 3.79 |
| rotate | rotate_cw_hd | 1.00 | 1.20 | 0.83 **faster** |
| transpose | transpose_hd | 0.23 | 0.15 | 1.50 |
| absdiff | absdiff_hd | 0.11 | 0.28 | 0.39 **faster** |
| bitwise_and | bitwise_and_hd | 0.12 | 0.20 | 0.62 **faster** |
| bitwise_not | bitwise_not_hd | 0.11 | 0.17 | 0.66 **faster** |
| convert | convert_u8_f32_hd | 0.44 | 13.79 | 0.03 **faster** |
| bitwise_or | bitwise_or_hd | 0.10 | 0.21 | 0.47 **faster** |
| bitwise_xor | bitwise_xor_hd | 0.12 | 0.22 | 0.57 **faster** |
| ORB | orb_hd | 12.43 | 14.74 | 0.84 **faster** |
| GaussianBlur | gauss5_hd | 0.23 | 0.59 | 0.40 **faster** |
| GaussianBlur | gauss_s2_hdg | 0.73 | 0.84 | 0.86 **faster** |
| pyrDown | pyrdown_hd | 0.11 | 0.36 | 0.31 **faster** |
| pyrDown | pyrdown_hdg | 0.04 | 0.40 | 0.11 **faster** |
| Sobel | sobel_dx_hdg | 0.22 | 0.22 | 1.04 |
| Sobel | sobel_dy5_hd | 0.88 | 1.93 | 0.46 **faster** |
| threshold | thresh_bin_hdg | 0.05 | 0.10 | 0.50 **faster** |
| threshold | thresh_tozinv_hd | 0.11 | 0.17 | 0.62 **faster** |
| threshold | thresh_otsu_hdg | 0.05 | 1.07 | 0.04 **faster** |
| threshold | thresh_tri_hdg | 0.05 | 1.05 | 0.05 **faster** |
| integral | integral_hdg | 0.40 | 0.47 | 0.85 **faster** |
| integral | integral_hd | 1.34 | 1.53 | 0.88 **faster** |
| erode | erode_hdg | 0.11 | 0.11 | 0.96 **faster** |
| erode | erode_hd5 | 0.39 | 0.22 | 1.78 |
| cvtColor | cvt_gray_hd | 0.11 | 0.09 | 1.29 |
| cvtColor | cvt_swap_hd | 0.11 | 0.07 | 1.63 |
| cvtColor | cvt_hsv_hd | 1.73 | 0.41 | 4.21 |
| cvtColor | cvt_ycrcb_hd | 0.41 | 0.14 | 2.85 |
| cvtColor | cvt_lab_hd | 1.54 | 0.79 | 1.94 |
| cvtColor | cvt_luv_hd | 1.62 | 1.36 | 1.19 |
| resize | resize_down_lin_hd | 0.04 | 0.13 | 0.30 **faster** |
| resize | resize_nn_hd | 0.14 | 0.09 | 1.52 |
| resize | resize_cub_hdg | 0.32 | 0.12 | 2.68 |
| resize | resize_lan_hdg | 2.31 | 4.96 | 0.47 **faster** |
| resize | resize_area_hd | 0.56 | 0.94 | 0.60 **faster** |
| resize | resize_area_frac | 0.71 | 3.75 | 0.19 **faster** |
| warpAffine | warp_rot_hd | 0.67 | 1.83 | 0.37 **faster** |
| warpAffine | warp_rot_hdg | 0.55 | 0.58 | 0.94 **faster** |
| remap | remap_lin_hd | 0.74 | 1.62 | 0.46 **faster** |
| remap | remap_lin_hdg | 1.11 | 0.70 | 1.58 |
| blobFromImage | blob_hd | 0.08 | 0.28 | 0.30 **faster** |
| Canny | canny_hdg | 5.26 | 2.67 | 1.97 |
| goodFeaturesToTrack | gftt_hdg | 3.86 | 27.40 | 0.14 **faster** |
| goodFeaturesToTrack | gftt_harris_hdg | 2.22 | 24.71 | 0.09 **faster** |
| line | line_diag_hd | 0.11 | 0.13 | 0.86 **faster** |
| line | line_thick_hd | 0.19 | 0.15 | 1.30 |
| HOGScores | hog_scores_hdg | 2.74 | 307.51 | 0.01 **faster** |
| detectMultiScale | hog_raw_hdg | 46.65 | 71.05 | 0.66 **faster** |
| detectMultiScale | hog_group_hdg | 32.40 | 72.26 | 0.45 **faster** |
| resize | resize_13_lin_hd | 0.33 | 0.37 | 0.90 **faster** |
| resize | resize_07_lin_hd | 0.34 | 0.52 | 0.65 **faster** |

## Other sizes (270x480 and smaller, point sets, stereo pairs)

| op | row | ours ms | cv2 ms | ours/cv2 |
|---|---|---:|---:|---:|
| Rodrigues | rod_vec | 0.00 | 0.00 | n/a |
| Rodrigues | rod_zero | 0.00 | 0.00 | n/a |
| Rodrigues | rod_mat | 0.01 | 0.00 | n/a |
| Rodrigues | rod_mat_pi | 0.00 | 0.00 | n/a |
| Rodrigues | rod_mat_id | 0.00 | 0.00 | n/a |
| projectPoints | proj_d4 | 0.00 | 0.01 | 0.11 **faster** |
| projectPoints | proj_d5 | 0.00 | 0.01 | 0.13 **faster** |
| projectPoints | proj_d8 | 0.00 | 0.01 | 0.11 **faster** |
| projectPoints | proj_d12 | 0.00 | 0.01 | 0.13 **faster** |
| projectPoints | proj_nodist | 0.00 | 0.01 | 0.12 **faster** |
| fisheye_projectPoints | fproj | 0.00 | 0.01 | 0.37 **faster** |
| undistortPoints | undist_d4 | 0.00 | 0.00 | n/a |
| undistortPoints | undist_d8 | 0.00 | 0.00 | n/a |
| undistortPoints | undist_rp | 0.00 | 0.01 | 0.56 **faster** |
| undistortPoints | undist_zero | 0.00 | 0.00 | n/a |
| fisheye_undistortPoints | fundist | 0.01 | 0.00 | n/a |
| fisheye_undistortPoints | fundist_rp | 0.00 | 0.00 | n/a |
| initUndistortRectifyMap | rectmap_d5 | 2.05 | 0.28 | 7.34 |
| initUndistortRectifyMap | rectmap_id | 0.03 | 0.01 | 4.49 |
| fisheye_initUndistortRectifyMap | frectmap | 0.57 | 1.25 | 0.46 **faster** |
| estimateAffine2D | aff_ransac | 0.03 | 0.02 | 1.13 |
| estimateAffine2D | aff_tight | 0.02 | 0.03 | 0.57 **faster** |
| solvePnP | pnp_d4 | 0.25 | 0.06 | 4.00 |
| solvePnP | pnp_planar | 0.09 | 0.07 | 1.25 |
| solvePnP | pnp_nodist | 0.35 | 0.08 | 4.28 |
| solvePnPRansac | pnp_ransac | 3.87 | 0.96 | 4.03 |
| findEssentialMat | ess_ransac | 14.79 | 2.27 | 6.51 |
| stereoRectify | rect_zero | 0.01 | 0.02 | 0.35 **faster** |
| stereoRectify | rect_dist | 0.01 | 0.02 | 0.38 **faster** |
| stereoRectify | rect_nozd | 0.01 | 0.02 | 0.34 **faster** |
| findEssentialMat | ess_noisy | 14.37 | 2.25 | 6.39 |
| recoverPose | pose_masked | 0.77 | 0.18 | 4.26 |
| recoverPose | pose_all | 0.99 | 0.20 | 5.03 |
| StereoSGBM | sgbm_dots | 8.58 | 1.99 | 4.30 |
| StereoSGBM | sgbm_block3 | 7.89 | 2.58 | 3.06 |
| StereoSGBM | sgbm_mind | 5.05 | 1.66 | 3.04 |
| findChessboardCorners | chess_a | 0.74 | 0.78 | 0.95 **faster** |
| findChessboardCorners | chess_b_rot | 0.63 | 0.97 | 0.65 **faster** |
| findChessboardCorners | chess_c | 0.52 | 3.84 | 0.14 **faster** |
| findChessboardCorners | chess_none | 6.19 | 9.81 | 0.63 **faster** |
| aruco_detect | aruco_four | 6.79 | 1.02 | 6.64 |
| aruco_detect | aruco_persp | 1.94 | 0.39 | 4.93 |
| aruco_detect | aruco_none | 1.51 | 0.67 | 2.26 |
| fisheye_projectPoints | fproj_zero_rot | 0.00 | 0.00 | n/a |
| fisheye_initUndistortRectifyMap | frectmap_small | 0.02 | 0.01 | 1.34 |
| solvePnPRansac | pnp_ransac_pl | 0.47 | 0.32 | 1.47 |
| solvePnP | pnp_epnp_four | 0.29 | 0.06 | 4.75 |
| solvePnP | pnp_epnp_noise | 0.30 | 0.08 | 3.67 |
| findEssentialMat | ess_five_minimal | 0.62 | 0.12 | 5.08 |
| findEssentialMat | ess_five_clean | 0.56 | 0.17 | 3.26 |
| findEssentialMat | ess_five_roots | 0.44 | 0.19 | 2.39 |
| stereoRectify | rect_alpha0 | 0.02 | 0.02 | 0.86 **faster** |
| stereoRectify | rect_alpha1 | 0.02 | 0.02 | 0.81 **faster** |
| stereoRectify | rect_vertical | 0.01 | 0.02 | 0.34 **faster** |
| stereoRectify | rect_vertical_a | 0.02 | 0.02 | 0.82 **faster** |
| flip | flip_y_odd | 0.01 | 0.00 | n/a |
| flip | flip_xy_one | 0.01 | 0.00 | n/a |
| flip | flip_y_f32 | 0.01 | 0.00 | n/a |
| rotate | rotate_180_odd | 0.01 | 0.00 | n/a |
| rotate | rotate_ccw_i16 | 0.01 | 0.00 | n/a |
| transpose | transpose_i32 | 0.02 | 0.00 | n/a |
| absdiff | absdiff_g1 | 0.01 | 0.00 | n/a |
| absdiff | absdiff_i16 | 0.01 | 0.00 | n/a |
| absdiff | absdiff_f32 | 0.01 | 0.00 | n/a |
| bitwise_or | bitwise_or_odd | 0.01 | 0.00 | n/a |
| bitwise_xor | bitwise_xor_f32 | 0.01 | 0.00 | n/a |
| bitwise_not | bitwise_not_one | 0.01 | 0.00 | n/a |
| convert | convert_f32_u8_half | 0.01 | 0.01 | 1.43 |
| convert | convert_f32_i16_scale | 0.01 | 0.01 | 1.12 |
| convert | convert_f64_i32 | 0.01 | 0.01 | 1.90 |
| convert | convert_u8_u8_one | 0.01 | 0.01 | 1.57 |
| bitwise_and | bitwise_and_odd | 0.01 | 0.00 | n/a |
| bitwise_xor | bitwise_xor_i16 | 0.01 | 0.00 | n/a |
| bfmatch | bf_ab | 0.82 | 0.15 | 5.36 |
| bfmatch | bf_one | 0.03 | 0.01 | 4.66 |
| bfmatch | bf_self | 0.54 | 0.14 | 3.89 |
| knnmatch | knn_ab | 0.77 | 0.27 | 2.86 |
| knnmatch | knn_ratio_tight | 0.83 | 0.31 | 2.66 |
| ORB | orb_sm | 2.63 | 2.08 | 1.26 |
| ORB | orb_blank | 0.26 | 0.03 | 9.63 |
| GaussianBlur | gauss5_odd | 0.01 | 0.00 | n/a |
| GaussianBlur | gauss5_smg | 0.01 | 0.02 | 0.55 **faster** |
| GaussianBlur | gauss5_sm | 0.02 | 0.04 | 0.52 **faster** |
| GaussianBlur | gauss7_odd | 0.01 | 0.01 | 1.43 |
| GaussianBlur | gauss_f32 | 0.01 | 0.00 | n/a |
| GaussianBlur | gauss3_one | 0.01 | 0.00 | n/a |
| pyrDown | pyrdown_smg | 0.01 | 0.02 | 0.75 **faster** |
| pyrDown | pyrdown_sm | 0.24 | 0.13 | 1.86 |
| pyrDown | pyrdown_odd | 0.01 | 0.00 | n/a |
| pyrDown | pyrdown_one | 0.01 | 0.00 | n/a |
| pyrDown | pyrdown_f32 | 0.01 | 0.00 | n/a |
| Sobel | sobel_scharr_g1 | 0.01 | 0.00 | n/a |
| Sobel | sobel_f32_dxx | 0.01 | 0.00 | n/a |
| threshold | thresh_inv_sm | 0.23 | 0.01 | 28.40 |
| threshold | thresh_trunc_g1 | 0.01 | 0.00 | n/a |
| threshold | thresh_tozero_g1 | 0.01 | 0.00 | n/a |
| threshold | thresh_otsu_one | 0.01 | 0.00 | n/a |
| threshold | thresh_tri_bin | 0.01 | 0.09 | 0.15 **faster** |
| threshold | thresh_tri_blank | 0.01 | 0.00 | n/a |
| threshold | thresh_tri_g2 | 0.01 | 0.00 | n/a |
| threshold | thresh_f32 | 0.01 | 0.00 | n/a |
| integral | integral_odd | 0.01 | 0.00 | n/a |
| integral | integral_f32 | 0.23 | 0.00 | n/a |
| integral | integral_one | 0.01 | 0.00 | n/a |
| dilate | dilate_cross_sm | 0.11 | 0.06 | 1.80 |
| dilate | dilate_ell_smg | 0.11 | 0.08 | 1.34 |
| erode | erode_ell_odd | 0.02 | 0.00 | n/a |
| dilate | dilate_one | 0.02 | 0.00 | n/a |
| erode | erode_f32 | 0.02 | 0.00 | n/a |
| cvtColor | cvt_rgb2gray_odd | 0.01 | 0.00 | n/a |
| cvtColor | cvt_gray2bgr_g1 | 0.22 | 0.00 | n/a |
| cvtColor | cvt_hsv_odd | 0.01 | 0.00 | n/a |
| cvtColor | cvt_hsv_one | 0.01 | 0.00 | n/a |
| cvtColor | cvt_hsv2bgr_odd | 0.01 | 0.00 | n/a |
| cvtColor | cvt_ycrcb2bgr_sm | 0.12 | 0.08 | 1.50 |
| cvtColor | cvt_gray_f32 | 0.01 | 0.00 | n/a |
| cvtColor | cvt_lab_one | 0.23 | 0.00 | n/a |
| cvtColor | cvt_lab_f32 | 0.18 | 0.00 | n/a |
| cvtColor | cvt_lab2bgr_sm | 0.32 | 0.25 | 1.27 |
| cvtColor | cvt_lab2bgr_one | 0.18 | 0.00 | n/a |
| cvtColor | cvt_luv_odd | 0.23 | 0.00 | n/a |
| cvtColor | cvt_luv_f32 | 0.20 | 0.00 | n/a |
| cvtColor | cvt_luv2bgr_sm | 0.35 | 0.31 | 1.13 |
| cvtColor | cvt_luv2bgr_f32 | 0.31 | 0.23 | 1.36 |
| cvtColor | cvt_lab2bgr_f32 | 0.34 | 0.24 | 1.40 |
| resize | resize_up_lin_sm | 0.07 | 0.22 | 0.30 **faster** |
| resize | resize_nn_odd | 0.01 | 0.00 | n/a |
| resize | resize_cub_sm | 0.54 | 0.07 | 7.19 |
| resize | resize_lan_sm | 0.22 | 1.53 | 0.14 **faster** |
| resize | resize_lin_one | 0.01 | 0.00 | n/a |
| resize | resize_lin_f32 | 0.02 | 0.00 | n/a |
| resize | resize_half_g | 0.01 | 0.01 | 1.20 |
| resize | resize_half_c | 0.01 | 0.03 | 0.38 **faster** |
| resize | resize_area_up2 | 0.54 | 0.17 | 3.19 |
| resize | resize_area_shrinkx | 0.10 | 0.45 | 0.23 **faster** |
| resize | resize_area_up | 0.37 | 0.24 | 1.59 |
| resize | resize_area_odd | 0.01 | 0.00 | n/a |
| resize | resize_area_one | 0.01 | 0.00 | n/a |
| resize | resize_area_f32 | 0.01 | 0.00 | n/a |
| warpAffine | warp_rot_sm | 0.04 | 0.11 | 0.36 **faster** |
| warpAffine | warp_nn_sm | 0.35 | 0.15 | 2.33 |
| warpAffine | warp_cub_smg | 0.59 | 0.10 | 6.10 |
| warpAffine | warp_rep_odd | 0.01 | 0.01 | 1.60 |
| warpAffine | warp_inv_g1 | 0.01 | 0.01 | 2.32 |
| warpAffine | warp_f32 | 0.01 | 0.01 | 2.03 |
| remap | remap_lin_sm | 0.51 | 0.40 | 1.27 |
| remap | remap_nn_sm | 0.27 | 0.09 | 2.93 |
| remap | remap_cub_smg | 0.47 | 0.16 | 2.87 |
| remap | remap_odd_const | 0.01 | 0.01 | 2.24 |
| remap | remap_odd_refl | 0.01 | 0.00 | n/a |
| blobFromImage | blob_sm | 0.08 | 0.24 | 0.34 **faster** |
| connectedComponentsWithStats | ccl8_bin | 1.43 | 0.31 | 4.64 |
| connectedComponentsWithStats | ccl4_binhd | 17.49 | 3.62 | 4.84 |
| connectedComponentsWithStats | ccl8_shapes | 0.04 | 0.04 | 1.11 |
| connectedComponentsWithStats | ccl_blank | 0.00 | 0.00 | n/a |
| Canny | canny_smg_l2 | 0.39 | 0.20 | 1.94 |
| Canny | canny_sm5 | 0.50 | 0.26 | 1.91 |
| Canny | canny_blank | 0.04 | 0.01 | 3.05 |
| Canny | canny_g1 | 0.04 | 0.01 | 4.53 |
| findContours | contours_ext_shapes | 0.03 | 0.02 | 2.03 |
| findContours | contours_list_shapes | 0.02 | 0.02 | 0.85 **faster** |
| findContours | contours_simple_shapes | 0.02 | 0.02 | 0.84 **faster** |
| findContours | contours_ext_bin | 0.53 | 0.22 | 2.38 |
| findContours | contours_list_bin | 0.59 | 0.58 | 1.02 |
| findContours | contours_blank | 0.00 | 0.00 | n/a |
| findContours | contours_full | 0.00 | 0.01 | 0.18 **faster** |
| findContours | contours_ccomp_shapes | 0.02 | 0.02 | 0.81 **faster** |
| findContours | contours_tree_shapes | 0.02 | 0.02 | 1.11 |
| findContours | contours_tree_simple | 0.02 | 0.02 | 0.86 **faster** |
| findContours | contours_ccomp_bin | 0.55 | 0.64 | 0.86 **faster** |
| findContours | contours_tree_bin | 0.65 | 0.81 | 0.80 **faster** |
| findContours | contours_tree_binhd | 16.01 | 14.08 | 1.14 |
| findContours | contours_tree_blank | 0.00 | 0.00 | n/a |
| findContours | contours_tree_full | 0.00 | 0.01 | 0.21 **faster** |
| goodFeaturesToTrack | gftt_smg | 0.27 | 1.95 | 0.14 **faster** |
| goodFeaturesToTrack | gftt_blank | 0.07 | 0.01 | 7.27 |
| goodFeaturesToTrack | gftt_harris_smg | 0.29 | 1.83 | 0.16 **faster** |
| goodFeaturesToTrack | gftt_harris_blank | 0.08 | 0.01 | 9.44 |
| line | line_steep_g1 | 0.00 | 0.00 | n/a |
| line | line_clip_sm | 0.01 | 0.01 | 0.76 **faster** |
| line | line_out_sm | 0.01 | 0.01 | 0.65 **faster** |
| line | line_thick_g1 | 0.00 | 0.00 | n/a |
| line | line_thick_flat | 0.02 | 0.01 | 1.17 |
| line | line_thick_dot | 0.01 | 0.01 | 0.52 **faster** |
| line | line_thick_clip | 0.02 | 0.02 | 1.48 |
| rectangle | rect_thick_sm | 0.04 | 0.02 | 1.84 |
| line | line_hor_odd | 0.00 | 0.00 | n/a |
| rectangle | rect_edge_sm | 0.01 | 0.01 | 0.93 **faster** |
| rectangle | rect_fill_odd | 0.00 | 0.00 | n/a |
| HOGScores | hog_scores_smg | 0.98 | 17.79 | 0.05 **faster** |
| detectMultiScale | hog_raw_smg | 2.49 | 3.27 | 0.76 **faster** |
| detectMultiScale | hog_group_smg | 2.52 | 3.59 | 0.70 **faster** |
| detectMultiScale | hog_group_smg2 | 1.91 | 3.65 | 0.52 **faster** |
| detectMultiScale | hog_none | 2.46 | 3.09 | 0.80 **faster** |
| groupRectangles | group_rects_hdg | 0.76 | 1.11 | 0.68 **faster** |
| grabCut | grabcut_scene | 35.89 | 11.16 | 3.22 |
| grabCut | grabcut_shot | 53.31 | 8.76 | 6.08 |
| grabCut | grabcut_tiny | 0.10 | 0.07 | 1.51 |
| cornerSubPix | subpix_sm | 1.14 | 0.50 | 2.27 |
| cornerSubPix | subpix_win3 | 0.55 | 0.30 | 1.82 |
| cornerSubPix | subpix_edge | 3.21 | 1.15 | 2.80 |
| calcOpticalFlowPyrLK | lk_sm | 11.01 | 0.37 | 30.05 |
| calcOpticalFlowPyrLK | lk_sm_win9 | 5.26 | 0.15 | 35.84 |
| calcOpticalFlowFarneback | fb_shift | 21.44 | 16.98 | 1.26 |
| calcOpticalFlowFarneback | fb_frames | 20.98 | 17.88 | 1.17 |
| calcOpticalFlowFarneback | fb_poly7 | 19.69 | 17.25 | 1.14 |
| calcOpticalFlowFarneback | fb_tiny | 2.22 | 0.01 | 181.96 |
