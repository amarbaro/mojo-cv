# ransac_iters at a vanishing inlier share: 1 - w^m rounds to 1.0 below 1.1e-16, so ln(1 - w^m) must come from log1p.
from mojocv.calib3d import ransac_iters


def main() raises:
    var got = ransac_iters(1, 1_000_000, 3, 0.99, 5000)  # w = 1e-6, w^3 = 1e-18
    if got != 5000:
        raise Error("FAIL ransac_iters: 1 inlier in 1e6 points gave " + String(got) + ", want the cap 5000")
    if ransac_iters(50, 100, 3, 0.99, 5000) != 35:  # (ln 0.01) / ln(1 - 0.125) = 34.5
        raise Error("FAIL ransac_iters: 50 of 100 gave " + String(ransac_iters(50, 100, 3, 0.99, 5000)) + ", want 35")
    print("ransac_iters: ok")
