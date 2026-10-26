# TAPVid-3D metrics (Koppula et al. 2024), the 3D extension of the TAP-Vid metrics. Inputs are M = T * N rows
# (any order, the metrics pool every frame and track): ground-truth and predicted camera-space positions
# (M x 3 float64, z = depth) and visibility flags (M x 1 uint8, nonzero = visible).
#   threshold:  delta px projected into 3D at the ground-truth depth, delta_3D = delta * z / f for
#               delta in {1, 2, 4, 8, 16} px (mode px), or fixed {1, 4, 16, 64, 256} cm (mode metric)
#   APD:  share of ground-truth-visible points predicted within delta_3D, mean over the five thresholds
#   OA:   share of all points whose predicted visibility equals the ground truth
#   AJ:   TP / (TP + FP + FN), mean over thresholds; TP = visible in both and within delta_3D,
#         FP = predicted visible and (ground truth hidden or beyond delta_3D),
#         FN = ground truth visible and (predicted hidden or beyond delta_3D)
# Output 1 x 3: [APD, OA, AJ]. The paper's median depth rescaling is a pre-step on the predictions, not applied here.
from std.math import sqrt
from mojocv.mat import Mat


def tapvid(
    gt: Mat[DType.float64],
    gv: Mat[DType.uint8],
    pr: Mat[DType.float64],
    pv: Mat[DType.uint8],
    focal: Float64,
    metric: Bool,
) raises -> Mat[DType.float64]:
    var m = gt.rows
    if gt.cols != 3 or pr.cols != 3 or pr.rows != m or gv.rows != m or pv.rows != m or gv.cols != 1 or pv.cols != 1:
        raise Error("mojocv: tapvid wants gt/pred as M x 3 float64 and both visibilities as M x 1 uint8")
    if focal <= 0.0:
        raise Error("mojocv: tapvid needs focal > 0")
    var nvis = 0
    var agree = 0
    for i in range(m):
        nvis += Int(gv.data[i] != 0)
        agree += Int((gv.data[i] != 0) == (pv.data[i] != 0))
    if nvis == 0:
        raise Error("mojocv: tapvid: no visible ground-truth point, the metrics are undefined")
    var apd = 0.0
    var aj = 0.0
    for k in range(5):
        var tp = 0
        var fp = 0
        var miss = 0
        var hit = 0
        for i in range(m):
            var d2 = 0.0
            for a in range(3):
                var e = pr.data[i * 3 + a] - gt.data[i * 3 + a]
                d2 += e * e
            var delta = (0.01 * Float64(4**k)) if metric else (Float64(1 << k) * gt.data[i * 3 + 2] / focal)
            var close = sqrt(d2) < delta
            var g = gv.data[i] != 0
            var p = pv.data[i] != 0
            hit += Int(g and close)
            tp += Int(g and p and close)
            fp += Int(p and not (g and close))
            miss += Int(g and not (p and close))
        apd += Float64(hit) / Float64(nvis) / 5.0
        aj += Float64(tp) / Float64(tp + fp + miss) / 5.0
    var out = Mat[DType.float64](1, 3)
    out.data[0] = apd
    out.data[1] = Float64(agree) / Float64(m)
    out.data[2] = aj
    return out^
