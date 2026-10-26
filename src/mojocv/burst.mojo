# Burst super-resolution (after Wronski et al. 2019, reduced to translation): register every frame to the first one
# with phase correlation, splat each pixel onto a `scale` times finer grid with bilinear weights, divide by the
# accumulated weight. Frame t is the reference shifted by s_t (frame(x) = ref(x - s_t)), so its pixel (x, y) lands at
# ((x - s_x) * scale, (y - s_y) * scale) on the fine grid, whose pixel (0, 0) coincides with the reference's. Fine
# pixels no frame reaches take the reference's bilinear upsample. Out-of-grid splats are dropped.
from std.math import floor
from mojocv.mat import Mat
from mojocv.phasecorr import phase_corr


def register(frames: List[Mat[DType.float64]], window: Bool) raises -> Mat[DType.float64]:
    """T x 2 [dx, dy] of every frame against frames[0]."""
    var s = Mat[DType.float64](len(frames), 2)
    for t in range(1, len(frames)):
        var r = phase_corr(frames[0], frames[t], window)
        s.data[t * 2] = r.data[0]
        s.data[t * 2 + 1] = r.data[1]
    return s^


def burst(frames: List[Mat[DType.float64]], shifts: Mat[DType.float64], scale: Int) raises -> Mat[DType.float64]:
    if len(frames) == 0 or scale < 1 or shifts.rows != len(frames) or shifts.cols != 2:
        raise Error("mojocv: burst wants >= 1 frames, scale >= 1 and one [dx, dy] shift per frame")
    var rows = frames[0].rows
    var cols = frames[0].cols
    for i in range(len(frames)):
        if not frames[i].same_shape(frames[0]) or frames[i].ch != 1:
            raise Error("mojocv: burst wants single-channel frames of one size")
    for k in range(len(frames) * 2):
        if not (abs(shifts.data[k]) < 1.0e6):  # also rejects NaN, which would make Int(floor(.)) undefined
            raise Error("mojocv: burst shifts must be finite and below 1e6 pixels")
    var hr = rows * scale
    var hc = cols * scale
    var num = List[Float64](length=hr * hc, fill=0.0)
    var den = List[Float64](length=hr * hc, fill=0.0)
    for t in range(len(frames)):
        for y in range(rows):
            for x in range(cols):
                var px = (Float64(x) - shifts.data[t * 2]) * Float64(scale)
                var py = (Float64(y) - shifts.data[t * 2 + 1]) * Float64(scale)
                var x0 = Int(floor(px))
                var y0 = Int(floor(py))
                var fx = px - Float64(x0)
                var fy = py - Float64(y0)
                var v = frames[t].data[y * cols + x]
                for j in range(2):
                    for i in range(2):
                        var gx = x0 + i
                        var gy = y0 + j
                        if 0 <= gx < hc and 0 <= gy < hr:
                            var w = (fx if i == 1 else 1.0 - fx) * (fy if j == 1 else 1.0 - fy)
                            num[gy * hc + gx] += w * v
                            den[gy * hc + gx] += w
    var out = Mat[DType.float64](hr, hc)
    for gy in range(hr):
        for gx in range(hc):
            if den[gy * hc + gx] > 1e-9:
                out.data[gy * hc + gx] = num[gy * hc + gx] / den[gy * hc + gx]
            else:  # bilinear upsample of the reference, clamped at the border
                var sx = min(Float64(gx) / Float64(scale), Float64(cols - 1))
                var sy = min(Float64(gy) / Float64(scale), Float64(rows - 1))
                var ix = min(Int(sx), cols - 2) if cols > 1 else 0
                var iy = min(Int(sy), rows - 2) if rows > 1 else 0
                var ax = sx - Float64(ix)
                var ay = sy - Float64(iy)
                var ix1 = min(ix + 1, cols - 1)
                var iy1 = min(iy + 1, rows - 1)
                ref r = frames[0].data
                out.data[gy * hc + gx] = (
                    (1.0 - ay) * ((1.0 - ax) * r[iy * cols + ix] + ax * r[iy * cols + ix1])
                    + ay * ((1.0 - ax) * r[iy1 * cols + ix] + ax * r[iy1 * cols + ix1])
                )
    return out^
