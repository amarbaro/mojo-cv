# granulo: size and texture measures for granular images (soil, sediment, powders).
# granulometry: sum of the image after a morphological opening by a disc of each radius (Matheron's
# granulometry; consecutive differences are the pattern spectrum, i.e. the size distribution by area).
# glcm: grey-level co-occurrence features with scikit-image's definitions. lbp_hist: 8-neighbour local binary
# pattern histogram. resize_mmpx: resample from one mm-per-pixel scale to another.
from std.math import sqrt
from mojocv.mat import Mat, load, save
from mojocv.core import Args, peek_code
from mojocv.imgproc import morph, resize

comptime F64 = DType.float64
comptime DY = SIMD[DType.int64, 8](-1, -1, -1, 0, 1, 1, 1, 0)
comptime DX = SIMD[DType.int64, 8](-1, 0, 1, 1, 1, 0, -1, -1)
comptime FW = SIMD[DType.int64, 4](0, 1, 2, 7)  # neighbours before p in raster order
comptime BW = SIMD[DType.int64, 4](3, 4, 5, 6)  # neighbours after p


def granulometry[dt: DType](src: Mat[dt], radii: List[Int]) -> Mat[F64]:
    """1 x len(radii): sum of all pixels after opening with an elliptic (2r+1)^2 element; r = 0 is the source."""
    var o = Mat[F64](1, len(radii), 1)
    for j in range(len(radii)):
        var r = radii[j]
        var s = 0.0
        if r <= 0:
            for v in src.data:
                s += Float64(v)
        else:
            var op = morph[False, dt](morph[True, dt](src, 2, 2 * r + 1, 1), 2, 2 * r + 1, 1)
            for v in op.data:
                s += Float64(v)
        o.data[j] = s
    return o^


def glcm(src: Mat[DType.uint8], levels: Int, dx: Int, dy: Int) -> Mat[F64]:
    """Symmetric normalised co-occurrence of (p, p + (dy, dx)) after quantising to `levels`; returns
    1 x 5: contrast, dissimilarity, homogeneity (1 / (1 + d^2)), energy (sqrt ASM), correlation."""
    var L = levels
    var m = List[Float64](length=L * L, fill=0.0)
    var p = src.ptr()
    var n = 0
    for y in range(max(0, -dy), min(src.rows, src.rows - dy)):
        for x in range(max(0, -dx), min(src.cols, src.cols - dx)):
            var a = Int(p[y * src.cols + x]) * L // 256
            var b = Int(p[(y + dy) * src.cols + x + dx]) * L // 256
            m[a * L + b] += 1.0
            m[b * L + a] += 1.0
            n += 2
    var con = 0.0
    var dis = 0.0
    var hom = 0.0
    var asm = 0.0
    var mu = 0.0
    for i in range(L):
        for j in range(L):
            var q = m[i * L + j] / Float64(n)
            var d = Float64(i - j)
            con += q * d * d
            dis += q * abs(d)
            hom += q / (1.0 + d * d)
            asm += q * q
            mu += q * Float64(i)
    var var_ = 0.0
    var cov = 0.0
    for i in range(L):
        for j in range(L):
            var q = m[i * L + j] / Float64(n)
            var_ += q * (Float64(i) - mu) ** 2
            cov += q * (Float64(i) - mu) * (Float64(j) - mu)
    var o = Mat[F64](1, 5, 1)
    o.data[0] = con
    o.data[1] = dis
    o.data[2] = hom
    o.data[3] = sqrt(asm)
    o.data[4] = cov / var_ if var_ > 0 else 1.0
    return o^


def lbp_hist(src: Mat[DType.uint8]) -> Mat[F64]:
    """1 x 256 normalised histogram of 8-neighbour LBP codes over interior pixels; bit k is set when neighbour k
    (clockwise from top-left) is >= the centre."""
    var h = List[Int](length=256, fill=0)
    var p = src.ptr()
    var w = src.cols
    for y in range(1, src.rows - 1):
        for x in range(1, w - 1):
            var i = y * w + x
            var c = p[unsafe_offset=i]
            var code = 0
            comptime for k in range(8):
                if p[unsafe_offset=i + Int(DY[k]) * w + Int(DX[k])] >= c:
                    code |= 1 << k
            h[code] += 1
    var o = Mat[F64](1, 256, 1)
    var t = Float64(max(1, (src.rows - 2) * (w - 2)))
    for k in range(256):
        o.data[k] = Float64(h[k]) / t
    return o^


def resize_mmpx[dt: DType](src: Mat[dt], mmpx_in: Float64, mmpx_out: Float64) -> Mat[dt]:
    """Resample so one pixel covers mmpx_out mm: area averaging when shrinking, linear when growing."""
    var f = mmpx_in / mmpx_out
    var w = max(1, Int(Float64(src.cols) * f + 0.5))
    var h = max(1, Int(Float64(src.rows) * f + 0.5))
    return resize(src, w, h, 4 if f < 1.0 else 1)


comptime SIEVES = SIMD[F64, 16](0.002, 0.0063, 0.02, 0.063, 0.2, 0.63, 2.0, 6.3, 20.0, 63.0, 200.0, 0, 0, 0, 0, 0)


def _mean[dt: DType](m: Mat[dt]) -> Float64:
    var t = 0.0
    for v in m.data:
        t += Float64(v)
    return t / Float64(max(1, len(m.data)))


def granulometry_mm[dt: DType](src: Mat[dt], ppm: Float64) -> Mat[F64]:
    """1 x 11 at the DIN EN ISO 14688-1 sieves: % of the image mean removed by opening with a square of side
    2 * round(d * ppm / 2) + 1 px. A square, like a sieve mesh; nested squares keep the curve monotone, and a square
    is separable, so a 63 mm sieve at 20 px/mm costs about 2 s on 24 MP. -1: half-side under 1 px (too fine to
    resolve); -2: the square is wider than half the image."""
    var o = Mat[F64](1, 11, 1)
    var m0 = _mean(src)
    for i in range(11):
        var r = Int(SIEVES[i] * ppm / 2.0 + 0.5)
        if r < 1:
            o.data[i] = -1.0
        elif 2 * r + 1 > min(src.rows, src.cols) // 2:
            o.data[i] = -2.0
        else:
            o.data[i] = 100.0 * (1.0 - _mean(morph[False, dt](morph[True, dt](src, 0, 2 * r + 1, 1), 0, 2 * r + 1, 1)) / m0)
    return o^


def tophat[dt: DType](src: Mat[dt], k: Int) -> Mat[dt]:
    """White top-hat with a k x k ellipse: src - opening (flattens slow lighting gradients)."""
    var op = morph[False, dt](morph[True, dt](src, 2, k, 1), 2, k, 1)
    for i in range(len(op.data)):
        op.data[i] = src.data[i] - op.data[i]
    return op^


def _tile_var(v: List[Float64], rows: Int, cols: Int, t: Int) -> Mat[F64]:
    """Variance of each full t x t tile (partial edge tiles dropped), as a (rows // t) x (cols // t) map."""
    var o = Mat[F64](rows // t, cols // t, 1)
    for ty in range(rows // t):
        for tx in range(cols // t):
            var s = 0.0
            var q = 0.0
            for y in range(ty * t, ty * t + t):
                for x in range(tx * t, tx * t + t):
                    var a = v[y * cols + x]
                    s += a
                    q += a * a
            var n = Float64(t * t)
            o.data[ty * (cols // t) + tx] = q / n - (s / n) ** 2
    return o^


def smooth_fraction(src: Mat[DType.uint8], t: Int, thr: Float64) -> Mat[F64]:
    """1 x 1: share of full t x t tiles whose grey variance is below thr (texture finer than a pixel)."""
    var v = List[Float64](capacity=len(src.data))
    for a in src.data:
        v.append(Float64(a))
    var m = _tile_var(v, src.rows, src.cols, t)
    var n = 0
    for a in m.data:
        if a < thr:
            n += 1
    var o = Mat[F64](1, 1, 1)
    o.data[0] = Float64(n) / Float64(max(1, len(m.data)))
    return o^


def sharp_map(src: Mat[DType.uint8], t: Int) -> Mat[F64]:
    """Variance of the 4-neighbour Laplacian (cv2.Laplacian ksize=1, reflect-101 border) per full t x t tile."""
    var R = src.rows
    var C = src.cols
    var p = src.ptr()
    var v = List[Float64](length=R * C, fill=0.0)
    for y in range(R):
        var yu = 1 if y == 0 else y - 1
        var yd = R - 2 if y == R - 1 else y + 1
        for x in range(C):
            var xl = 1 if x == 0 else x - 1
            var xr = C - 2 if x == C - 1 else x + 1
            v[y * C + x] = Float64(p[yu * C + x]) + Float64(p[yd * C + x]) + Float64(p[y * C + xl]) + Float64(p[y * C + xr]) - 4.0 * Float64(p[y * C + x])
    return _tile_var(v, R, C, t)


def autocorr(src: Mat[DType.uint8], lags: List[Int]) -> Mat[F64]:
    """1 x len(lags): Pearson correlation of the image with itself shifted by L px, mean of the horizontal and
    vertical shifts (Rubin 2004: the curve's decay length tracks grain size)."""
    var o = Mat[F64](1, len(lags), 1)
    var p = src.ptr()
    var C = src.cols
    for j in range(len(lags)):
        var r = 0.0
        for axis in range(2):
            var L = lags[j]
            var dy = L if axis == 1 else 0
            var dx = L if axis == 0 else 0
            var sa = 0.0
            var sb = 0.0
            var saa = 0.0
            var sbb = 0.0
            var sab = 0.0
            var n = 0
            for y in range(src.rows - dy):
                for x in range(C - dx):
                    var a = Float64(p[y * C + x])
                    var b = Float64(p[(y + dy) * C + x + dx])
                    sa += a
                    sb += b
                    saa += a * a
                    sbb += b * b
                    sab += a * b
                    n += 1
            var nf = Float64(n)
            var cov = sab / nf - sa * sb / nf / nf
            var va = saa / nf - (sa / nf) ** 2
            var vb = sbb / nf - (sb / nf) ** 2
            r += cov / sqrt(va * vb) if va > 0 and vb > 0 else 1.0
        o.data[j] = r / 2.0
    return o^


def wavelet_energy(src: Mat[DType.uint8], levels: Int) -> Mat[F64]:
    """1 x levels: share of the image's detail energy (LH + HL + HH) at each orthonormal 2D Haar level, finest
    first, over the top-left crop whose sides are multiples of 2^levels."""
    var R = (src.rows >> levels) << levels
    var C = (src.cols >> levels) << levels
    var a = List[Float64](length=R * C, fill=0.0)
    for y in range(R):
        for x in range(C):
            a[y * C + x] = Float64(src.data[y * src.cols + x])
    var e = List[Float64](length=levels, fill=0.0)
    var h = R
    var w = C
    for lv in range(levels):
        var h2 = h // 2
        var w2 = w // 2
        var lo = List[Float64](length=h2 * w2, fill=0.0)
        for y in range(h2):
            for x in range(w2):
                var p00 = a[(2 * y) * C + 2 * x]
                var p01 = a[(2 * y) * C + 2 * x + 1]
                var p10 = a[(2 * y + 1) * C + 2 * x]
                var p11 = a[(2 * y + 1) * C + 2 * x + 1]
                lo[y * w2 + x] = (p00 + p01 + p10 + p11) / 2.0
                var lh = (p00 - p01 + p10 - p11) / 2.0
                var hl = (p00 + p01 - p10 - p11) / 2.0
                var hh = (p00 - p01 - p10 + p11) / 2.0
                e[lv] += lh * lh + hl * hl + hh * hh
        for y in range(h2):
            for x in range(w2):
                a[y * C + x] = lo[y * w2 + x]
        h = h2
        w = w2
    var tot = 0.0
    for v in e:
        tot += v
    var o = Mat[F64](1, levels, 1)
    for i in range(levels):
        o.data[i] = e[i] / tot if tot > 0 else 0.0
    return o^


def _rim(v: List[Float64]) -> Tuple[Int, Int]:
    """Inner edges of a dark border band in a profile: a run of >= max(3, n / 333) samples darker than 0.55 x the
    median of the middle half, then the first sample brighter than 0.75 x that median; -1 where there is none."""
    var n = len(v)
    var mid = List[Float64]()
    for i in range(n // 4, 3 * n // 4 + 1):
        mid.append(v[i])
    sort(mid)
    var m = mid[len(mid) // 2]
    var lo = 0.55 * m
    var hi = 0.75 * m
    var run = max(3, n // 333)
    var a = -1
    var i = 0
    while i < n and v[i] < lo:
        i += 1
    if i >= run:
        while i < n and v[i] < hi:
            i += 1
        a = i
    var b = -1
    var j = n - 1
    while j >= 0 and v[j] < lo:
        j -= 1
    if n - 1 - j >= run:
        while j >= 0 and v[j] < hi:
            j -= 1
        b = j
    return (a, b)


def tray_rim(src: Mat[DType.uint8]) -> Mat[F64]:
    """1 x 6: left, right, top, bottom inner rim edges (px) and the spans right - left, bottom - top; -1 = none.
    Profiles are exact column and row means of the grey image."""
    var cm = List[Float64](length=src.cols, fill=0.0)
    var rm = List[Float64](length=src.rows, fill=0.0)
    for y in range(src.rows):
        for x in range(src.cols):
            var a = Float64(src.data[y * src.cols + x])
            cm[x] += a
            rm[y] += a
    var c = _rim(cm)
    var r = _rim(rm)
    var o = Mat[F64](1, 6, 1)
    o.data[0] = Float64(c[0])
    o.data[1] = Float64(c[1])
    o.data[2] = Float64(r[0])
    o.data[3] = Float64(r[1])
    o.data[4] = Float64(c[1] - c[0]) if c[0] >= 0 and c[1] >= 0 else -1.0
    o.data[5] = Float64(r[1] - r[0]) if r[0] >= 0 and r[1] >= 0 else -1.0
    return o^


def tray_crop[dt: DType](src: Mat[dt], margin: Int) -> Mat[dt]:
    """Crop to the inside of the tray rim found by tray_rim, shrunk by `margin` px on every side that has a rim;
    sides without a rim keep the frame edge. Returns a copy when no rim is found."""
    var g = Mat[DType.uint8](src.rows, src.cols, 1)
    for i in range(src.rows * src.cols):
        var a = 0.0
        for c in range(src.ch):
            a += Float64(src.data[i * src.ch + c])
        g.data[i] = UInt8(Int(a / Float64(src.ch) + 0.5))
    var e = tray_rim(g)
    var x0 = Int(e.data[0]) + margin if e.data[0] >= 0 else 0
    var x1 = Int(e.data[1]) - margin if e.data[1] >= 0 else src.cols - 1
    var y0 = Int(e.data[2]) + margin if e.data[2] >= 0 else 0
    var y1 = Int(e.data[3]) - margin if e.data[3] >= 0 else src.rows - 1
    if x1 <= x0 or y1 <= y0:
        return Mat[dt](copy=src)
    var o = Mat[dt](y1 - y0 + 1, x1 - x0 + 1, src.ch, False)
    var w = (x1 - x0 + 1) * src.ch
    for y in range(y0, y1 + 1):
        for k in range(w):
            o.data[(y - y0) * w + k] = src.data[(y * src.cols + x0) * src.ch + k]
    return o^


def reconstruct(marker: Mat[DType.uint8], mask: Mat[DType.uint8]) -> Mat[DType.uint8]:
    """Grey reconstruction by dilation of marker under mask, 8-connected (Vincent 1993 hybrid: raster, anti-raster,
    then a FIFO). marker must be <= mask."""
    var R = mask.rows
    var C = mask.cols
    var J = Mat[DType.uint8](copy=marker)
    var I = mask.ptr()
    var p = J.ptr()
    for y in range(R):
        for x in range(C):
            var v = p[y * C + x]
            comptime for t in range(4):
                var yy = y + Int(DY[Int(FW[t])])
                var xx = x + Int(DX[Int(FW[t])])
                if yy >= 0 and xx >= 0 and xx < C:
                    v = max(v, p[yy * C + xx])
            p[y * C + x] = min(v, I[y * C + x])
    var q = List[UInt32]()
    for y in range(R - 1, -1, -1):
        for x in range(C - 1, -1, -1):
            var v = p[y * C + x]
            comptime for t in range(4):
                var yy = y + Int(DY[Int(BW[t])])
                var xx = x + Int(DX[Int(BW[t])])
                if yy < R and xx >= 0 and xx < C:
                    v = max(v, p[yy * C + xx])
            v = min(v, I[y * C + x])
            p[y * C + x] = v
            comptime for t in range(4):
                var yy = y + Int(DY[Int(BW[t])])
                var xx = x + Int(DX[Int(BW[t])])
                if yy < R and xx >= 0 and xx < C:
                    var w = p[yy * C + xx]
                    if w < v and w < I[yy * C + xx]:
                        q.append(UInt32(y * C + x))
    var head = 0
    while head < len(q):
        if head >= 1 << 22 and 2 * head >= len(q):
            # drop the consumed prefix so the FIFO stays near its live size
            var rest = List[UInt32](capacity=len(q) - head)
            for t in range(head, len(q)):
                rest.append(q[t])
            q = rest^
            head = 0
        var i = Int(q[head])
        head += 1
        var y = i // C
        var x = i % C
        var v = p[i]
        comptime for k in range(8):
            var yy = y + Int(DY[k])
            var xx = x + Int(DX[k])
            if yy >= 0 and yy < R and xx >= 0 and xx < C:
                var j = yy * C + xx
                if p[j] < v and I[j] != p[j]:
                    p[j] = min(v, I[j])
                    q.append(UInt32(j))
    _ = mask.rows
    return J^


def granulometry_rec(src: Mat[DType.uint8], radii: List[Int]) -> Mat[F64]:
    """Like granulometry, with opening by reconstruction: erode by the disc, then reconstruct under the source, so
    surviving grains keep their exact shape."""
    var o = Mat[F64](1, len(radii), 1)
    for j in range(len(radii)):
        var r = radii[j]
        var s = 0.0
        if r <= 0:
            for v in src.data:
                s += Float64(v)
        else:
            var rec = reconstruct(morph[True, DType.uint8](src, 2, 2 * r + 1, 1), src)
            for v in rec.data:
                s += Float64(v)
        o.data[j] = s
    return o^


def run(mut a: Args) raises -> Bool:
    if not (a.op in ["granulometry", "glcm", "lbpHist", "resizeMmpx", "granulometryMm", "tophat", "smoothFraction", "sharpMap", "autocorr", "waveletEnergy", "trayRim", "granulometryRec", "trayCrop"]):
        return False
    if a.op in ["granulometry", "resizeMmpx", "granulometryMm", "tophat", "trayCrop"]:
        if peek_code(a.ins[0]) == 1:
            return _run_dt[DType.float32](a)
        return _run_dt[DType.uint8](a)
    var m = load[DType.uint8](a.ins[0])
    var o = Mat[F64](0, 0)
    while a.again():
        if a.op == "glcm":
            o = glcm(m, a.i("levels", 16), a.i("dx", 1), a.i("dy", 0))
        elif a.op == "lbpHist":
            o = lbp_hist(m)
        elif a.op == "smoothFraction":
            o = smooth_fraction(m, a.i("tile", 16), a.f("thr", 4.0))
        elif a.op == "sharpMap":
            o = sharp_map(m, a.i("tile", 64))
        elif a.op == "autocorr" or a.op == "granulometryRec":
            var rl = List[Int]()
            for v in a.fl("lags" if a.op == "autocorr" else "radii", [1.0, 2.0, 4.0, 8.0]):
                rl.append(Int(v))
            o = autocorr(m, rl) if a.op == "autocorr" else granulometry_rec(m, rl)
        elif a.op == "waveletEnergy":
            o = wavelet_energy(m, a.i("levels", 5))
        else:
            o = tray_rim(m)
    save(o, a.out)
    return True


def _run_dt[dt: DType](mut a: Args) raises -> Bool:
    var m = load[dt](a.ins[0])
    if a.op == "granulometry":
        var rl = List[Int]()
        for v in a.fl("radii", [0.0, 1.0, 2.0, 4.0]):
            rl.append(Int(v))
        var o = Mat[F64](0, 0)
        while a.again():
            o = granulometry(m, rl)
        save(o, a.out)
    elif a.op == "granulometryMm":
        var o = Mat[F64](0, 0)
        while a.again():
            o = granulometry_mm(m, a.f("ppm", 1.0))
        save(o, a.out)
    else:
        var o = Mat[dt](0, 0)
        while a.again():
            if a.op == "tophat":
                o = tophat(m, a.i("ksize", 15))
            elif a.op == "trayCrop":
                o = tray_crop(m, a.i("margin", 8))
            else:
                o = resize_mmpx(m, a.f("mmpx_in", 1.0), a.f("mmpx_out", 1.0))
        save(o, a.out)
    return True
