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


def run(mut a: Args) raises -> Bool:
    if not (a.op in ["granulometry", "glcm", "lbpHist", "resizeMmpx"]):
        return False
    if a.op == "granulometry" or a.op == "resizeMmpx":
        if peek_code(a.ins[0]) == 1:
            return _run_dt[DType.float32](a)
        return _run_dt[DType.uint8](a)
    var m = load[DType.uint8](a.ins[0])
    var o = Mat[F64](0, 0)
    while a.again():
        o = glcm(m, a.i("levels", 16), a.i("dx", 1), a.i("dy", 0)) if a.op == "glcm" else lbp_hist(m)
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
    else:
        var o = Mat[dt](0, 0)
        while a.again():
            o = resize_mmpx(m, a.f("mmpx_in", 1.0), a.f("mmpx_out", 1.0))
        save(o, a.out)
    return True
