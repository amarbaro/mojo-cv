# Orthonormal 3D Haar wavelet for static voxel grids. A grid of side n (a power of two) is a Mat[float64] with
# rows = z, cols = y, ch = x, so element (z, y, x) sits at ((z * n + y) * n + x). Each level transforms the
# low-pass cube along x, y and z (low half first, high half second), then recurses into the (m/2)^3 corner.
# Compression keeps the coefficients with |c| >= thr; orthonormality makes the squared error the dropped energy.
from std.math import sqrt
from mojocv.mat import Mat

comptime R: Float64 = 0.7071067811865476


def _line[inverse: Bool](mut d: List[Float64], base: Int, stride: Int, m: Int, mut tmp: List[Float64]):
    var h = m // 2
    for i in range(h):
        comptime if inverse:
            var lo = d[base + i * stride]
            var hi = d[base + (h + i) * stride]
            tmp[2 * i] = (lo + hi) * R
            tmp[2 * i + 1] = (lo - hi) * R
        else:
            var a = d[base + 2 * i * stride]
            var b = d[base + (2 * i + 1) * stride]
            tmp[i] = (a + b) * R
            tmp[h + i] = (a - b) * R
    for i in range(m):
        d[base + i * stride] = tmp[i]


def _axis[inverse: Bool](mut d: List[Float64], n: Int, m: Int, axis: Int, mut tmp: List[Float64]):
    """Transform every line of the m^3 corner cube along one axis (0 = x, 1 = y, 2 = z)."""
    var stride = n ** axis
    var s1 = n ** ((axis + 1) % 3)
    var s2 = n ** ((axis + 2) % 3)
    for a in range(m):
        for b in range(m):
            _line[inverse](d, a * s1 + b * s2, stride, m, tmp)


def levels_of(n: Int) raises -> Int:
    var l = 0
    while (1 << l) < n:
        l += 1
    if (1 << l) != n or n < 2:
        raise Error(String("mojocv: haar3 wants a cubic grid with a power-of-two side >= 2, got ", n))
    return l


def _check(v: Mat[DType.float64]) raises -> Int:
    if v.rows != v.cols or v.rows != v.ch:
        raise Error("mojocv: haar3 wants a grid stored as rows = cols = ch = n")
    return v.rows


def haar3[inverse: Bool](v: Mat[DType.float64], levels: Int = 0) raises -> Mat[DType.float64]:
    var n = _check(v)
    var full = levels_of(n)
    var nl = levels if levels > 0 else full
    if levels < 0 or nl > full:
        raise Error("mojocv: haar3 levels must be between 1 and log2(n) (0 = all)")
    var out = Mat[DType.float64](v.rows, v.cols, v.ch)
    out.data = v.data.copy()
    var tmp = List[Float64](length=n, fill=0.0)
    for k in range(nl):
        var m = n >> (k if not inverse else nl - 1 - k)
        comptime for ax in range(3):
            _axis[inverse](out.data, n, m, ax if not inverse else 2 - ax, tmp)
    return out^


def compress(v: Mat[DType.float64], thr: Float64, levels: Int = 0) raises -> Tuple[Mat[DType.float64], Int]:
    """Forward, drop |c| < thr, inverse. Returns the reconstruction and the number of kept coefficients."""
    var c = haar3[False](v, levels)
    var kept = 0
    for i in range(len(c.data)):
        if abs(c.data[i]) < thr:
            c.data[i] = 0.0
        else:
            kept += 1
    return (haar3[True](c, levels), kept)
