# CAE post-processing of a stress field. A field is N x 6 float64 in Voigt order [xx, yy, zz, xy, yz, xz].
#   stress_measures -> N x 6 [hydrostatic (tr/3), von Mises, s1, s2, s3 (descending), max shear (s1 - s3) / 2]
#   colormap        -> N x 3 RGB in [0, 1], blue - cyan - green - yellow - red over [lo, hi] (clamped outside)
# Principal values by the trigonometric solution of the characteristic cubic on the deviator (Smith 1961).
from std.math import acos, cos, sqrt, pi
from mojocv.mat import Mat


def _principal(xx: Float64, yy: Float64, zz: Float64, xy: Float64, yz: Float64, xz: Float64) -> Tuple[Float64, Float64, Float64]:
    var q = (xx + yy + zz) / 3.0
    var off = xy * xy + yz * yz + xz * xz
    var a = xx - q
    var b = yy - q
    var c = zz - q
    var p2 = a * a + b * b + c * c + 2.0 * off
    if p2 <= 1e-300:  # hydrostatic: three equal values
        return (q, q, q)
    var p = sqrt(p2 / 6.0)
    # det(B) with B = (S - q I) / p
    var det = a * (b * c - yz * yz) - xy * (xy * c - yz * xz) + xz * (xy * yz - b * xz)
    var r = det / (p * p * p) / 2.0
    r = min(max(r, -1.0), 1.0)
    var phi = acos(r) / 3.0
    var e1 = q + 2.0 * p * cos(phi)
    var e3 = q + 2.0 * p * cos(phi + 2.0 * pi / 3.0)
    return (e1, 3.0 * q - e1 - e3, e3)


def stress_measures(s: Mat[DType.float64]) raises -> Mat[DType.float64]:
    if s.cols != 6 or s.ch != 1:
        raise Error("mojocv: stress wants an N x 6 float64 field [xx, yy, zz, xy, yz, xz]")
    var out = Mat[DType.float64](s.rows, 6)
    for i in range(s.rows):
        var xx = s.data[i * 6]
        var yy = s.data[i * 6 + 1]
        var zz = s.data[i * 6 + 2]
        var xy = s.data[i * 6 + 3]
        var yz = s.data[i * 6 + 4]
        var xz = s.data[i * 6 + 5]
        for k in range(6):
            if s.data[i * 6 + k] != s.data[i * 6 + k]:
                raise Error(String("mojocv: stress: NaN in row ", i))
        var e = _principal(xx, yy, zz, xy, yz, xz)
        out.data[i * 6] = (xx + yy + zz) / 3.0
        out.data[i * 6 + 1] = sqrt(
            0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3.0 * (xy * xy + yz * yz + xz * xz)
        )
        out.data[i * 6 + 2] = e[0]
        out.data[i * 6 + 3] = e[1]
        out.data[i * 6 + 4] = e[2]
        out.data[i * 6 + 5] = 0.5 * (e[0] - e[2])
    return out^


def colormap(v: Mat[DType.float64], lo: Float64, hi: Float64) raises -> Mat[DType.float64]:
    if v.cols != 1 or v.ch != 1 or not (hi > lo):
        raise Error("mojocv: colormap wants an N x 1 float64 field and lo < hi")
    var out = Mat[DType.float64](v.rows, 3)
    # stops at t = 0, .25, .5, .75, 1: (0,0,1) (0,1,1) (0,1,0) (1,1,0) (1,0,0)
    var stops: List[Float64] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0]
    for i in range(v.rows):
        var x = v.data[i]
        if x != x:
            raise Error(String("mojocv: colormap: NaN in row ", i))
        var t = min(max((x - lo) / (hi - lo), 0.0), 1.0) * 4.0
        var k = min(Int(t), 3)
        var f = t - Float64(k)
        for c in range(3):
            out.data[i * 3 + c] = stops[k * 3 + c] * (1.0 - f) + stops[(k + 1) * 3 + c] * f
    return out^
