# Least-squares similarity (rotation R, translation t, scale c) between two N x 3 point sets, Umeyama 1991.
# dst_i ~ c R src_i + t. 3x3 matrices are row-major List[Float64] of 9.
from std.math import sqrt
from mojocv.mat import Mat


def _det3(m: List[Float64]) -> Float64:
    return (
        m[0] * (m[4] * m[8] - m[5] * m[7])
        - m[1] * (m[3] * m[8] - m[5] * m[6])
        + m[2] * (m[3] * m[7] - m[4] * m[6])
    )


def _mul3(a: List[Float64], b: List[Float64], ta: Bool = False, tb: Bool = False) -> List[Float64]:
    """a * b with optional transposes, one body for every product the file needs."""
    var r = List[Float64](length=9, fill=0.0)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                var x = a[k * 3 + i] if ta else a[i * 3 + k]
                var y = b[j * 3 + k] if tb else b[k * 3 + j]
                r[i * 3 + j] += x * y
    return r^


struct SVD3(Movable):
    """a = u * diag(d) * v^T with d descending; u and v orthogonal (a missing third direction, rank 2, is completed)."""
    var u: List[Float64]
    var d: List[Float64]
    var v: List[Float64]

    def __init__(out self, a: List[Float64]) raises:
        # One-sided Jacobi (Hestenes): rotate column pairs of w until they are orthogonal, v collects the rotations.
        var w = a.copy()
        self.v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        for _ in range(60):
            var rotated = False
            for p in range(2):
                for q in range(p + 1, 3):
                    var alpha = 0.0
                    var beta = 0.0
                    var gamma = 0.0
                    for i in range(3):
                        alpha += w[i * 3 + p] * w[i * 3 + p]
                        beta += w[i * 3 + q] * w[i * 3 + q]
                        gamma += w[i * 3 + p] * w[i * 3 + q]
                    if abs(gamma) <= 1e-15 * sqrt(alpha * beta) or gamma == 0.0:
                        continue
                    rotated = True
                    var zeta = (beta - alpha) / (2.0 * gamma)
                    var t = (1.0 if zeta >= 0.0 else -1.0) / (abs(zeta) + sqrt(1.0 + zeta * zeta))
                    var c = 1.0 / sqrt(1.0 + t * t)
                    var s = c * t
                    for i in range(3):
                        var wp = w[i * 3 + p]
                        var wq = w[i * 3 + q]
                        w[i * 3 + p] = c * wp - s * wq
                        w[i * 3 + q] = s * wp + c * wq
                        var vp = self.v[i * 3 + p]
                        var vq = self.v[i * 3 + q]
                        self.v[i * 3 + p] = c * vp - s * vq
                        self.v[i * 3 + q] = s * vp + c * vq
            if not rotated:
                break
        var norm = List[Float64]()
        for j in range(3):
            norm.append(sqrt(w[j] * w[j] + w[3 + j] * w[3 + j] + w[6 + j] * w[6 + j]))
        var order = [0, 1, 2]
        for i in range(2):  # sort descending, three elements
            for j in range(2 - i):
                if norm[order[j]] < norm[order[j + 1]]:
                    var t = order[j]
                    order[j] = order[j + 1]
                    order[j + 1] = t
        self.u = List[Float64](length=9, fill=0.0)
        self.d = List[Float64]()
        var v2 = List[Float64](length=9, fill=0.0)
        var tiny = 1e-13 * norm[order[0]]
        var missing = -1
        if norm[order[1]] <= tiny:
            raise Error("mojocv: rank 1 (collinear points) leaves the rotation undetermined")
        for k in range(3):
            var j = order[k]
            self.d.append(norm[j])
            for i in range(3):
                v2[i * 3 + k] = self.v[i * 3 + j]
                if norm[j] > tiny:
                    self.u[i * 3 + k] = w[i * 3 + j] / norm[j]
            if norm[j] <= tiny:
                missing = k
                self.d[k] = 0.0
        self.v = v2^
        if missing == 2:  # rank 2: the last direction is the cross product of the first two
            for i in range(3):
                var a1 = (i + 1) % 3
                var a2 = (i + 2) % 3
                self.u[i * 3 + 2] = self.u[a1 * 3] * self.u[a2 * 3 + 1] - self.u[a2 * 3] * self.u[a1 * 3 + 1]


def umeyama(src: Mat[DType.float64], dst: Mat[DType.float64], with_scale: Bool) raises -> Mat[DType.float64]:
    """4 x 4 homogeneous similarity [cR t; 0 1] with dst ~ c R src + t."""
    if src.cols != 3 or dst.cols != 3 or src.ch != 1 or dst.ch != 1 or src.rows != dst.rows or src.rows < 3:
        raise Error("mojocv: umeyama wants two N x 3 float64 clouds with the same N >= 3")
    var n = Float64(src.rows)
    var mx = List[Float64](length=3, fill=0.0)
    var my = List[Float64](length=3, fill=0.0)
    for i in range(src.rows):
        for a in range(3):
            mx[a] += src.data[i * 3 + a] / n
            my[a] += dst.data[i * 3 + a] / n
    var cov = List[Float64](length=9, fill=0.0)
    var var_x = 0.0
    for i in range(src.rows):
        for a in range(3):
            var xa = src.data[i * 3 + a] - mx[a]
            var_x += xa * xa / n
            for b in range(3):
                cov[b * 3 + a] += (dst.data[i * 3 + b] - my[b]) * xa / n  # sum y x^T
    if var_x <= 0.0:
        raise Error("mojocv: umeyama needs a source cloud with nonzero spread")
    var f = SVD3(cov)
    var s = [1.0, 1.0, 1.0]
    if _det3(f.u) * _det3(f.v) < 0.0:
        s[2] = -1.0
    var us = f.u.copy()
    for i in range(3):
        for k in range(3):
            us[i * 3 + k] *= s[k]
    var r = _mul3(us, f.v, False, True)
    var c = 1.0
    if with_scale:
        c = (f.d[0] * s[0] + f.d[1] * s[1] + f.d[2] * s[2]) / var_x
    var m = Mat[DType.float64](4, 4)
    for i in range(3):
        for j in range(3):
            m.data[i * 4 + j] = c * r[i * 3 + j]
        m.data[i * 4 + 3] = my[i] - c * (r[i * 3] * mx[0] + r[i * 3 + 1] * mx[1] + r[i * 3 + 2] * mx[2])
    m.data[15] = 1.0
    return m^
