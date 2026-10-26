# B-spline / NURBS surfaces on clamped uniform knots (Piegl & Tiller, The NURBS Book, A2.1, A2.3, A4.4).
# A control net is a Mat[float64] nu x nv x 4 holding [x, y, z, w] (w = 1 for a plain B-spline), degrees p and q
# in u and v. Sample grids are M x N x 3. Parameters are uniform on [0, 1]: sample k of M sits at k / (M - 1).
from std.math import sqrt
from mojocv.mat import Mat


def knots(n: Int, p: Int) -> List[Float64]:
    """Clamped uniform knot vector of n control points and degree p: p + 1 zeros, uniform interior, p + 1 ones."""
    var u = List[Float64]()
    for j in range(n + p + 1):
        u.append(0.0 if j <= p else (1.0 if j >= n else Float64(j - p) / Float64(n - p)))
    return u^


def span(n: Int, p: Int, u: Float64, kn: List[Float64]) -> Int:
    if u >= kn[n]:
        return n - 1
    var s = p
    while kn[s + 1] <= u:
        s += 1
    return s


def basis(sp: Int, u: Float64, p: Int, kn: List[Float64]) -> List[Float64]:
    """Values and first derivatives of the p + 1 non-zero basis functions: [N_0..N_p, N'_0..N'_p] (A2.3, n = 1)."""
    var q = p + 1
    var ndu = List[Float64](length=q * q, fill=0.0)
    var left = List[Float64](length=q, fill=0.0)
    var right = List[Float64](length=q, fill=0.0)
    ndu[0] = 1.0
    for j in range(1, q):
        left[j] = u - kn[sp + 1 - j]
        right[j] = kn[sp + j] - u
        var saved = 0.0
        for r in range(j):
            ndu[j * q + r] = right[r + 1] + left[j - r]
            var temp = ndu[r * q + j - 1] / ndu[j * q + r]
            ndu[r * q + j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j * q + j] = saved
    var out = List[Float64](length=2 * q, fill=0.0)
    for j in range(q):
        out[j] = ndu[j * q + p]
    if p == 0:
        return out^
    for r in range(q):  # first derivative: the k = 1 pass of A2.3
        var d = 0.0
        var pk = p - 1
        var rk = r - 1
        var a0 = 1.0
        if r >= 1:
            a0 = a0 / ndu[(pk + 1) * q + rk]
            d = a0 * ndu[rk * q + pk]
        if r <= pk:
            d += (-1.0 / ndu[(pk + 1) * q + r]) * ndu[r * q + pk]
        out[q + r] = d * Float64(p)
    return out^


def eval_surface(
    net: Mat[DType.float64], p: Int, q: Int, m: Int, n: Int
) raises -> Mat[DType.float64]:
    """m x n x 6 grid of [point, unit normal] at uniform parameters. NURBS quotient rule for the derivatives."""
    var nu = net.rows
    var nv = net.cols
    if net.ch != 4 or nu <= p or nv <= q or p < 1 or q < 1 or m < 2 or n < 2:
        raise Error("mojocv: nurbs eval wants a nu x nv x 4 net with nu > p >= 1, nv > q >= 1 and a grid of at least 2 x 2")
    var ku = knots(nu, p)
    var kv = knots(nv, q)
    var out = Mat[DType.float64](m, n, 6)
    for a in range(m):
        var u = Float64(a) / Float64(m - 1)
        var su = span(nu, p, u, ku)
        var bu = basis(su, u, p, ku)
        for b in range(n):
            var v = Float64(b) / Float64(n - 1)
            var sv = span(nv, q, v, kv)
            var bv = basis(sv, v, q, kv)
            # homogeneous sums: h[0..3] = (wx, wy, wz, w), then d/du and d/dv of each
            var h = List[Float64](length=4, fill=0.0)
            var hu = List[Float64](length=4, fill=0.0)
            var hv = List[Float64](length=4, fill=0.0)
            for i in range(p + 1):
                for j in range(q + 1):
                    var idx = ((su - p + i) * nv + (sv - q + j)) * 4
                    var w = net.data[idx + 3]
                    for c in range(4):
                        var cp = net.data[idx + c] * w if c < 3 else w
                        h[c] += bu[i] * bv[j] * cp
                        hu[c] += bu[p + 1 + i] * bv[j] * cp
                        hv[c] += bu[i] * bv[q + 1 + j] * cp
            var nrm = List[Float64](length=3, fill=0.0)
            var s = List[Float64](length=3, fill=0.0)
            var du = List[Float64](length=3, fill=0.0)
            var dv = List[Float64](length=3, fill=0.0)
            for c in range(3):
                s[c] = h[c] / h[3]
                du[c] = (hu[c] - hu[3] * s[c]) / h[3]
                dv[c] = (hv[c] - hv[3] * s[c]) / h[3]
            nrm[0] = du[1] * dv[2] - du[2] * dv[1]
            nrm[1] = du[2] * dv[0] - du[0] * dv[2]
            nrm[2] = du[0] * dv[1] - du[1] * dv[0]
            var len = sqrt(nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2])
            for c in range(3):
                out.data[(a * n + b) * 6 + c] = s[c]
                out.data[(a * n + b) * 6 + 3 + c] = nrm[c] / len if len > 1e-300 else 0.0
    return out^


def _basis_matrix(count: Int, n: Int, p: Int) -> List[Float64]:
    """count x n matrix of basis values at count uniform parameters."""
    var kn = knots(n, p)
    var bm = List[Float64](length=count * n, fill=0.0)
    for k in range(count):
        var u = Float64(k) / Float64(count - 1)
        var sp = span(n, p, u, kn)
        var b = basis(sp, u, p, kn)
        for i in range(p + 1):
            bm[k * n + sp - p + i] = b[i]
    return bm^


def _solve_spd(mut a: List[Float64], n: Int, mut b: List[Float64], m: Int) raises:
    """Solve A X = B in place (B is n x m, overwritten by X) by Cholesky; A is n x n symmetric positive definite."""
    for j in range(n):
        var d = a[j * n + j]
        for k in range(j):
            d -= a[j * n + k] * a[j * n + k]
        if d <= 1e-14 * a[j * n + j] or d <= 0.0:
            raise Error("mojocv: nurbs fit is rank deficient: too few samples for the control net")
        a[j * n + j] = sqrt(d)
        for i in range(j + 1, n):
            var s = a[i * n + j]
            for k in range(j):
                s -= a[i * n + k] * a[j * n + k]
            a[i * n + j] = s / a[j * n + j]
    for c in range(m):
        for i in range(n):
            var s = b[i * m + c]
            for k in range(i):
                s -= a[i * n + k] * b[k * m + c]
            b[i * m + c] = s / a[i * n + i]
        for i in range(n - 1, -1, -1):
            var s = b[i * m + c]
            for k in range(i + 1, n):
                s -= a[k * n + i] * b[k * m + c]
            b[i * m + c] = s / a[i * n + i]


def _gram(bm: List[Float64], count: Int, n: Int) -> List[Float64]:
    var g = List[Float64](length=n * n, fill=0.0)
    for k in range(count):
        for i in range(n):
            for j in range(n):
                g[i * n + j] += bm[k * n + i] * bm[k * n + j]
    return g^


def fit_surface(s: Mat[DType.float64], nu: Int, nv: Int, p: Int, q: Int) raises -> Mat[DType.float64]:
    """Least-squares tensor-product B-spline: minimise |Bu P Bv^T - S|^2 over the nu x nv control net P."""
    var m = s.rows
    var n = s.cols
    if s.ch != 3 or nu <= p or nv <= q or p < 1 or q < 1 or m < nu or n < nv:
        raise Error("mojocv: nurbs fit wants an M x N x 3 grid with M >= nu > p >= 1 and N >= nv > q >= 1")
    var bu = _basis_matrix(m, nu, p)
    var bv = _basis_matrix(n, nv, q)
    var gu = _gram(bu, m, nu)
    var gv = _gram(bv, n, nv)
    var net = Mat[DType.float64](nu, nv, 4)
    for c in range(3):
        # T = Bu^T S_c (nu x N), solve Gu Q = T, then R = Q Bv (nu x nv), solve Gv P^T = R^T
        var t = List[Float64](length=nu * n, fill=0.0)
        for k in range(m):
            for i in range(nu):
                var w = bu[k * nu + i]
                if w != 0.0:
                    for j in range(n):
                        t[i * n + j] += w * s.data[(k * n + j) * 3 + c]
        var gu2 = gu.copy()
        _solve_spd(gu2, nu, t, n)
        var rt = List[Float64](length=nv * nu, fill=0.0)  # R transposed
        for i in range(nu):
            for l in range(n):
                for j in range(nv):
                    rt[j * nu + i] += t[i * n + l] * bv[l * nv + j]
        var gv2 = gv.copy()
        _solve_spd(gv2, nv, rt, nu)
        for i in range(nu):
            for j in range(nv):
                net.data[(i * nv + j) * 4 + c] = rt[j * nu + i]
    for k in range(nu * nv):
        net.data[k * 4 + 3] = 1.0
    return net^
