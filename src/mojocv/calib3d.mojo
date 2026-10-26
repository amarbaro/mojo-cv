# calib3d: camera models, pose and epipolar geometry, rectification, stereo and fiducials, all on small float64
# matrices stored as Mat[float64] (rows x cols x 1). Conventions follow cv2 5.0: dist = (k1, k2, p1, p2[, k3[, k4,
# k5, k6[, s1, s2, s3, s4]]]), rvec/tvec map object to camera coordinates.
from std.math import sqrt, atan, atan2, tan, cos, sin, acos, floor
from mojocv.mat import Mat, load, save
from mojocv.core import Args, par
from mojocv.imgproc import rint, good_features, sepfilter, find_contours
from mojocv.video import corner_sub_pix

comptime F64 = DType.float64
comptime F32 = DType.float32
comptime Mx = Mat[F64]


# ---- small dense linear algebra ---------------------------------------------------------------------------


def mx(r: Int, c: Int) -> Mx:
    return Mx(r, c, 1)


def mx_from(v: List[Float64], r: Int, c: Int) -> Mx:
    var m = mx(r, c)
    for i in range(r * c):
        m.data[i] = v[i]
    return m^


def ident(n: Int) -> Mx:
    var m = mx(n, n)
    for i in range(n):
        m.set(i, i, 0, 1.0)
    return m^


def mt(a: Mx) -> Mx:
    var o = mx(a.cols, a.rows)
    for i in range(a.rows):
        for j in range(a.cols):
            o.set(j, i, 0, a.at(i, j))
    return o^


def mm(a: Mx, b: Mx) -> Mx:
    var o = mx(a.rows, b.cols)
    for i in range(a.rows):
        for k in range(a.cols):
            var v = a.at(i, k)
            for j in range(b.cols):
                o.data[i * b.cols + j] += v * b.at(k, j)
    return o^


def madd(a: Mx, b: Mx, s: Float64 = 1.0) -> Mx:
    var o = Mx(copy=a)
    for i in range(a.size()):
        o.data[i] += s * b.data[i]
    return o^


def mscale(a: Mx, s: Float64) -> Mx:
    var o = Mx(copy=a)
    for i in range(a.size()):
        o.data[i] *= s
    return o^


def fnorm(a: Mx) -> Float64:
    var s = Float64(0)
    for v in a.data:
        s += v * v
    return sqrt(s)


def solve(a_in: Mx, b_in: Mx) -> Mx:
    """Gauss-Jordan with partial pivoting; a is n x n, b is n x m."""
    var n = a_in.rows
    var a = Mx(copy=a_in)
    var b = Mx(copy=b_in)
    for c in range(n):
        var p = c
        for r in range(c + 1, n):
            if abs(a.at(r, c)) > abs(a.at(p, c)):
                p = r
        if p != c:
            for j in range(n):
                var t = a.at(c, j)
                a.set(c, j, 0, a.at(p, j))
                a.set(p, j, 0, t)
            for j in range(b.cols):
                var t = b.at(c, j)
                b.set(c, j, 0, b.at(p, j))
                b.set(p, j, 0, t)
        var d = a.at(c, c)
        if d == 0:
            continue
        for r in range(n):
            if r == c:
                continue
            var f = a.at(r, c) / d
            if f == 0:
                continue
            for j in range(n):
                a.data[r * n + j] -= f * a.at(c, j)
            for j in range(b.cols):
                b.data[r * b.cols + j] -= f * b.at(c, j)
    for r in range(n):
        var d = a.at(r, r)
        for j in range(b.cols):
            b.data[r * b.cols + j] = b.at(r, j) / d if d != 0 else 0.0
    return b^


def inv(a: Mx) -> Mx:
    return solve(a, ident(a.rows))


def eigh(a_in: Mx) -> Tuple[List[Float64], Mx]:
    """Jacobi eigen-decomposition of a symmetric matrix: ascending eigenvalues, eigenvectors in columns."""
    var n = a_in.rows
    var a = Mx(copy=a_in)
    var v = ident(n)
    for _ in range(100):
        var off = Float64(0)
        for i in range(n):
            for j in range(i + 1, n):
                off += a.at(i, j) * a.at(i, j)
        if off < 1e-300:
            break
        for p in range(n):
            for q in range(p + 1, n):
                var apq = a.at(p, q)
                if abs(apq) < 1e-300:
                    continue
                var th = (a.at(q, q) - a.at(p, p)) / (2 * apq)
                var t = 1.0 / (abs(th) + sqrt(th * th + 1)) * (1.0 if th >= 0 else -1.0)
                var c = 1.0 / sqrt(t * t + 1)
                var s = t * c
                for k in range(n):
                    var akp = a.at(k, p)
                    var akq = a.at(k, q)
                    a.set(k, p, 0, c * akp - s * akq)
                    a.set(k, q, 0, s * akp + c * akq)
                for k in range(n):
                    var apk = a.at(p, k)
                    var aqk = a.at(q, k)
                    a.set(p, k, 0, c * apk - s * aqk)
                    a.set(q, k, 0, s * apk + c * aqk)
                for k in range(n):
                    var vkp = v.at(k, p)
                    var vkq = v.at(k, q)
                    v.set(k, p, 0, c * vkp - s * vkq)
                    v.set(k, q, 0, s * vkp + c * vkq)
    var order = List[Int]()
    for i in range(n):
        order.append(i)
    for i in range(n):
        for j in range(i + 1, n):
            if a.at(order[j], order[j]) < a.at(order[i], order[i]):
                var t = order[i]
                order[i] = order[j]
                order[j] = t
    var vals = List[Float64]()
    var vecs = mx(n, n)
    for c in range(n):
        vals.append(a.at(order[c], order[c]))
        for r in range(n):
            vecs.set(r, c, 0, v.at(r, order[c]))
    return (vals^, vecs^)


def fl(*v: Float64) -> List[Float64]:
    var o = List[Float64]()
    for x in v:
        o.append(x)
    return o^


def cross(a: List[Float64], b: List[Float64]) -> List[Float64]:
    return fl(a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def det3(a: Mx) -> Float64:
    return a.at(0, 0) * (a.at(1, 1) * a.at(2, 2) - a.at(1, 2) * a.at(2, 1)) - a.at(0, 1) * (a.at(1, 0) * a.at(2, 2) - a.at(1, 2) * a.at(2, 0)) + a.at(0, 2) * (a.at(1, 0) * a.at(2, 1) - a.at(1, 1) * a.at(2, 0))


# ---- Rodrigues --------------------------------------------------------------------------------------------


def rodrigues_vec(r: List[Float64]) -> Mx:
    """Rotation vector to matrix."""
    var th = sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
    if th < 2.220446049250313e-16:
        return ident(3)
    var c = cos(th)
    var s = sin(th)
    var c1 = 1.0 - c
    var x = r[0] / th
    var y = r[1] / th
    var z = r[2] / th
    return mx_from(fl(c + c1 * x * x, c1 * x * y - s * z, c1 * x * z + s * y, c1 * y * x + s * z, c + c1 * y * y, c1 * y * z - s * x, c1 * z * x - s * y, c1 * z * y + s * x, c + c1 * z * z), 3, 3)


def rodrigues_mat(rm: Mx) -> List[Float64]:
    """Rotation matrix to vector (the nearest rotation first, through the polar factor of a Jacobi eigen-decomposition)."""
    var ev = eigh(mm(mt(rm), rm))
    var inv_sqrt = mx(3, 3)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                inv_sqrt.data[i * 3 + j] += ev[1].at(i, k) * ev[1].at(j, k) / sqrt(ev[0][k])
    var r = mm(rm, inv_sqrt)
    var rx = r.at(2, 1) - r.at(1, 2)
    var ry = r.at(0, 2) - r.at(2, 0)
    var rz = r.at(1, 0) - r.at(0, 1)
    var s = sqrt((rx * rx + ry * ry + rz * rz) * 0.25)
    var c = (r.at(0, 0) + r.at(1, 1) + r.at(2, 2) - 1) * 0.5
    c = min(max(c, -1.0), 1.0)
    var th = acos(c)
    if s < 1e-5:
        if c > 0:
            return fl(0.0, 0.0, 0.0)
        var t = (r.at(0, 0) + 1) * 0.5
        var vx = sqrt(max(t, 0.0))
        t = (r.at(1, 1) + 1) * 0.5
        var vy = sqrt(max(t, 0.0)) * (-1.0 if r.at(0, 1) < 0 else 1.0)
        t = (r.at(2, 2) + 1) * 0.5
        var vz = sqrt(max(t, 0.0)) * (-1.0 if r.at(0, 2) < 0 else 1.0)
        if abs(vx) < abs(vy) and abs(vx) < abs(vz) and (r.at(1, 2) > 0) != (vy * vz > 0):
            vz = -vz
        var nn = sqrt(vx * vx + vy * vy + vz * vz)
        th = th / nn
        return fl(vx * th, vy * th, vz * th)
    var vth = 1.0 / (2.0 * s)
    vth *= th
    return fl(rx * vth, ry * vth, rz * vth)


# ---- camera model -----------------------------------------------------------------------------------------


def dist_at(d: Mx, i: Int) -> Float64:
    return d.data[i] if i < d.size() else 0.0


def distort(x: Float64, y: Float64, d: Mx) -> Tuple[Float64, Float64]:
    """Apply the radial, tangential and thin-prism terms to normalized coordinates."""
    var k1 = dist_at(d, 0)
    var k2 = dist_at(d, 1)
    var p1 = dist_at(d, 2)
    var p2 = dist_at(d, 3)
    var k3 = dist_at(d, 4)
    var k4 = dist_at(d, 5)
    var k5 = dist_at(d, 6)
    var k6 = dist_at(d, 7)
    var r2 = x * x + y * y
    var r4 = r2 * r2
    var r6 = r4 * r2
    var cd = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
    var xd = x * cd + 2 * p1 * x * y + p2 * (r2 + 2 * x * x) + dist_at(d, 8) * r2 + dist_at(d, 9) * r4
    var yd = y * cd + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y + dist_at(d, 10) * r2 + dist_at(d, 11) * r4
    return (xd, yd)


def fisheye_distort(x: Float64, y: Float64, d: Mx) -> Tuple[Float64, Float64]:
    var r = sqrt(x * x + y * y)
    var th = atan(r)
    var t2 = th * th
    var td = th * (1 + dist_at(d, 0) * t2 + dist_at(d, 1) * t2 * t2 + dist_at(d, 2) * t2 * t2 * t2 + dist_at(d, 3) * t2 * t2 * t2 * t2)
    var s = td / r if r > 1e-8 else 1.0
    return (x * s, y * s)


def project_points(obj: Mx, rvec: Mx, tvec: Mx, k: Mx, d: Mx, fisheye: Bool) -> Mx:
    var rm = rodrigues_vec(fl(rvec.data[0], rvec.data[1], rvec.data[2]))
    var o = mx(obj.rows, 2)
    for i in range(obj.rows):
        var x = rm.at(0, 0) * obj.at(i, 0) + rm.at(0, 1) * obj.at(i, 1) + rm.at(0, 2) * obj.at(i, 2) + tvec.data[0]
        var y = rm.at(1, 0) * obj.at(i, 0) + rm.at(1, 1) * obj.at(i, 1) + rm.at(1, 2) * obj.at(i, 2) + tvec.data[1]
        var z = rm.at(2, 0) * obj.at(i, 0) + rm.at(2, 1) * obj.at(i, 1) + rm.at(2, 2) * obj.at(i, 2) + tvec.data[2]
        z = z if z != 0 else 1.0
        var xy = fisheye_distort(x / z, y / z, d) if fisheye else distort(x / z, y / z, d)
        o.set(i, 0, 0, k.at(0, 0) * xy[0] + k.at(0, 1) * xy[1] + k.at(0, 2))
        o.set(i, 1, 0, k.at(1, 1) * xy[1] + k.at(1, 2))
    return o^


def undistort_points(pts: Mx, k: Mx, d: Mx, rr: Mx, p: Mx, fisheye: Bool) -> Mx:
    """Pixel to normalized (or P-projected) coordinates. Pinhole: five fixed-point iterations like cv2's default;
    fisheye: Newton on theta (10 iterations, 1e-8)."""
    var o = mx(pts.rows, 2)
    var fx = k.at(0, 0)
    var fy = k.at(1, 1)
    for i in range(pts.rows):
        var u = pts.at(i, 0)
        var v = pts.at(i, 1)
        var y0 = (v - k.at(1, 2)) / fy
        var x0 = (u - k.at(0, 2) - k.at(0, 1) * y0) / fx
        var x = x0
        var y = y0
        if fisheye:
            var td = min(max(sqrt(x0 * x0 + y0 * y0), -1.5707963267948966), 1.5707963267948966)
            var th = td
            var scale = 0.0
            var converged = False
            if abs(td) > 1e-8:
                for _ in range(10):
                    var t2 = th * th
                    var t4 = t2 * t2
                    var t6 = t4 * t2
                    var t8 = t6 * t2
                    var fix = (th * (1 + dist_at(d, 0) * t2 + dist_at(d, 1) * t4 + dist_at(d, 2) * t6 + dist_at(d, 3) * t8) - td) / (1 + 3 * dist_at(d, 0) * t2 + 5 * dist_at(d, 1) * t4 + 7 * dist_at(d, 2) * t6 + 9 * dist_at(d, 3) * t8)
                    th -= fix
                    if abs(fix) < 1e-8:
                        converged = True
                        break
                scale = tan(th) / td
            else:
                converged = True
            if not converged or (td > 0 and th < 0):
                o.set(i, 0, 0, -1e6)
                o.set(i, 1, 0, -1e6)
                continue
            x = x0 * scale
            y = y0 * scale
        else:
            for _ in range(5):
                var r2 = x * x + y * y
                var icd = (1 + ((dist_at(d, 7) * r2 + dist_at(d, 6)) * r2 + dist_at(d, 5)) * r2) / (1 + ((dist_at(d, 4) * r2 + dist_at(d, 1)) * r2 + dist_at(d, 0)) * r2)
                if icd < 0:
                    x = x0
                    y = y0
                    break
                var dx = 2 * dist_at(d, 2) * x * y + dist_at(d, 3) * (r2 + 2 * x * x) + dist_at(d, 8) * r2 + dist_at(d, 9) * r2 * r2
                var dy = dist_at(d, 2) * (r2 + 2 * y * y) + 2 * dist_at(d, 3) * x * y + dist_at(d, 10) * r2 + dist_at(d, 11) * r2 * r2
                x = (x0 - dx) * icd
                y = (y0 - dy) * icd
        var xx = rr.at(0, 0) * x + rr.at(0, 1) * y + rr.at(0, 2)
        var yy = rr.at(1, 0) * x + rr.at(1, 1) * y + rr.at(1, 2)
        var ww = rr.at(2, 0) * x + rr.at(2, 1) * y + rr.at(2, 2)
        ww = 1.0 / ww if ww != 0 else 1.0
        x = xx * ww
        y = yy * ww
        if p.rows > 0:
            o.set(i, 0, 0, p.at(0, 0) * x + p.at(0, 1) * y + p.at(0, 2))
            o.set(i, 1, 0, p.at(1, 1) * y + p.at(1, 2))
        else:
            o.set(i, 0, 0, x)
            o.set(i, 1, 0, y)
    return o^


def init_undistort_rectify_map(k: Mx, d: Mx, rr: Mx, p: Mx, w: Int, h: Int, fisheye: Bool) -> Tuple[Mat[DType.float32], Mat[DType.float32]]:
    """cv2.initUndistortRectifyMap: for each destination pixel the source position, via R^-1 and the distortion."""
    var mapx = Mat[DType.float32](h, w, 1)
    var mapy = Mat[DType.float32](h, w, 1)
    var pn = mx(3, 3)
    for i in range(3):
        for j in range(3):
            pn.set(i, j, 0, p.at(i, j))
    var fi = inv(pn)
    var rt = inv(rr)
    var pmx = mapx.ptr()
    var pmy = mapy.ptr()

    def row(v: Int) {imm}:
        for u in range(w):
            var x = fi.at(0, 0) * Float64(u) + fi.at(0, 1) * Float64(v) + fi.at(0, 2)
            var y = fi.at(1, 0) * Float64(u) + fi.at(1, 1) * Float64(v) + fi.at(1, 2)
            var z = fi.at(2, 0) * Float64(u) + fi.at(2, 1) * Float64(v) + fi.at(2, 2)
            var X = rt.at(0, 0) * x + rt.at(0, 1) * y + rt.at(0, 2) * z
            var Y = rt.at(1, 0) * x + rt.at(1, 1) * y + rt.at(1, 2) * z
            var W = rt.at(2, 0) * x + rt.at(2, 1) * y + rt.at(2, 2) * z
            W = 1.0 / W if W != 0 else 1.0
            var xy = fisheye_distort(X * W, Y * W, d) if fisheye else distort(X * W, Y * W, d)
            pmx[unsafe_offset=v * w + u] = Float32(k.at(0, 0) * xy[0] + k.at(0, 1) * xy[1] + k.at(0, 2))
            pmy[unsafe_offset=v * w + u] = Float32(k.at(1, 1) * xy[1] + k.at(1, 2))

    par(row, h)
    _ = (fi.rows, rt.rows)
    return (mapx^, mapy^)


# ---- RANSAC helpers ---------------------------------------------------------------------------------------


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64 = 0x9E3779B97F4A7C15):
        self.s = seed

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def sample(mut self, n: Int, k: Int) -> List[Int]:
        """k distinct indices below n."""
        var out = List[Int]()
        while len(out) < k:
            var v = Int(self.next() % UInt64(n))
            var dup = False
            for u in out:
                if u == v:
                    dup = True
            if not dup:
                out.append(v)
        return out^


def ransac_iters(inliers: Int, n: Int, m: Int, conf: Float64, cap: Int) -> Int:
    """Adaptive stopping: ceil(ln(1 - conf) / ln(1 - w^m)) with w the inlier share."""
    var w = Float64(inliers) / Float64(n)
    var wm = 1.0
    for _ in range(m):
        wm *= w
    if wm < 1e-300:
        return cap
    if wm >= 1.0:
        return 1
    return min(cap, Int(floor(log_(1.0 - conf) / log_(1.0 - wm))) + 1)


def log_(x: Float64) -> Float64:
    from std.math import log
    return log(x)


# ---- estimateAffine2D -------------------------------------------------------------------------------------


def affine_fit(a: Mx, b: Mx, idx: List[Int]) -> Mx:
    """Least-squares 2 x 3 affine map taking a[idx] to b[idx] (exact for three points)."""
    var n = len(idx)
    var ata = mx(3, 3)
    var atb = mx(3, 2)
    for t in range(n):
        var i = idx[t]
        var row = fl(a.at(i, 0), a.at(i, 1), 1.0)
        for r in range(3):
            for c in range(3):
                ata.data[r * 3 + c] += row[r] * row[c]
            atb.data[r * 2] += row[r] * b.at(i, 0)
            atb.data[r * 2 + 1] += row[r] * b.at(i, 1)
    var x = solve(ata, atb)
    return mt(x)


def estimate_affine(a: Mx, b: Mx, thresh: Float64, max_iters: Int, conf: Float64) -> Tuple[Mx, Mat[DType.uint8]]:
    var n = a.rows
    var rng = Rng()
    var best = mx(2, 3)
    var best_in = -1
    var mask = Mat[DType.uint8](n, 1, 1)
    var t2 = thresh * thresh
    var iters = max_iters
    var it = 0
    while it < iters:
        it += 1
        var s = rng.sample(n, 3)
        var d = (a.at(s[1], 0) - a.at(s[0], 0)) * (a.at(s[2], 1) - a.at(s[0], 1)) - (a.at(s[2], 0) - a.at(s[0], 0)) * (a.at(s[1], 1) - a.at(s[0], 1))
        if abs(d) < 1e-9:
            continue
        var m = affine_fit(a, b, s)
        var cnt = 0
        for i in range(n):
            var ex = m.at(0, 0) * a.at(i, 0) + m.at(0, 1) * a.at(i, 1) + m.at(0, 2) - b.at(i, 0)
            var ey = m.at(1, 0) * a.at(i, 0) + m.at(1, 1) * a.at(i, 1) + m.at(1, 2) - b.at(i, 1)
            if ex * ex + ey * ey <= t2:
                cnt += 1
        if cnt > best_in:
            best_in = cnt
            best = m^
            iters = ransac_iters(cnt, n, 3, conf, max_iters)
    var inl = List[Int]()
    for i in range(n):
        var ex = best.at(0, 0) * a.at(i, 0) + best.at(0, 1) * a.at(i, 1) + best.at(0, 2) - b.at(i, 0)
        var ey = best.at(1, 0) * a.at(i, 0) + best.at(1, 1) * a.at(i, 1) + best.at(1, 2) - b.at(i, 1)
        if ex * ex + ey * ey <= t2:
            inl.append(i)
            mask.data[i] = 1
    if len(inl) >= 3:
        best = affine_fit(a, b, inl)
    return (best^, mask^)


# ---- pose from 2D-3D --------------------------------------------------------------------------------------


def polar_rotation(m: Mx) -> Mx:
    """The rotation nearest to m: m (m^T m)^(-1/2)."""
    var ev = eigh(mm(mt(m), m))
    var isq = mx(3, 3)
    for k in range(3):
        for i in range(3):
            for j in range(3):
                isq.data[i * 3 + j] += ev[1].at(i, k) * ev[1].at(j, k) / sqrt(ev[0][k])
    return mm(m, isq)


def null_vector(a: Mx) -> List[Float64]:
    """Unit right singular vector of the smallest singular value of a."""
    var ev = eigh(mm(mt(a), a))
    var v = List[Float64]()
    for i in range(a.cols):
        v.append(ev[1].at(i, 0))
    return v^


def is_planar(obj: Mx) -> Bool:
    var c = List[Float64](length=3, fill=0)
    for i in range(obj.rows):
        for k in range(3):
            c[k] += obj.at(i, k) / Float64(obj.rows)
    var cov = mx(3, 3)
    for i in range(obj.rows):
        for r in range(3):
            for q in range(3):
                cov.data[r * 3 + q] += (obj.at(i, r) - c[r]) * (obj.at(i, q) - c[q])
    var ev = eigh(cov)
    var lo = ev[0][0]
    var hi = ev[0][2]
    return lo <= 1e-9 * hi


def pose_epnp(obj: Mx, nrm: Mx, idx: List[Int]) -> Tuple[Mx, Mx]:
    """EPnP control points, distance constraints and Gauss-Newton beta refinement (Lepetit 2009)."""
    var n = len(idx)
    var center = mx(1, 3)
    var op = obj.ptr()
    var cp = center.ptr()
    for i in idx:
        for c in range(3):
            cp[c] += op[i * 3 + c] / Float64(n)
    var cov = mx(3, 3)
    for i in idx:
        for r in range(3):
            for c in range(3):
                cov.data[r * 3 + c] += (op[i * 3 + r] - cp[r]) * (op[i * 3 + c] - cp[c])
    var eig = eigh(cov)
    var smallest = eig[0][0]
    var largest = eig[0][2]
    var nc = 3 if smallest < largest * 1e-10 else 4
    var controls = mx(nc, 3)
    var alpha = mx(n, nc)
    var ap = alpha.ptr()
    for c in range(3):
        controls.data[c] = cp[c]
    for j in range(1, nc):
        var scale = sqrt(max(eig[0][3 - j] / Float64(n), 1e-30))
        for c in range(3):
            controls.data[3 * j + c] = cp[c] + eig[1].at(c, 3 - j) * scale
        for i in range(n):
            for c in range(3):
                ap[i * nc + j] += (op[idx[i] * 3 + c] - cp[c]) * eig[1].at(c, 3 - j) / scale
    var a = mx(2 * n, 3 * nc)
    var mp = a.ptr()
    var np = nrm.ptr()
    for i in range(n):
        ap[i * nc] = 1.0
        for j in range(1, nc):
            ap[i * nc] -= ap[i * nc + j]
        for j in range(nc):
            var v = ap[i * nc + j]
            mp[(2 * i) * 3 * nc + 3 * j] = v
            mp[(2 * i + 1) * 3 * nc + 3 * j + 1] = v
            mp[(2 * i) * 3 * nc + 3 * j + 2] = -v * np[idx[i] * 2]
            mp[(2 * i + 1) * 3 * nc + 3 * j + 2] = -v * np[idx[i] * 2 + 1]
    var nulls = eigh(mm(mt(a), a))
    var nb = 2 if nc == 3 else 4
    var pairs = nc * (nc - 1) // 2
    var dv = mx(pairs * 3, nb)
    var rho = mx(pairs, 1)
    var pair = 0
    for i in range(nc):
        for j in range(i + 1, nc):
            for c in range(3):
                var delta = controls.data[3 * i + c] - controls.data[3 * j + c]
                rho.data[pair] += delta * delta
                for b in range(nb):
                    dv.set(pair * 3 + c, b, 0, nulls[1].at(3 * i + c, b) - nulls[1].at(3 * j + c, b))
            pair += 1
    var best_cost = Float64(1e300)
    var best_r = ident(3)
    var best_t = mx(3, 1)
    # Solve the linearized distance equations for 1, 2, 3 null vectors,
    # plus the four-vector first-row approximation; refine all beta weights.
    for init in range(1, nb + 1):
        var cols = init * (init + 1) // 2 if init < 4 else 4
        var l = mx(pairs, cols)
        for p in range(pairs):
            var col = 0
            for b in range(init):
                for c in range(b, init):
                    if init == 4 and b != 0:
                        continue
                    for k in range(3):
                        l.data[p * cols + col] += dv.at(p * 3 + k, b) * dv.at(p * 3 + k, c) * (1.0 if b == c else 2.0)
                    col += 1
        var ls = solve(mm(mt(l), l), mm(mt(l), rho))
        var beta = mx(nb, 1)
        beta.data[0] = sqrt(abs(ls.data[0]))
        if beta.data[0] < 1e-15:
            continue
        for b in range(1, init):
            beta.data[b] = ls.data[b] / beta.data[0]
        for iteration in range(15):
            var j = mx(pairs, nb)
            var res = mx(pairs, 1)
            for p in range(pairs):
                res.data[p] = rho.data[p]
                for c in range(3):
                    var v = Float64(0)
                    for b in range(nb):
                        v += dv.at(p * 3 + c, b) * beta.data[b]
                    res.data[p] -= v * v
                    for b in range(nb):
                        j.data[p * nb + b] += 2 * v * dv.at(p * 3 + c, b)
            var delta = solve(mm(mt(j), j), mm(mt(j), res))
            for b in range(nb):
                beta.data[b] += delta.data[b]
            if fnorm(delta) < 1e-10:
                break
        var cc = mx(nc, 3)
        for i in range(nc * 3):
            for b in range(nb):
                cc.data[i] += nulls[1].at(i, b) * beta.data[b]
        var camera = mm(alpha, cc)
        var sign = 1.0 if camera.data[2] > 0 else -1.0
        camera = mscale(camera, sign)
        var mean = mx(3, 1)
        for i in range(n):
            for c in range(3):
                mean.data[c] += camera.data[i * 3 + c] / Float64(n)
        var h = mx(3, 3)
        for i in range(n):
            for r in range(3):
                for c in range(3):
                    h.data[r * 3 + c] += (camera.data[i * 3 + r] - mean.data[r]) * (op[idx[i] * 3 + c] - cp[c])
        var svd = svd3(h)
        var u = Mx(copy=svd[0])
        var v = Mx(copy=svd[2])
        var rot = mm(u, mt(v))
        if det3(rot) < 0:
            for c in range(3):
                u.data[c * 3 + 2] = -u.data[c * 3 + 2]
            rot = mm(u, mt(v))
        var t = mx(3, 1)
        for r in range(3):
            t.data[r] = mean.data[r]
            for c in range(3):
                t.data[r] -= rot.at(r, c) * cp[c]
        var cost = Float64(0)
        for i in idx:
            var xyz = fl(t.data[0], t.data[1], t.data[2])
            for r in range(3):
                for c in range(3):
                    xyz[r] += rot.at(r, c) * op[i * 3 + c]
            if xyz[2] <= 0:
                cost += 1e100
            for c in range(2):
                var e = xyz[c] / xyz[2] - np[i * 2 + c]
                cost += e * e
        if cost < best_cost:
            best_cost = cost
            best_r = rot^
            best_t = t^
    _ = obj.rows
    _ = center.rows
    _ = alpha.rows
    _ = a.rows
    _ = nrm.rows
    return (best_r^, best_t^)


def pose_residual(obj: Mx, img: Mx, k: Mx, d: Mx, rv: List[Float64], tv: List[Float64]) -> List[Float64]:
    var r = mx_from(rv, 3, 1)
    var t = mx_from(tv, 3, 1)
    var pr = project_points(obj, r, t, k, d, False)
    var out = List[Float64](length=2 * obj.rows, fill=0)
    for i in range(obj.rows):
        out[2 * i] = pr.at(i, 0) - img.at(i, 0)
        out[2 * i + 1] = pr.at(i, 1) - img.at(i, 1)
    return out^


def refine_pose(obj: Mx, img: Mx, k: Mx, d: Mx, mut rv: List[Float64], mut tv: List[Float64]):
    """Levenberg-Marquardt on the pixel reprojection error, central-difference Jacobian."""
    var lam = 1e-3
    var res = pose_residual(obj, img, k, d, rv, tv)
    var cost = Float64(0)
    for v in res:
        cost += v * v
    for _ in range(100):
        var n = len(res)
        var jac = mx(n, 6)
        var p = fl(rv[0], rv[1], rv[2], tv[0], tv[1], tv[2])
        for c in range(6):
            var h = 1e-6
            var pp = p.copy()
            var pm = p.copy()
            pp[c] += h
            pm[c] -= h
            var rp = pose_residual(obj, img, k, d, fl(pp[0], pp[1], pp[2]), fl(pp[3], pp[4], pp[5]))
            var rm_ = pose_residual(obj, img, k, d, fl(pm[0], pm[1], pm[2]), fl(pm[3], pm[4], pm[5]))
            for i in range(n):
                jac.set(i, c, 0, (rp[i] - rm_[i]) / (2 * h))
        var jt = mt(jac)
        var jtj = mm(jt, jac)
        var rvv = mx_from(res, n, 1)
        var g = mm(jt, rvv)
        var improved = False
        for _try in range(12):
            var a = Mx(copy=jtj)
            for c in range(6):
                a.data[c * 6 + c] += lam * (jtj.at(c, c) + 1e-9)
            var delta = solve(a, mscale(g, -1.0))
            var np_ = List[Float64]()
            for c in range(6):
                np_.append(p[c] + delta.data[c])
            var nres = pose_residual(obj, img, k, d, fl(np_[0], np_[1], np_[2]), fl(np_[3], np_[4], np_[5]))
            var ncost = Float64(0)
            for v in nres:
                ncost += v * v
            if ncost <= cost:
                var step = fnorm(delta)
                rv = fl(np_[0], np_[1], np_[2])
                tv = fl(np_[3], np_[4], np_[5])
                res = nres^
                var done = cost - ncost < 1e-20 * (1 + cost) or step < 1e-13
                cost = ncost
                lam = max(lam * 0.3, 1e-12)
                improved = True
                if done:
                    return
                break
            lam *= 10.0
        if not improved:
            return


def solve_pnp(obj: Mx, img: Mx, k: Mx, d: Mx) -> Tuple[List[Float64], List[Float64]]:
    var nrm = undistort_points(img, k, d, ident(3), mx(0, 0), False)
    var idx = List[Int]()
    for i in range(obj.rows):
        idx.append(i)
    var pose = pose_epnp(obj, nrm, idx)
    var rv = rodrigues_mat(pose[0])
    ref pt = pose[1]
    var tv = fl(pt.data[0], pt.data[1], pt.data[2])
    refine_pose(obj, img, k, d, rv, tv)
    return (rv^, tv^)


def solve_pnp_ransac(obj: Mx, img: Mx, k: Mx, d: Mx, thresh: Float64, max_iters: Int, conf: Float64) -> Tuple[List[Float64], List[Float64], List[Int]]:
    var n = obj.rows
    var nrm = undistort_points(img, k, d, ident(3), mx(0, 0), False)
    var planar = is_planar(obj)
    var m = 4
    var rng = Rng()
    var best_in = List[Int]()
    var iters = max_iters
    var it = 0
    var t2 = thresh * thresh
    while it < iters:
        it += 1
        var s = rng.sample(n, m)
        var sub = mx(m, 3)
        for t in range(m):
            for c in range(3):
                sub.set(t, c, 0, obj.at(s[t], c))
        var idx = List[Int]()
        for t in range(m):
            idx.append(s[t])
        if is_planar(sub) != planar:
            continue
        var pose = pose_epnp(obj, nrm, idx)
        var pr = project_points(obj, mx_from(rodrigues_mat(pose[0]), 3, 1), pose[1], k, d, False)
        var inl = List[Int]()
        for i in range(n):
            var ex = pr.at(i, 0) - img.at(i, 0)
            var ey = pr.at(i, 1) - img.at(i, 1)
            if ex * ex + ey * ey <= t2:
                inl.append(i)
        if len(inl) > len(best_in):
            best_in = inl^
            iters = ransac_iters(len(best_in), n, m, conf, max_iters)
    var so = mx(len(best_in), 3)
    var si = mx(len(best_in), 2)
    for t in range(len(best_in)):
        for c in range(3):
            so.set(t, c, 0, obj.at(best_in[t], c))
        for c in range(2):
            si.set(t, c, 0, img.at(best_in[t], c))
    var pose = solve_pnp(so, si, k, d)
    var pr = project_points(obj, mx_from(pose[0], 3, 1), mx_from(pose[1], 3, 1), k, d, False)
    var inl = List[Int]()
    for i in range(n):
        var ex = pr.at(i, 0) - img.at(i, 0)
        var ey = pr.at(i, 1) - img.at(i, 1)
        if ex * ex + ey * ey <= t2:
            inl.append(i)
    return (pose[0].copy(), pose[1].copy(), inl^)


# ---- epipolar geometry ------------------------------------------------------------------------------------


def svd3(e: Mx) -> Tuple[Mx, List[Float64], Mx]:
    """e = U diag(s) V^T for a 3 x 3 matrix, singular values descending."""
    var ev = eigh(mm(mt(e), e))
    var v = mx(3, 3)
    var s = List[Float64]()
    for c in range(3):
        s.append(sqrt(max(ev[0][2 - c], 0.0)))
        for r in range(3):
            v.set(r, c, 0, ev[1].at(r, 2 - c))
    var u = mx(3, 3)
    for c in range(2):
        for r in range(3):
            var t = Float64(0)
            for q in range(3):
                t += e.at(r, q) * v.at(q, c)
            u.set(r, c, 0, t / s[c] if s[c] > 1e-300 else 0.0)
    var u0 = fl(u.at(0, 0), u.at(1, 0), u.at(2, 0))
    var u1 = fl(u.at(0, 1), u.at(1, 1), u.at(2, 1))
    var u2 = cross(u0, u1)
    for r in range(3):
        u.set(r, 2, 0, u2[r])
    return (u^, s^, v^)


def real_eigenvalues(matrix: Mx) -> List[Float64]:
    """Real roots of a small nonsymmetric matrix via Hessenberg shifted QR."""
    var h = Mx(copy=matrix)
    var n = h.rows
    for k in range(n - 2):
        var v = mx(n, 1)
        var norm = Float64(0)
        for i in range(k + 1, n):
            v.data[i] = h.at(i, k)
            norm += v.data[i] * v.data[i]
        norm = sqrt(norm)
        if norm < 1e-30:
            continue
        v.data[k + 1] += norm if v.data[k + 1] >= 0 else -norm
        var vn = fnorm(v)
        v = mscale(v, 1.0 / vn)
        for j in range(n):
            var dot = Float64(0)
            for i in range(k + 1, n):
                dot += v.data[i] * h.at(i, j)
            for i in range(k + 1, n):
                h.data[i * n + j] -= 2 * v.data[i] * dot
        for i in range(n):
            var dot = Float64(0)
            for j in range(k + 1, n):
                dot += v.data[j] * h.at(i, j)
            for j in range(k + 1, n):
                h.data[i * n + j] -= 2 * v.data[j] * dot
    var roots = List[Float64]()
    var m = n
    var iteration = 0
    while m > 0 and iteration < 1200:
        iteration += 1
        if m == 1 or abs(h.at(m - 1, m - 2)) < 1e-12 * (1 + abs(h.at(m - 1, m - 1)) + abs(h.at(m - 2, m - 2))):
            roots.append(h.at(m - 1, m - 1))
            m -= 1
            continue
        var a = h.at(m - 2, m - 2)
        var b = h.at(m - 2, m - 1)
        var c = h.at(m - 1, m - 2)
        var d = h.at(m - 1, m - 1)
        var disc = (a - d) * (a - d) + 4 * b * c
        if m == 2 or abs(h.at(m - 2, m - 3)) < 1e-12 * (1 + abs(a) + abs(h.at(m - 3, m - 3))):
            if disc >= 0:
                roots.append((a + d + sqrt(disc)) * 0.5)
                roots.append((a + d - sqrt(disc)) * 0.5)
            m -= 2
            continue
        var shift = (a + d) * 0.5
        if disc >= 0:
            var root = sqrt(disc) * 0.5
            var v1 = shift + root
            var v2 = shift - root
            shift = v1 if abs(v1 - d) < abs(v2 - d) else v2
        if iteration % 30 == 0:
            shift = d + abs(c)
        var r = mx(m, m)
        var q = ident(m)
        for i in range(m):
            for j in range(m):
                r.set(i, j, 0, h.at(i, j) - (shift if i == j else 0.0))
        for k in range(m - 1):
            var x = r.at(k, k)
            var y = r.at(k + 1, k)
            var norm = sqrt(x * x + y * y)
            if norm < 1e-30:
                continue
            var co = x / norm
            var si = y / norm
            for j in range(k, m):
                var top = r.at(k, j)
                var bot = r.at(k + 1, j)
                r.set(k, j, 0, co * top + si * bot)
                r.set(k + 1, j, 0, -si * top + co * bot)
            for i in range(m):
                var left = q.at(i, k)
                var right = q.at(i, k + 1)
                q.set(i, k, 0, co * left + si * right)
                q.set(i, k + 1, 0, -si * left + co * right)
        var next = mm(r, q)
        for i in range(m):
            for j in range(m):
                h.set(i, j, 0, next.at(i, j) + (shift if i == j else 0.0))
    return roots^


def cubic_term(mut coefficients: Mx, row: Int, basis: Mx, ia: Int, ib: Int, ic: Int, scale: Float64, columns: List[Int]):
    var codes = SIMD[DType.int32, 4](16, 4, 1, 0)
    var bp = basis.ptr()
    var out = coefficients.ptr() + row * 20
    for a in range(4):
        for b in range(4):
            for c in range(4):
                var code = Int(codes[a] + codes[b] + codes[c])
                out[columns[code]] += scale * bp[ia * 9 + a] * bp[ib * 9 + b] * bp[ic * 9 + c]
    _ = basis.rows
    _ = coefficients.rows


def essential_5pt(x1: Mx, x2: Mx, idx: List[Int]) -> List[Mx]:
    """Five-point cubic essential constraints, eliminated to multiplication by z."""
    var a = mx(5, 9)
    for i in range(5):
        var p = idx[i]
        var u = fl(x1.at(p, 0), x1.at(p, 1), 1.0)
        var v = fl(x2.at(p, 0), x2.at(p, 1), 1.0)
        for r in range(3):
            for c in range(3):
                a.set(i, 3 * r + c, 0, v[r] * u[c])
    var eig = eigh(mm(mt(a), a))
    var basis = Mx(copy=eig[1])
    # Cubics first, then x2,xy,y2,xz,yz,z2,x,y,z,1.
    var codes: List[Int] = [48, 36, 24, 12, 33, 21, 9, 18, 6, 3, 32, 20, 8, 17, 5, 2, 16, 4, 1, 0]
    var columns = List[Int](length=64, fill=0)
    for i in range(20):
        columns[codes[i]] = i
    var eq = mx(10, 20)
    for r in range(3):
        for c in range(3):
            for k in range(3):
                for l in range(3):
                    cubic_term(eq, 3 * r + c, basis, 3 * r + k, 3 * l + k, 3 * l + c, 2.0, columns)
                    cubic_term(eq, 3 * r + c, basis, 3 * k + l, 3 * k + l, 3 * r + c, -1.0, columns)
    for c in range(3):
        cubic_term(eq, 9, basis, c, 3 + (c + 1) % 3, 6 + (c + 2) % 3, 1.0, columns)
        cubic_term(eq, 9, basis, c, 3 + (c + 2) % 3, 6 + (c + 1) % 3, -1.0, columns)
    var lhs = mx(10, 10)
    var rhs = mx(10, 10)
    for i in range(10):
        for j in range(10):
            lhs.set(i, j, 0, eq.at(i, j))
            rhs.set(i, j, 0, -eq.at(i, j + 10))
    var reduced = solve(lhs, rhs)
    var action = mx(10, 10)
    for i in range(6):
        for j in range(10):
            action.set(i, j, 0, reduced.at(i + 4, j))
    action.set(6, 3, 0, 1.0)
    action.set(7, 4, 0, 1.0)
    action.set(8, 5, 0, 1.0)
    action.set(9, 8, 0, 1.0)
    var roots = real_eigenvalues(action)
    var out = List[Mx]()
    for z in roots:
        var shifted = Mx(copy=action)
        for i in range(10):
            shifted.data[i * 10 + i] -= z
        var v = null_vector(shifted)
        if abs(v[9]) < 1e-12:
            continue
        var x = v[6] / v[9]
        var y = v[7] / v[9]
        var e = mx(3, 3)
        for i in range(9):
            e.data[i] = basis.at(i, 0) * x + basis.at(i, 1) * y + basis.at(i, 2) * z + basis.at(i, 3)
        var norm = fnorm(e)
        e = mscale(e, 1.0 / norm)
        var eet = mm(e, mt(e))
        var cubic = madd(mscale(mm(eet, e), 2.0), e, -1.0)
        if fnorm(cubic) < 1e-5 and abs(det3(e)) < 1e-5:
            out.append(e^)
    return out^


def sampson(e: Mx, x1: Mx, x2: Mx, i: Int) -> Float64:
    var p = fl(x1.at(i, 0), x1.at(i, 1), 1.0)
    var q = fl(x2.at(i, 0), x2.at(i, 1), 1.0)
    var ex = fl(0.0, 0.0, 0.0)
    var etx = fl(0.0, 0.0, 0.0)
    for r in range(3):
        for c in range(3):
            ex[r] += e.at(r, c) * p[c]
            etx[r] += e.at(c, r) * q[c]
    var num = q[0] * ex[0] + q[1] * ex[1] + q[2] * ex[2]
    var den = ex[0] * ex[0] + ex[1] * ex[1] + etx[0] * etx[0] + etx[1] * etx[1]
    return num * num / den if den > 0 else 1e30


def normalize_pixels(p: Mx, k: Mx) -> Mx:
    var o = mx(p.rows, 2)
    for i in range(p.rows):
        var y = (p.at(i, 1) - k.at(1, 2)) / k.at(1, 1)
        o.set(i, 1, 0, y)
        o.set(i, 0, 0, (p.at(i, 0) - k.at(0, 2) - k.at(0, 1) * y) / k.at(0, 0))
    return o^


def find_essential(p1: Mx, p2: Mx, k: Mx, thresh: Float64, conf: Float64) -> Tuple[Mx, Mat[DType.uint8]]:
    """RANSAC over all real five-point essential models; unit Frobenius norm, largest entry positive."""
    var n = p1.rows
    var x1 = normalize_pixels(p1, k)
    var x2 = normalize_pixels(p2, k)
    if n == 5:
        var idx: List[Int] = [0, 1, 2, 3, 4]
        var models = essential_5pt(x1, x2, idx)
        var order = List[Int]()
        for j in range(len(models)):
            ref e = models[j]
            var big = 0
            for c in range(9):
                if abs(e.data[c]) > abs(e.data[big]):
                    big = c
            e = mscale(e, (1.0 if e.data[big] > 0 else -1.0) / fnorm(e))
            order.append(j)
        def first(i: Int, j: Int) {imm} -> Bool:
            var left = models[i].data[0]
            var right = models[j].data[0]
            return left < right
        sort(Span(order), first)
        var stacked = mx(3 * len(models), 3)
        for j in range(len(models)):
            for c in range(9):
                stacked.data[j * 9 + c] = models[order[j]].data[c]
        var mask = Mat[DType.uint8](n, 1)
        for i in range(n):
            mask.data[i] = 1
        return (stacked^, mask^)
    var f = (k.at(0, 0) + k.at(1, 1)) * 0.5
    var t2 = (thresh / f) * (thresh / f)
    var rng = Rng()
    var best_e = mx(3, 3)
    var best_n = -1
    var best_cost = 1e300
    var iters = 1000
    var it = 0
    while it < iters:
        it += 1
        var models = essential_5pt(x1, x2, rng.sample(n, 5))
        for model_index in range(len(models)):
            ref e = models[model_index]
            var cnt = 0
            var cost = Float64(0)
            for i in range(n):
                var er = sampson(e, x1, x2, i)
                cost += min(er, t2)
                if er <= t2:
                    cnt += 1
            if cost < best_cost:
                best_n = cnt
                best_cost = cost
                best_e = Mx(copy=e)
                iters = ransac_iters(cnt, n, 5, conf, 1000)
    var e = best_e^
    var mask = Mat[DType.uint8](n, 1, 1)
    for i in range(n):
        if sampson(e, x1, x2, i) <= t2:
            mask.data[i] = 1
    var nrm = fnorm(e)
    var big = 0
    for i in range(9):
        if abs(e.data[i]) > abs(e.data[big]):
            big = i
    var sg = 1.0 if e.data[big] > 0 else -1.0
    return (mscale(e, sg / nrm), mask^)


def triangulate(p1: Mx, p2: Mx, x1: Float64, y1: Float64, x2: Float64, y2: Float64) -> List[Float64]:
    var a = mx(4, 4)
    for c in range(4):
        a.set(0, c, 0, x1 * p1.at(2, c) - p1.at(0, c))
        a.set(1, c, 0, y1 * p1.at(2, c) - p1.at(1, c))
        a.set(2, c, 0, x2 * p2.at(2, c) - p2.at(0, c))
        a.set(3, c, 0, y2 * p2.at(2, c) - p2.at(1, c))
    return null_vector(a)


def recover_pose(e: Mx, p1: Mx, p2: Mx, k: Mx, mask: Mat[DType.uint8], has_mask: Bool) -> Tuple[Mx, Mx, Int]:
    var f = svd3(e)
    var u = Mx(copy=f[0])
    var v = Mx(copy=f[2])
    if det3(u) < 0:
        for r in range(3):
            u.set(r, 2, 0, -u.at(r, 2))
    if det3(v) < 0:
        for r in range(3):
            v.set(r, 2, 0, -v.at(r, 2))
    var w = mx_from(fl(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0), 3, 3)
    var ra = mm(mm(u, w), mt(v))
    var rb = mm(mm(u, mt(w)), mt(v))
    var tt = fl(u.at(0, 2), u.at(1, 2), u.at(2, 2))
    var x1 = normalize_pixels(p1, k)
    var x2 = normalize_pixels(p2, k)
    var pa = mx(3, 4)
    for i in range(3):
        pa.set(i, i, 0, 1.0)
    var best = -1
    var best_r = ident(3)
    var best_t = mx(3, 1)
    for c in range(4):
        var rr = Mx(copy=ra) if c < 2 else Mx(copy=rb)
        var sg = 1.0 if c % 2 == 0 else -1.0
        var pb = mx(3, 4)
        for i in range(3):
            for j in range(3):
                pb.set(i, j, 0, rr.at(i, j))
            pb.set(i, 3, 0, sg * tt[i])
        var good = 0
        for i in range(p1.rows):
            if has_mask and mask.data[i] == 0:
                continue
            var X = triangulate(pa, pb, x1.at(i, 0), x1.at(i, 1), x2.at(i, 0), x2.at(i, 1))
            var z1 = X[2] / X[3]
            var z2 = (pb.at(2, 0) * X[0] + pb.at(2, 1) * X[1] + pb.at(2, 2) * X[2] + pb.at(2, 3) * X[3]) / X[3]
            if z1 > 0 and z2 > 0 and z1 < 50 and z2 < 50:
                good += 1
        if good > best:
            best = good
            best_r = rr^
            best_t = mx_from(fl(sg * tt[0], sg * tt[1], sg * tt[2]), 3, 1)
    return (best_r^, best_t^, best)


# ---- stereoRectify ----------------------------------------------------------------------------------------


def stereo_rectify(k1: Mx, d1: Mx, k2: Mx, d2: Mx, rm: Mx, t: Mx, w: Int, h: Int, zero_disp: Bool, alpha: Float64 = -1.0) -> List[Mx]:
    """Bouguet rectification, horizontal/vertical baselines, alpha in [-1, 1]: R1, R2, P1, P2, Q."""
    var om = rodrigues_mat(rm)
    var half = rodrigues_vec(fl(-0.5 * om[0], -0.5 * om[1], -0.5 * om[2]))
    var tv = List[Float64](length=3, fill=0)
    for i in range(3):
        for j in range(3):
            tv[i] += half.at(i, j) * t.data[j]
    var idx = 0 if abs(tv[0]) > abs(tv[1]) else 1
    var c = tv[idx]
    var nt = sqrt(tv[0] * tv[0] + tv[1] * tv[1] + tv[2] * tv[2])
    var uu = fl(0.0, 0.0, 0.0)
    uu[idx] = 1.0 if c > 0 else -1.0
    var ww = cross(tv, uu)
    var nw = sqrt(ww[0] * ww[0] + ww[1] * ww[1] + ww[2] * ww[2])
    if nw > 0:
        var sc = acos(min(abs(c) / nt, 1.0)) / nw
        ww = fl(ww[0] * sc, ww[1] * sc, ww[2] * sc)
    var wr = rodrigues_vec(ww)
    var r1 = mm(wr, mt(half))
    var r2 = mm(wr, half)
    var t2 = mm(r2, t)
    var f = (k1.at(idx ^ 1, idx ^ 1) + k2.at(idx ^ 1, idx ^ 1)) * 0.5
    var corners = mx(4, 2)
    corners.set(1, 0, 0, Float64(w - 1))
    corners.set(2, 1, 0, Float64(h - 1))
    corners.set(3, 0, 0, Float64(w - 1))
    corners.set(3, 1, 0, Float64(h - 1))
    var cx = List[Float64]()
    var cy = List[Float64]()
    for kk in range(2):
        var u = undistort_points(corners, k1 if kk == 0 else k2, d1 if kk == 0 else d2, ident(3), mx(0, 0), False)
        var rk = Mx(copy=r1) if kk == 0 else Mx(copy=r2)
        var ax = Float64(0)
        var ay = Float64(0)
        for i in range(4):
            var z = rk.at(2, 0) * u.at(i, 0) + rk.at(2, 1) * u.at(i, 1) + rk.at(2, 2)
            ax += f * (rk.at(0, 0) * u.at(i, 0) + rk.at(0, 1) * u.at(i, 1) + rk.at(0, 2)) / z * 0.25
            ay += f * (rk.at(1, 0) * u.at(i, 0) + rk.at(1, 1) * u.at(i, 1) + rk.at(1, 2)) / z * 0.25
        cx.append(Float64(w - 1) / 2 - ax)
        cy.append(Float64(h - 1) / 2 - ay)
    if zero_disp:
        cx[0] = (cx[0] + cx[1]) * 0.5
        cx[1] = cx[0]
        cy[0] = (cy[0] + cy[1]) * 0.5
        cy[1] = cy[0]
    elif idx == 0:
        cy[0] = (cy[0] + cy[1]) * 0.5
        cy[1] = cy[0]
    else:
        cx[0] = (cx[0] + cx[1]) * 0.5
        cx[1] = cx[0]
    if alpha >= 0:
        # Sample the transformed border at nine positions per side. The inner
        # rectangle excludes invalid pixels; the outer rectangle retains them all.
        var grid = mx(81, 2)
        for y in range(9):
            for x in range(9):
                grid.set(y * 9 + x, 0, 0, Float64(x * (w - 1)) / 8.0)
                grid.set(y * 9 + x, 1, 0, Float64(y * (h - 1)) / 8.0)
        var s0 = Float64(0)
        var s1 = Float64(1e300)
        for camera in range(2):
            var proj = ident(3)
            proj.set(0, 0, 0, f)
            proj.set(1, 1, 0, f)
            proj.set(0, 2, 0, cx[camera])
            proj.set(1, 2, 0, cy[camera])
            var pixels = undistort_points(grid, k1 if camera == 0 else k2, d1 if camera == 0 else d2, r1 if camera == 0 else r2, proj, False)
            var inner = fl(-1e300, -1e300, 1e300, 1e300)
            var outer = fl(1e300, 1e300, -1e300, -1e300)
            for y in range(9):
                for x in range(9):
                    var u = pixels.at(y * 9 + x, 0)
                    var v = pixels.at(y * 9 + x, 1)
                    outer[0] = min(outer[0], u)
                    outer[1] = min(outer[1], v)
                    outer[2] = max(outer[2], u)
                    outer[3] = max(outer[3], v)
                    if x == 0:
                        inner[0] = max(inner[0], u)
                    if y == 0:
                        inner[1] = max(inner[1], v)
                    if x == 8:
                        inner[2] = min(inner[2], u)
                    if y == 8:
                        inner[3] = min(inner[3], v)
            for side in range(4):
                var center = cx[camera] if side % 2 == 0 else cy[camera]
                var edge = 0.0 if side < 2 else Float64(w - 1 if side == 2 else h - 1)
                s0 = max(s0, (edge - center) / (inner[side] - center))
                s1 = min(s1, (edge - center) / (outer[side] - center))
        f *= (1.0 - min(alpha, 1.0)) * s0 + min(alpha, 1.0) * s1
    var p1 = mx(3, 4)
    p1.set(0, 0, 0, f)
    p1.set(1, 1, 0, f)
    p1.set(0, 2, 0, cx[0])
    p1.set(1, 2, 0, cy[0])
    p1.set(2, 2, 0, 1.0)
    var p2 = Mx(copy=p1)
    p2.set(0, 2, 0, cx[1])
    p2.set(1, 2, 0, cy[1])
    p2.set(idx, 3, 0, t2.data[idx] * f)
    var q = mx(4, 4)
    q.set(0, 0, 0, 1.0)
    q.set(1, 1, 0, 1.0)
    q.set(0, 3, 0, -cx[0])
    q.set(1, 3, 0, -cy[0])
    q.set(2, 3, 0, f)
    q.set(3, 2, 0, -1.0 / t2.data[idx])
    q.set(3, 3, 0, ((cx[0] - cx[1]) if idx == 0 else (cy[0] - cy[1])) / t2.data[idx])
    var out = List[Mx]()
    out.append(r1^)
    out.append(r2^)
    out.append(p1^)
    out.append(p2^)
    out.append(q^)
    return out^


# ---- StereoSGBM -------------------------------------------------------------------------------------------


def bt_cost(l: Mat[DType.uint8], r: Mat[DType.uint8], y: Int, x: Int, xr: Int) -> Int:
    """Birchfield-Tomasi dissimilarity of l(y, x) and r(y, xr), insensitive to half-pixel sampling."""
    var w = r.cols
    var lc = Int(l.at(y, x))
    var rc = Int(r.at(y, xr))
    var rm = (rc + Int(r.at(y, max(xr - 1, 0)))) // 2
    var rp = (rc + Int(r.at(y, min(xr + 1, w - 1)))) // 2
    var rmin = min(rc, min(rm, rp))
    var rmax = max(rc, max(rm, rp))
    var c1 = max(0, max(lc - rmax, rmin - lc))
    var lw = l.cols
    var lm = (lc + Int(l.at(y, max(x - 1, 0)))) // 2
    var lp = (lc + Int(l.at(y, min(x + 1, lw - 1)))) // 2
    var lmin = min(lc, min(lm, lp))
    var lmax = max(lc, max(lm, lp))
    var c2 = max(0, max(rc - lmax, lmin - rc))
    return min(c1, c2)


def stereo_sgm(l: Mat[DType.uint8], r: Mat[DType.uint8], min_d: Int, nd: Int, block: Int, p1: Int, p2: Int, lr_max: Int, uniq: Int) -> Mat[DType.int16]:
    """Semi-global matching (Hirschmuller 2008): BT pixel cost summed over a block x block window, eight path
    directions, winner-takes-all with cv2's parabola refinement, left-right and uniqueness checks. Disparities are
    returned times 16; invalid pixels hold (min_d - 1) * 16, as in cv2.StereoSGBM."""
    var h = l.rows
    var w = l.cols
    var min_x = max(min_d + nd, 0)
    var max_x = w + min(min_d, 0)
    var o = Mat[DType.int16](h, w, 1)
    var inval = Int16((min_d - 1) * 16)
    for i in range(o.size()):
        o.data[i] = inval
    if max_x <= min_x:
        return o^
    var cw = max_x - min_x
    var pc = Mat[DType.int32](h, cw, nd)
    var cp = pc.ptr()

    def cost_row(y: Int) {imm}:
        for x in range(cw):
            for d in range(nd):
                cp[unsafe_offset=(y * cw + x) * nd + d] = Int32(bt_cost(l, r, y, x + min_x, x + min_x - min_d - d))

    par(cost_row, h, 16)
    var half = block // 2
    var agg = Mat[DType.int32](h, cw, nd)
    var ap = agg.ptr()

    def win_row(y: Int) {imm}:
        for x in range(cw):
            for d in range(nd):
                var acc = Int32(0)
                for dy in range(-half, half + 1):
                    var yy = min(max(y + dy, 0), h - 1)
                    for dx in range(-half, half + 1):
                        var xx = min(max(x + dx, 0), cw - 1)
                        acc += cp[unsafe_offset=(yy * cw + xx) * nd + d]
                ap[unsafe_offset=(y * cw + x) * nd + d] = acc

    if block > 1:
        par(win_row, h, 16)
        cp = ap
    var total = Mat[DType.int32](h, cw, nd)
    var tp = total.ptr()
    var dirs_x = SIMD[DType.int32, 8](1, -1, 0, 0, 1, -1, 1, -1)
    var dirs_y = SIMD[DType.int32, 8](0, 0, 1, -1, 1, 1, -1, -1)
    var parts = List[Mat[DType.int32]]()
    for _ in range(8):
        parts.append(Mat[DType.int32](h, cw, nd))

    def path(k: Int) {imm}:
        var dx = Int(dirs_x[k])
        var dy = Int(dirs_y[k])
        var lr = parts[k].ptr()
        for yi in range(h):
            var y = yi if dy >= 0 else h - 1 - yi
            for xi in range(cw):
                var x = xi if dx >= 0 else cw - 1 - xi
                var px = x - dx
                var py = y - dy
                var base = (y * cw + x) * nd
                if px < 0 or px >= cw or py < 0 or py >= h:
                    for d in range(nd):
                        lr[unsafe_offset=base + d] = cp[unsafe_offset=base + d]
                    continue
                var pb = (py * cw + px) * nd
                var mn = lr[unsafe_offset=pb]
                for d in range(1, nd):
                    mn = min(mn, lr[unsafe_offset=pb + d])
                for d in range(nd):
                    var v = lr[unsafe_offset=pb + d]
                    if d > 0:
                        v = min(v, lr[unsafe_offset=pb + d - 1] + Int32(p1))
                    if d + 1 < nd:
                        v = min(v, lr[unsafe_offset=pb + d + 1] + Int32(p1))
                    v = min(v, mn + Int32(p2))
                    lr[unsafe_offset=base + d] = cp[unsafe_offset=base + d] + v - mn

    par(path, 8, 16)
    for k in range(8):
        var pk = parts[k].ptr()
        for i in range(h * cw * nd):
            tp[unsafe_offset=i] += pk[unsafe_offset=i]
    _ = (len(parts), pc.rows, agg.rows)
    var d2 = List[Int](length=h * cw, fill=-1)
    var m2 = List[Int](length=h * cw, fill=1 << 30)
    var dl = List[Int](length=h * cw, fill=-1)
    for y in range(h):
        for x in range(cw):
            var base = (y * cw + x) * nd
            var best = 1 << 30
            var bd = 0
            for d in range(nd):
                var c = Int(tp[unsafe_offset=base + d])
                if c < best:
                    best = c
                    bd = d
                var xr = x - d - min_d
                if xr >= 0 and xr < cw and c < m2[y * cw + xr]:
                    m2[y * cw + xr] = c
                    d2[y * cw + xr] = d
            var ok = True
            if uniq > 0:
                for d in range(nd):
                    if abs(d - bd) > 1 and Int(tp[unsafe_offset=base + d]) * 100 <= best * (100 + uniq) - 0 and Int(tp[unsafe_offset=base + d]) * 100 < best * (100 + uniq):
                        ok = False
            dl[y * cw + x] = bd if ok else -1
    for y in range(h):
        for x in range(cw):
            var bd = dl[y * cw + x]
            if bd < 0:
                continue
            if lr_max >= 0:
                var xr = x - bd - min_d
                if xr < 0 or xr >= cw or abs(d2[y * cw + xr] - bd) > lr_max:
                    continue
            var v = (bd + min_d) * 16
            if bd > 0 and bd + 1 < nd:
                var base = (y * cw + x) * nd
                var pp = Int(tp[unsafe_offset=base + bd - 1])
                var nn = Int(tp[unsafe_offset=base + bd + 1])
                var cc = Int(tp[unsafe_offset=base + bd])
                var den = max(pp + nn - 2 * cc, 1)
                v += ((pp - nn) * 16 + den) // (den * 2)
            o.set(y, x + min_x, 0, Int16(v))
    return o^


# ---- findChessboardCorners --------------------------------------------------------------------------------


def bilinear(img: Mat[DType.uint8], x: Float64, y: Float64) -> Float64:
    var xi = Int(floor(x))
    var yi = Int(floor(y))
    var fx = x - Float64(xi)
    var fy = y - Float64(yi)
    var x0 = min(max(xi, 0), img.cols - 1)
    var x1 = min(max(xi + 1, 0), img.cols - 1)
    var y0 = min(max(yi, 0), img.rows - 1)
    var y1 = min(max(yi + 1, 0), img.rows - 1)
    return (Float64(img.at(y0, x0)) * (1 - fx) + Float64(img.at(y0, x1)) * fx) * (1 - fy) + (Float64(img.at(y1, x0)) * (1 - fx) + Float64(img.at(y1, x1)) * fx) * fy


def is_saddle(img: Mat[DType.uint8], px: Float64, py: Float64, ux: Float64, uy: Float64, vx: Float64, vy: Float64) -> Bool:
    """X junction test: opposite diagonal samples agree, adjacent ones differ by a lot."""
    var a = bilinear(img, px + 0.3 * (ux + vx), py + 0.3 * (uy + vy))
    var b = bilinear(img, px + 0.3 * (ux - vx), py + 0.3 * (uy - vy))
    var c = bilinear(img, px - 0.3 * (ux + vx), py - 0.3 * (uy + vy))
    var d = bilinear(img, px - 0.3 * (ux - vx), py - 0.3 * (uy - vy))
    var same = max(abs(a - c), abs(b - d))
    var diff = min(abs(a - b), min(abs(a - d), min(abs(c - b), abs(c - d))))
    return diff > 40 and same < 0.5 * diff


def find_chessboard(img: Mat[DType.uint8], cols: Int, rows: Int) raises -> Mat[F32]:
    """Inner-corner grid of a cols x rows board: corner candidates, lattice growth from the strongest, then the
    cols x rows window of the lattice made only of X junctions. (n, 2) float32 in row-major lattice order; empty if absent."""
    var cand = good_features(img, 600, 0.02, 5.0, 3)
    var nc = cand.rows
    var none = Mat[F32](0, 2, 1)
    if nc < cols * rows:
        return none^
    var pts = corner_sub_pix(img, cand, 3, -1, 20, 0.01)
    var px = List[Float64]()
    var py = List[Float64]()
    for i in range(nc):
        px.append(Float64(pts.at(i, 0)))
        py.append(Float64(pts.at(i, 1)))
    # try seeds in order until a lattice covers the pattern
    for seed in range(min(nc, 40)):
        var near = List[Int]()
        var dist = List[Float64]()
        for j in range(nc):
            if j != seed:
                var d = sqrt((px[j] - px[seed]) ** 2 + (py[j] - py[seed]) ** 2)
                near.append(j)
                dist.append(d)
        var n1 = -1
        for q in range(len(near)):
            if n1 < 0 or dist[q] < dist[n1]:
                n1 = q
        if n1 < 0:
            continue
        var s0 = dist[n1]
        var ax = px[near[n1]] - px[seed]
        var ay = py[near[n1]] - py[seed]
        var n2 = -1
        for q in range(len(near)):
            if dist[q] > 1.6 * s0 or q == n1:
                continue
            var bx = px[near[q]] - px[seed]
            var by = py[near[q]] - py[seed]
            var cosang = abs(ax * bx + ay * by) / (s0 * dist[q])
            if cosang < 0.45 and (n2 < 0 or dist[q] < dist[n2]):
                n2 = q
        if n2 < 0:
            continue
        var lat = Dict[Int, Int]()
        var eax = List[Float64](length=nc, fill=0)
        var eay = List[Float64](length=nc, fill=0)
        var ebx = List[Float64](length=nc, fill=0)
        var eby = List[Float64](length=nc, fill=0)
        var used = List[Bool](length=nc, fill=False)
        var queue = List[Int]()
        var qi = List[Int]()
        var qj = List[Int]()
        var bx0 = px[near[n2]] - px[seed]
        var by0 = py[near[n2]] - py[seed]
        lat[500 * 1000 + 500] = seed
        used[seed] = True
        eax[seed] = ax
        eay[seed] = ay
        ebx[seed] = bx0
        eby[seed] = by0
        queue.append(seed)
        qi.append(0)
        qj.append(0)
        var head = 0
        while head < len(queue):
            var cur = queue[head]
            var ci = qi[head]
            var cj = qj[head]
            head += 1
            for dirn in range(4):
                var di = 0
                var dj = 0
                var vx = Float64(0)
                var vy = Float64(0)
                if dirn == 0:
                    di = 1
                    vx = eax[cur]
                    vy = eay[cur]
                elif dirn == 1:
                    di = -1
                    vx = -eax[cur]
                    vy = -eay[cur]
                elif dirn == 2:
                    dj = 1
                    vx = ebx[cur]
                    vy = eby[cur]
                else:
                    dj = -1
                    vx = -ebx[cur]
                    vy = -eby[cur]
                var key = (500 + ci + di) * 1000 + 500 + cj + dj
                if key in lat:
                    continue
                var tx = px[cur] + vx
                var ty = py[cur] + vy
                var tol = 0.3 * sqrt(vx * vx + vy * vy)
                var bestq = -1
                var bestd = tol
                for q in range(nc):
                    if used[q]:
                        continue
                    var d = sqrt((px[q] - tx) ** 2 + (py[q] - ty) ** 2)
                    if d < bestd:
                        bestd = d
                        bestq = q
                if bestq < 0:
                    continue
                used[bestq] = True
                lat[key] = bestq
                var nvx = px[bestq] - px[cur]
                var nvy = py[bestq] - py[cur]
                if di != 0:
                    eax[bestq] = nvx * Float64(di)
                    eay[bestq] = nvy * Float64(di)
                    ebx[bestq] = ebx[cur]
                    eby[bestq] = eby[cur]
                else:
                    ebx[bestq] = nvx * Float64(dj)
                    eby[bestq] = nvy * Float64(dj)
                    eax[bestq] = eax[cur]
                    eay[bestq] = eay[cur]
                queue.append(bestq)
                qi.append(ci + di)
                qj.append(cj + dj)
        # saddle flags on lattice points
        var saddle = Dict[Int, Bool]()
        for k in range(len(queue)):
            var c = queue[k]
            saddle[(500 + qi[k]) * 1000 + 500 + qj[k]] = is_saddle(img, px[c], py[c], eax[c] * 0.5, eay[c] * 0.5, ebx[c] * 0.5, eby[c] * 0.5)
        for orient in range(2):
            var w = cols if orient == 0 else rows
            var h = rows if orient == 0 else cols
            for i0 in range(-40, 41):
                for j0 in range(-40, 41):
                    var ok = True
                    for jj in range(h):
                        for ii in range(w):
                            var key = (500 + i0 + ii) * 1000 + 500 + j0 + jj
                            if not (key in saddle) or not saddle[key]:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        # grid[r][c]: c runs along the cols-point axis; flip so that axis points right and the rows axis points down
                        var o = Mat[F32](cols * rows, 2, 1)
                        var g = List[Int]()
                        for r in range(rows):
                            for cc in range(cols):
                                var ii = cc if orient == 0 else r
                                var jj = r if orient == 0 else cc
                                g.append(lat[(500 + i0 + ii) * 1000 + 500 + j0 + jj])
                        var vx = px[g[cols - 1]] - px[g[0]]
                        var vy = py[g[(rows - 1) * cols]] - py[g[0]]
                        var flip_c = vx < 0
                        var flip_r = vy < 0
                        var n = 0
                        for r in range(rows):
                            for cc in range(cols):
                                var q = g[(rows - 1 - r if flip_r else r) * cols + (cols - 1 - cc if flip_c else cc)]
                                o.set(n, 0, 0, Float32(px[q]))
                                o.set(n, 1, 0, Float32(py[q]))
                                n += 1
                        return o^
    return none^


# ---- ArUco 4x4_50 -----------------------------------------------------------------------------------------
# The marker words of the predefined 4x4_50 dictionary (row-major 4x4 bits, MSB first, upright orientation), read
# from cv2's rendered markers; the construction is Garrido-Jurado et al. 2014.
comptime ARUCO_4X4_50 = SIMD[DType.uint32, 64](
    46386, 3994, 13101, 39238, 21662, 31181, 40494, 50418, 65242, 53078, 63889, 4519, 3767, 10767, 9393, 9790,
    18021, 26112, 27742, 30383, 34443, 45099, 52437, 56706, 65095, 38001, 44260, 42324, 8483, 13423, 17429, 22450,
    40655, 61643, 2222, 2345, 6261, 1279, 3574, 7258, 5912, 10792, 12940, 14514, 9448, 12011, 11583, 19300,
    20526, 20499, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)


def adaptive_mean_inv(g: Mat[DType.uint8], win: Int, c: Int) -> Mat[DType.uint8]:
    """cv2.adaptiveThreshold(MEAN_C, BINARY_INV): 255 where the pixel is at most the window mean minus c."""
    var box = List[Float32](length=win, fill=Float32(1.0 / Float64(win)))
    var mean = sepfilter[DType.uint8, DType.uint8](g, box, box, 1)
    var o = Mat[DType.uint8](g.rows, g.cols, 1)
    for i in range(g.size()):
        o.data[i] = 255 if Int(g.data[i]) <= Int(mean.data[i]) - c else 0
    return o^


def dp_split(px: List[Int], py: List[Int], a: Int, b: Int, eps: Float64, mut keep: List[Bool]):
    """Douglas-Peucker on the open run a..b of a point list (indices modulo n)."""
    var n = len(px)
    var ia = a % n
    var ib = b % n
    var dx = Float64(px[ib] - px[ia])
    var dy = Float64(py[ib] - py[ia])
    var best = -1.0
    var bi = -1
    var k = a + 1
    while k < b:
        var i = k % n
        var d = abs(Float64(px[i] - px[ia]) * dy - Float64(py[i] - py[ia]) * dx)
        d = d / sqrt(dx * dx + dy * dy) if (dx != 0 or dy != 0) else sqrt(Float64((px[i] - px[ia]) ** 2 + (py[i] - py[ia]) ** 2))
        if d > best:
            best = d
            bi = k
        k += 1
    if bi >= 0 and best > eps:
        keep[bi % n] = True
        dp_split(px, py, a, bi, eps, keep)
        dp_split(px, py, bi, b, eps, keep)


def approx_poly(px: List[Int], py: List[Int], eps: Float64) -> List[Int]:
    """Indices of a closed contour kept by Douglas-Peucker, starting from the point farthest from point 0 and its opposite."""
    var n = len(px)
    var far1 = 0
    var bd = -1
    for i in range(n):
        var d = (px[i] - px[0]) ** 2 + (py[i] - py[0]) ** 2
        if d > bd:
            bd = d
            far1 = i
    var far2 = far1
    bd = -1
    for i in range(n):
        var d = (px[i] - px[far1]) ** 2 + (py[i] - py[far1]) ** 2
        if d > bd:
            bd = d
            far2 = i
    var keep = List[Bool](length=n, fill=False)
    keep[far1] = True
    keep[far2] = True
    var lo = min(far1, far2)
    var hi = max(far1, far2)
    dp_split(px, py, lo, hi, eps, keep)
    dp_split(px, py, hi, lo + n, eps, keep)
    var out = List[Int]()
    for i in range(n):
        if keep[i]:
            out.append(i)
    return out^


def homography4(src: List[Float64], dst: List[Float64]) -> Mx:
    """3 x 3 homography mapping 4 points src (x0, y0, x1, ...) to dst."""
    var a = mx(8, 8)
    var b = mx(8, 1)
    for i in range(4):
        var x = src[2 * i]
        var y = src[2 * i + 1]
        var u = dst[2 * i]
        var v = dst[2 * i + 1]
        var r0 = fl(x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y)
        var r1 = fl(0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y)
        for c in range(8):
            a.set(2 * i, c, 0, r0[c])
            a.set(2 * i + 1, c, 0, r1[c])
        b.set(2 * i, 0, 0, u)
        b.set(2 * i + 1, 0, 0, v)
    var h = solve(a, b)
    return mx_from(fl(h.data[0], h.data[1], h.data[2], h.data[3], h.data[4], h.data[5], h.data[6], h.data[7], 1.0), 3, 3)


def decode_marker(g: Mat[DType.uint8], corners: List[Float64]) -> Int:
    """Marker id (0..49) and, packed above it, the rotation: id + 100 * rot; -1 when the quad is no 4x4_50 marker."""
    var hm = homography4(fl(0.0, 0.0, 6.0, 0.0, 6.0, 6.0, 0.0, 6.0), corners)
    var samples = List[Float64](length=24 * 24, fill=0)
    var hist = List[Int](length=256, fill=0)
    for v in range(24):
        for u in range(24):
            var x = (Float64(u) + 0.5) / 4.0
            var y = (Float64(v) + 0.5) / 4.0
            var w = hm.at(2, 0) * x + hm.at(2, 1) * y + hm.at(2, 2)
            var px = (hm.at(0, 0) * x + hm.at(0, 1) * y + hm.at(0, 2)) / w
            var py = (hm.at(1, 0) * x + hm.at(1, 1) * y + hm.at(1, 2)) / w
            var val = bilinear(g, px, py)
            samples[v * 24 + u] = val
            hist[min(max(Int(val + 0.5), 0), 255)] += 1
    # Otsu on the sampled values
    var total = 576
    var sum_all = Float64(0)
    for i in range(256):
        sum_all += Float64(i * hist[i])
    var wb = 0
    var sb = Float64(0)
    var best = -1.0
    var thr = 127
    for t in range(256):
        wb += hist[t]
        if wb == 0:
            continue
        var wf = total - wb
        if wf == 0:
            break
        sb += Float64(t * hist[t])
        var mb = sb / Float64(wb)
        var mf = (sum_all - sb) / Float64(wf)
        var between = Float64(wb) * Float64(wf) * (mb - mf) * (mb - mf)
        if between > best:
            best = between
            thr = t
    var bits = List[Int](length=36, fill=0)
    for cy in range(6):
        for cx in range(6):
            var cnt = 0
            for v in range(4):
                for u in range(4):
                    if samples[(cy * 4 + v) * 24 + cx * 4 + u] > Float64(thr):
                        cnt += 1
            bits[cy * 6 + cx] = 1 if cnt > 8 else 0
    var border = 0
    for i in range(6):
        border += bits[i] + bits[30 + i] + bits[i * 6] + bits[i * 6 + 5]
    if border > 5:
        return -1
    for rot in range(4):
        var word = 0
        for r in range(4):
            for c in range(4):
                var rr = r
                var cc = c
                for _ in range(rot):
                    var t = rr
                    rr = 3 - cc
                    cc = t
                word = (word << 1) | bits[(rr + 1) * 6 + cc + 1]
        for id in range(50):
            if Int(ARUCO_4X4_50[id]) == word:
                return id + 100 * rot
    return -1


def aruco_detect(img: Mat[DType.uint8]) -> Tuple[Mat[DType.int32], Mat[F32]]:
    """ArUco 4x4_50 markers: multi-window adaptive threshold, contours, 4-gon test, bit decoding. Returns ids (sorted)
    and 4 integer-valued corners per marker, clockwise starting at the marker's own top-left corner."""
    var ids = List[Int]()
    var peris = List[Float64]()
    var cs = List[Float64]()
    var maxdim = max(img.rows, img.cols)
    for wi in range(3):
        var win = 3 + 10 * wi
        var bw = adaptive_mean_inv(img, win, 7)
        var cn = find_contours(bw, 1, False)
        var pts = Mat[DType.int32](copy=cn[0])
        var off = 0
        for ci in range(cn[1].rows):
            var len_ = Int(cn[1].at(ci, 0))
            var start = off
            off += len_
            if len_ < Int(0.03 * Float64(maxdim)) or len_ > 4 * maxdim:
                continue
            var px = List[Int]()
            var py = List[Int]()
            for k in range(len_):
                px.append(Int(pts.at(start + k, 0)))
                py.append(Int(pts.at(start + k, 1)))
            var keep = approx_poly(px, py, Float64(len_) * 0.03)
            if len(keep) != 4:
                continue
            var qx = List[Float64]()
            var qy = List[Float64]()
            for k in keep:
                qx.append(Float64(px[k]))
                qy.append(Float64(py[k]))
            # orientation and convexity
            var sg = 0
            var convex = True
            for k in range(4):
                var x1 = qx[(k + 1) % 4] - qx[k]
                var y1 = qy[(k + 1) % 4] - qy[k]
                var x2 = qx[(k + 2) % 4] - qx[(k + 1) % 4]
                var y2 = qy[(k + 2) % 4] - qy[(k + 1) % 4]
                var cr = x1 * y2 - y1 * x2
                var sgn = 1 if cr > 0 else -1
                if k == 0:
                    sg = sgn
                elif sgn != sg:
                    convex = False
            if not convex:
                continue
            if sg < 0:
                var t = qx[1]
                qx[1] = qx[3]
                qx[3] = t
                t = qy[1]
                qy[1] = qy[3]
                qy[3] = t
            # canonicalize starting vertex: topmost (min y, then min x), preserving cyclic order
            var si = 0
            for k in range(1, 4):
                if qy[k] < qy[si] or (qy[k] == qy[si] and qx[k] < qx[si]):
                    si = k
            if si != 0:
                var nqx = List[Float64]()
                var nqy = List[Float64]()
                for k in range(4):
                    nqx.append(qx[(si + k) % 4])
                    nqy.append(qy[(si + k) % 4])
                qx = nqx^
                qy = nqy^
            var peri = Float64(0)
            var mind = 1e30
            for k in range(4):
                var d = sqrt((qx[(k + 1) % 4] - qx[k]) ** 2 + (qy[(k + 1) % 4] - qy[k]) ** 2)
                peri += d
                mind = min(mind, d)
            if mind < 0.05 * peri:
                continue
            var inside = True
            for k in range(4):
                if qx[k] < 3 or qy[k] < 3 or qx[k] >= Float64(img.cols - 3) or qy[k] >= Float64(img.rows - 3):
                    inside = False
            if not inside:
                continue
            var code = decode_marker(img, fl(qx[0], qy[0], qx[1], qy[1], qx[2], qy[2], qx[3], qy[3]))
            if code < 0:
                continue
            var rot = code // 100
            var sh = 0
            if rot % 2 == 1:
                sh = 2
            ids.append(code % 100)
            peris.append(peri)
            for k in range(4):
                cs.append(qx[(k + rot + sh) % 4])
                cs.append(qy[(k + rot + sh) % 4])
    # the same marker shows up at several window sizes and as its inner border: keep the largest of each cluster
    var byp = List[Int]()
    for i in range(len(ids)):
        byp.append(i)
    for i in range(len(byp)):
        for j in range(i + 1, len(byp)):
            if peris[byp[j]] > peris[byp[i]]:
                var t = byp[i]
                byp[i] = byp[j]
                byp[j] = t
    var kept = List[Int]()
    for m in byp:
        var dup = False
        for q in kept:
            for rr in range(4):
                var dsum = Float64(0)
                for k in range(4):
                    var kq = (k + rr) % 4
                    dsum += sqrt((cs[8 * m + 2 * k] - cs[8 * q + 2 * kq]) ** 2 + (cs[8 * m + 2 * k + 1] - cs[8 * q + 2 * kq + 1]) ** 2)
                if dsum / 4.0 < 0.1 * peris[q]:
                    dup = True
        if not dup:
            kept.append(m)
    var nids = List[Int]()
    var ncs = List[Float64]()
    for m in kept:
        nids.append(ids[m])
        for k in range(8):
            ncs.append(cs[8 * m + k])
    ids = nids^
    cs = ncs^
    var order = List[Int]()
    for i in range(len(ids)):
        order.append(i)
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if ids[order[j]] < ids[order[i]]:
                var t = order[i]
                order[i] = order[j]
                order[j] = t
    var oi = Mat[DType.int32](len(ids), 1, 1)
    var oc = Mat[F32](4 * len(ids), 2, 1)
    for r in range(len(order)):
        var m = order[r]
        oi.data[r] = Int32(ids[m])
        for k in range(8):
            oc.data[8 * r + k] = Float32(cs[8 * m + k])
    return (oi^, oc^)


def run(mut a: Args) raises -> Bool:
    if a.op == "Rodrigues":
        var m = load[F64](a.ins[0])
        var o = mx(0, 0)
        while a.again():
            if m.size() == 3:
                o = rodrigues_vec(fl(m.data[0], m.data[1], m.data[2]))
            else:
                o = mx_from(rodrigues_vec_from(m), 3, 1)
        save(o, a.out)
    elif a.op == "projectPoints" or a.op == "fisheye_projectPoints":
        var obj = load[F64](a.ins[0])
        var o = mx(0, 0)
        var rv = load[F64](a.ins[1])
        var tv = load[F64](a.ins[2])
        var k = load[F64](a.ins[3])
        var d = load[F64](a.ins[4])
        while a.again():
            o = project_points(obj, rv, tv, k, d, a.op == "fisheye_projectPoints")
        save(o, a.out)
    elif a.op == "undistortPoints" or a.op == "fisheye_undistortPoints":
        var pts = load[F64](a.ins[0])
        var k = load[F64](a.ins[1])
        var d = load[F64](a.ins[2])
        var rr = load[F64](a.ins[3]) if len(a.ins) > 3 else ident(3)
        var p = load[F64](a.ins[4]) if len(a.ins) > 4 else mx(0, 0)
        var o = mx(0, 0)
        while a.again():
            o = undistort_points(pts, k, d, rr, p, a.op == "fisheye_undistortPoints")
        save(o, a.out)
    elif a.op == "initUndistortRectifyMap" or a.op == "fisheye_initUndistortRectifyMap":
        var k = load[F64](a.ins[0])
        var d = load[F64](a.ins[1])
        var rr = load[F64](a.ins[2])
        var p = load[F64](a.ins[3])
        var o = (Mat[DType.float32](0, 0), Mat[DType.float32](0, 0))
        while a.again():
            o = init_undistort_rectify_map(k, d, rr, p, a.i("w", 640), a.i("h", 480), a.op == "fisheye_initUndistortRectifyMap")
        save(o[0], a.out)
        save(o[1], a.out + ".1")
    elif a.op == "aruco_detect":
        var im = load[DType.uint8](a.ins[0])
        var o = (Mat[DType.int32](0, 0), Mat[F32](0, 0))
        while a.again():
            o = aruco_detect(im)
        save(o[0], a.out)
        save(o[1], a.out + ".1")
    elif a.op == "findChessboardCorners":
        var im = load[DType.uint8](a.ins[0])
        var o = Mat[F32](0, 0)
        while a.again():
            o = find_chessboard(im, a.i("cols", 9), a.i("rows", 6))
        save(o, a.out)
    elif a.op == "StereoSGBM":
        var l = load[DType.uint8](a.ins[0])
        var r = load[DType.uint8](a.ins[1])
        var o = Mat[DType.int16](0, 0)
        var b = a.i("block", 5)
        while a.again():
            o = stereo_sgm(l, r, a.i("minDisp", 0), a.i("numDisp", 32), b, a.i("P1", 8 * b * b), a.i("P2", 32 * b * b), a.i("lrMax", 0), a.i("uniq", 0))
        save(o, a.out)
    elif a.op == "estimateAffine2D":
        var f = load[F64](a.ins[0])
        var t = load[F64](a.ins[1])
        var o = (mx(0, 0), Mat[DType.uint8](0, 0))
        while a.again():
            o = estimate_affine(f, t, a.f("thresh", 3.0), a.i("maxIters", 2000), a.f("conf", 0.99))
        save(o[0], a.out)
        save(o[1], a.out + ".1")
    elif a.op == "solvePnP":
        var obj = load[F64](a.ins[0])
        var img = load[F64](a.ins[1])
        var k = load[F64](a.ins[2])
        var d = load[F64](a.ins[3])
        var o = (List[Float64](), List[Float64]())
        while a.again():
            o = solve_pnp(obj, img, k, d)
        save(mx_from(o[0], 3, 1), a.out)
        save(mx_from(o[1], 3, 1), a.out + ".1")
    elif a.op == "solvePnPRansac":
        var obj = load[F64](a.ins[0])
        var img = load[F64](a.ins[1])
        var k = load[F64](a.ins[2])
        var d = load[F64](a.ins[3])
        var o = (List[Float64](), List[Float64](), List[Int]())
        while a.again():
            o = solve_pnp_ransac(obj, img, k, d, a.f("thresh", 8.0), a.i("iters", 100), a.f("conf", 0.99))
        save(mx_from(o[0], 3, 1), a.out)
        save(mx_from(o[1], 3, 1), a.out + ".1")
        var inl = Mat[DType.int32](len(o[2]), 1, 1)
        for i in range(len(o[2])):
            inl.data[i] = Int32(o[2][i])
        save(inl, a.out + ".2")
    elif a.op == "findEssentialMat":
        var p1 = load[F64](a.ins[0])
        var p2 = load[F64](a.ins[1])
        var k = load[F64](a.ins[2])
        var o = (mx(0, 0), Mat[DType.uint8](0, 0))
        while a.again():
            o = find_essential(p1, p2, k, a.f("thresh", 1.0), a.f("prob", 0.999))
        save(o[0], a.out)
        save(o[1], a.out + ".1")
    elif a.op == "recoverPose":
        var e = load[F64](a.ins[0])
        var p1 = load[F64](a.ins[1])
        var p2 = load[F64](a.ins[2])
        var k = load[F64](a.ins[3])
        var mk = load[DType.uint8](a.ins[4]) if len(a.ins) > 4 else Mat[DType.uint8](0, 0)
        var o = (mx(0, 0), mx(0, 0), 0)
        while a.again():
            o = recover_pose(e, p1, p2, k, mk, len(a.ins) > 4)
        save(o[0], a.out)
        save(o[1], a.out + ".1")
        var cnt = Mat[DType.int32](1, 1, 1)
        cnt.data[0] = Int32(o[2])
        save(cnt, a.out + ".2")
    elif a.op == "stereoRectify":
        var k1 = load[F64](a.ins[0])
        var d1 = load[F64](a.ins[1])
        var k2 = load[F64](a.ins[2])
        var d2 = load[F64](a.ins[3])
        var rm = load[F64](a.ins[4])
        var t = load[F64](a.ins[5])
        var o = List[Mx]()
        while a.again():
            o = stereo_rectify(k1, d1, k2, d2, rm, t, a.i("w", 640), a.i("h", 480), a.i("zero", 1) == 1, a.f("alpha", -1.0))
        for i in range(5):
            save(o[i], a.out if i == 0 else a.out + "." + String(i))
    else:
        return False
    return True


def rodrigues_vec_from(m: Mx) -> List[Float64]:
    return rodrigues_mat(m)
