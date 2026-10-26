# Phase correlation (Kuglin & Hines 1975): the translation between two frames from the normalised cross-power
# spectrum. Sizes need no padding: a mixed-radix FFT handles any n (cost O(n * sum of prime factors), so sizes
# built from small primes are fast and a large prime n degrades to O(n^2)).
# phase_corr(a, b) = [dx, dy, peak] with b(x, y) = a(x - dx, y - dy): the content of b sits (dx, dy) right and down
# of a's (the cv2.phaseCorrelate convention). Integer part from the surface maximum; the fraction per axis from the
# exact Dirichlet-kernel ratio of the peak and its larger neighbour, which is exact for a periodic fractional shift
# (cv2's 5x5 centroid returned 2.175 for a true 2.5). peak is the surface maximum (1.0 for an integer shift).
from std.math import cos, sin, sqrt, atan, pi
from max.algorithm import parallelize
from mojocv.mat import Mat


def _spf(n: Int) -> Int:
    var p = 2
    while p * p <= n:
        if n % p == 0:
            return p
        p += 1
    return n


comptime FP = MutPointer[Float64, MutAnyOrigin]


def _fp(l: List[Float64]) -> FP:
    return FP(unsafe_from_address=Int(l.unsafe_ptr()))


def _rec(
    cs: FP, sn: FP, n0: Int, sign: Float64, ire: FP, iim: FP, istr: Int, ore: FP, oim: FP, n: Int, tr: FP, ti: FP
):
    """Decimation in time, recursive: n = p * m, p sub-DFTs of the strided input, then a length-p butterfly per bin."""
    if n == 1:
        ore[0] = ire[0]
        oim[0] = iim[0]
        return
    var p = _spf(n)
    var m = n // p
    for r in range(p):
        _rec(cs, sn, n0, sign, ire + r * istr, iim + r * istr, istr * p, ore + r * m, oim + r * m, m, tr, ti)
    var step = n0 // n
    if p == 2:
        for k in range(m):
            var wr = cs[k * step]
            var wi = sign * sn[k * step]
            var xr = ore[m + k]
            var xi = oim[m + k]
            var yr = xr * wr - xi * wi
            var yi = xr * wi + xi * wr
            ore[m + k] = ore[k] - yr
            oim[m + k] = oim[k] - yi
            ore[k] += yr
            oim[k] += yi
        return
    var pstep = n0 // p
    for k in range(m):
        for r in range(p):  # twiddle W_n^(r k); r * k < n so the index stays below n0
            var wr = cs[r * k * step]
            var wi = sign * sn[r * k * step]
            var xr = ore[r * m + k]
            var xi = oim[r * m + k]
            tr[r] = xr * wr - xi * wi
            ti[r] = xr * wi + xi * wr
        if p == 3:
            var sr = tr[1] + tr[2]
            var si = ti[1] + ti[2]
            var dr = 0.8660254037844386 * sign * (tr[1] - tr[2])
            var di = 0.8660254037844386 * sign * (ti[1] - ti[2])
            var ar = tr[0] - 0.5 * sr
            var ai = ti[0] - 0.5 * si
            ore[k] = tr[0] + sr
            oim[k] = ti[0] + si
            ore[m + k] = ar - di
            oim[m + k] = ai + dr
            ore[2 * m + k] = ar + di
            oim[2 * m + k] = ai - dr
            continue
        for q in range(p):  # generic length-p DFT
            var sr = 0.0
            var si = 0.0
            for r in range(p):
                var idx = ((r * q) % p) * pstep
                var wr = cs[idx]
                var wi = sign * sn[idx]
                sr += tr[r] * wr - ti[r] * wi
                si += tr[r] * wi + ti[r] * wr
            ore[q * m + k] = sr
            oim[q * m + k] = si


struct Fft(Movable):
    """Tables for one length n; run() is read-only on them, so one plan serves many threads."""
    var n: Int
    var big: Int  # largest prime factor: size of the butterfly scratch
    var cs: List[Float64]
    var sn: List[Float64]

    def __init__(out self, n: Int):
        self.n = n
        self.cs = List[Float64]()
        self.sn = List[Float64]()
        for j in range(n):
            self.cs.append(cos(2.0 * pi * Float64(j) / Float64(n)))
            self.sn.append(sin(2.0 * pi * Float64(j) / Float64(n)))
        self.big = 1
        var r = n
        while r > 1:
            var p = _spf(r)
            self.big = max(self.big, p)
            r //= p

    def run(self, ire: FP, iim: FP, istr: Int, ore: FP, oim: FP, inverse: Bool, tr: FP, ti: FP):
        """DFT of the n values at ire/iim (stride istr) into ore/oim (contiguous); the inverse is unnormalised."""
        _rec(_fp(self.cs), _fp(self.sn), self.n, 1.0 if inverse else -1.0, ire, iim, istr, ore, oim, self.n, tr, ti)


def fft2(re: FP, im: FP, rows: Int, cols: Int, inverse: Bool):
    """In-place 2D DFT of a row-major rows x cols complex array (the inverse is unnormalised), threaded."""
    var fr = Fft(cols)
    var fc = Fft(rows)

    def row_work(c: Int) {imm}:
        var per = (rows + 63) // 64
        var sre = List[Float64](length=cols, fill=0.0)
        var sim = List[Float64](length=cols, fill=0.0)
        var tr = List[Float64](length=fr.big, fill=0.0)
        var ti = List[Float64](length=fr.big, fill=0.0)
        for y in range(c * per, min(rows, (c + 1) * per)):
            for x in range(cols):
                sre[x] = re[y * cols + x]
                sim[x] = im[y * cols + x]
            fr.run(_fp(sre), _fp(sim), 1, re + y * cols, im + y * cols, inverse, _fp(tr), _fp(ti))
        _ = (sre[0], sim[0], tr[0], ti[0])

    parallelize(row_work, 64)

    def col_work(c: Int) {imm}:
        # 8 columns per sweep: each row visit reads one 64-byte cache line instead of 8 scattered ones
        var per = ((cols + 63) // 64 + 7) // 8 * 8
        var sre = List[Float64](length=8 * rows, fill=0.0)
        var sim = List[Float64](length=8 * rows, fill=0.0)
        var ore = List[Float64](length=rows, fill=0.0)
        var oim = List[Float64](length=rows, fill=0.0)
        var tr = List[Float64](length=fc.big, fill=0.0)
        var ti = List[Float64](length=fc.big, fill=0.0)
        var x0 = c * per
        while x0 < min(cols, (c + 1) * per):
            var nb = min(8, min(cols, (c + 1) * per) - x0)
            for y in range(rows):
                for b in range(nb):
                    sre[b * rows + y] = re[y * cols + x0 + b]
                    sim[b * rows + y] = im[y * cols + x0 + b]
            for b in range(nb):
                fc.run(_fp(sre) + b * rows, _fp(sim) + b * rows, 1, _fp(ore), _fp(oim), inverse, _fp(tr), _fp(ti))
                for y in range(rows):
                    re[y * cols + x0 + b] = ore[y]
                    im[y * cols + x0 + b] = oim[y]
            x0 += 8
        _ = (sre[0], sim[0], ore[0], oim[0], tr[0], ti[0])

    parallelize(col_work, 64)
    _ = (fr.n, fc.n)


def _signed(i: Int, n: Int) -> Int:
    return i - n if i > (n - 1) // 2 else i


def _fraction(rm: Float64, r0: Float64, rp: Float64, n: Int) -> Float64:
    """Signed sub-sample offset of the true peak from the integer peak r0, given its two neighbours."""
    var nb = max(rm, rp)
    if r0 <= 0.0 or nb <= 0.0:
        return 0.0
    var rho = nb / r0
    var f = Float64(n) / pi * atan(rho * sin(pi / Float64(n)) / (1.0 + rho * cos(pi / Float64(n))))
    return f if rp >= rm else -f


def phase_corr(a: Mat[DType.float64], b: Mat[DType.float64], window: Bool) raises -> Mat[DType.float64]:
    if not a.same_shape(b) or a.ch != 1 or a.rows < 2 or a.cols < 2:
        raise Error("mojocv: phase_corr wants two single-channel frames of one size, at least 2 x 2")
    var rows = a.rows
    var cols = a.cols
    var n = rows * cols
    var ar = a.data.copy()
    var br = b.data.copy()
    if window:  # separable Hann, 0.5 (1 - cos(2 pi i / (n - 1))), as cv2.createHanningWindow
        for y in range(rows):
            for x in range(cols):
                var w = 0.25 * (1.0 - cos(2.0 * pi * Float64(y) / Float64(rows - 1))) * (
                    1.0 - cos(2.0 * pi * Float64(x) / Float64(cols - 1))
                )
                ar[y * cols + x] *= w
                br[y * cols + x] *= w
    # One complex FFT carries both real frames: z = a + i b, A(k) = (Z(k) + conj Z(-k)) / 2,
    # B(k) = (Z(k) - conj Z(-k)) / 2i. The cross power b * conj(a) / |.| then goes back through one inverse FFT.
    var zr = ar^
    var zi = br^
    fft2(_fp(zr), _fp(zi), rows, cols, False)
    var cr = List[Float64](length=n, fill=0.0)
    var ci = List[Float64](length=n, fill=0.0)
    var zrp = _fp(zr)
    var zip_ = _fp(zi)
    var crp = _fp(cr)
    var cip = _fp(ci)

    def cross_work(c: Int) {imm}:
        var per = (rows + 63) // 64
        for y in range(c * per, min(rows, (c + 1) * per)):
            for x in range(cols):
                var k = y * cols + x
                var m = ((rows - y) % rows) * cols + (cols - x) % cols
                var a_r = 0.5 * (zrp[k] + zrp[m])
                var a_i = 0.5 * (zip_[k] - zip_[m])
                var b_r = 0.5 * (zip_[k] + zip_[m])
                var b_i = -0.5 * (zrp[k] - zrp[m])
                var pr = b_r * a_r + b_i * a_i
                var pi_ = b_i * a_r - b_r * a_i
                var mag = sqrt(pr * pr + pi_ * pi_) + 1e-12
                crp[k] = pr / mag
                cip[k] = pi_ / mag

    parallelize(cross_work, 64)
    fft2(_fp(cr), _fp(ci), rows, cols, True)
    ar = cr^
    var best = -1e300
    var at = 0
    for i in range(n):
        ar[i] /= Float64(n)
        if ar[i] > best:
            best = ar[i]
            at = i
    var py = at // cols
    var px = at % cols
    var rx = (ar[py * cols + (px + cols - 1) % cols], best, ar[py * cols + (px + 1) % cols])
    var ry = (ar[((py + rows - 1) % rows) * cols + px], best, ar[((py + 1) % rows) * cols + px])
    var out = Mat[DType.float64](1, 3)
    out.data[0] = Float64(_signed(px, cols)) + _fraction(rx[0], rx[1], rx[2], cols)
    out.data[1] = Float64(_signed(py, rows)) + _fraction(ry[0], ry[1], ry[2], rows)
    out.data[2] = best
    return out^
