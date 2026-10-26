# imgproc: filters, colour, geometry and segmentation ops with cv2 5.0 semantics. Border-reading ops work on a
# padded copy of the source, so every inner loop is a plain contiguous SIMD loop over the flattened row
# (pixel i of channel c sits at i * ch + c, so a horizontal tap at dx is a load at offset dx * ch).
from std.sys.info import simd_width_of
from std.utils import IndexList
from std.sys.intrinsics import gather
from std.math import sqrt, exp, floor, ceil, cos, sin, atan2, cbrt
from mojocv.mat import Mat, load, save
from mojocv.core import Args, DTS, peek_code, par, par_dyn
from std.math import log as log_

comptime Ptr = MutPointer[_, MutAnyOrigin]
comptime F32 = DType.float32
comptime WF = simd_width_of[DType.float32]()


def rint(x: Float64) -> Int:
    """Round half to even, like cvRound."""
    return Int(x.__round__())


def ptr_of[T: Copyable & Movable](l: List[T]) -> Ptr[T]:
    return Ptr[T](unsafe_from_address=Int(l.unsafe_ptr()))


def bidx(i: Int, n: Int, mode: Int) -> Int:
    """Map i into [0, n): mode 0 reflect101 (cv2 default), 1 replicate, 2 constant (-1 outside)."""
    if i >= 0 and i < n:
        return i
    if mode == 1:
        return 0 if i < 0 else n - 1
    if mode == 2:
        return -1
    if n == 1:
        return 0
    var p = 2 * (n - 1)
    var j = i % p
    return p - j if j >= n else j


def pad[dt: DType](a: Mat[dt], py: Int, px: Int, mode: Int = 0, val: Scalar[dt] = 0) -> Mat[dt]:
    """a with py rows and px columns added on every side."""
    var o = Mat[dt](a.rows + 2 * py, a.cols + 2 * px, a.ch)
    var ch = a.ch
    var pa = a.ptr()
    var po = o.ptr()
    var ostride = o.cols * ch
    var astride = a.cols * ch

    def row(r: Int) {imm}:
        var sr = bidx(r - py, a.rows, mode)
        var dst = po + r * ostride
        if sr < 0:
            for i in range(ostride):
                dst[unsafe_offset=i] = val
            return
        var src = pa + sr * astride
        for i in range(astride):
            dst[unsafe_offset=px * ch + i] = src[unsafe_offset=i]
        for c in range(px):
            var lc = bidx(c - px, a.cols, mode)
            var rc = bidx(a.cols + c, a.cols, mode)
            for k in range(ch):
                dst[unsafe_offset=c * ch + k] = val if lc < 0 else src[unsafe_offset=lc * ch + k]
                dst[unsafe_offset=(px + a.cols + c) * ch + k] = val if rc < 0 else src[unsafe_offset=rc * ch + k]

    par(row, o.rows)
    return o^


def _ld[sd: DType, w: Int](p: Ptr[Scalar[sd]], at: Int) -> SIMD[F32, w]:
    return p.unsafe_load[width=w](at).cast[F32]()


def _fin[od: DType, w: Int](v: SIMD[F32, w]) -> SIMD[od, w]:
    comptime if od.is_integral():
        return v.__round__().clamp(Scalar[F32](Scalar[od].MIN), Scalar[F32](Scalar[od].MAX)).cast[od]()
    else:
        return v.cast[od]()


def _tap[sd: DType, w: Int](p: Ptr[Scalar[sd]], k: Ptr[Float32], nk: Int, at: Int, step: Int) -> SIMD[F32, w]:
    """sum_t k[t] * p[at + t * step]"""
    var s = SIMD[F32, w](0)
    for t in range(nk):
        s = s + _ld[sd, w](p, at + t * step) * k[t]
    return s


def _hrow_f[sd: DType](s: Ptr[Scalar[sd]], d: Ptr[Float32], k: Ptr[Float32], nk: Int, cols: Int, ch: Int, mode: Int):
    """d[i] = sum_t k[t] * s[i + (t - nk // 2) * ch], one row; SIMD interior, scalar edges mapped through bidx."""
    var n = cols * ch
    var ex = nk // 2
    var e = ex * ch
    var i = e
    while i + WF <= n - e:
        d.unsafe_store(i, _tap[sd, WF](s, k, nk, i - e, ch))
        i += WF
    for side in range(2):
        for ii in range(0 if side == 0 else max(i, e), min(e, n) if side == 0 else n):
            var acc = Float32(0)
            for t in range(nk):
                var px = bidx(ii // ch + t - ex, cols, mode)
                if px >= 0:
                    acc += k[t] * Float32(s[unsafe_offset=px * ch + ii % ch])
            d[unsafe_offset=ii] = acc


def sepfilter[sd: DType, od: DType](src: Mat[sd], kx: List[Float32], ky: List[Float32], mode: Int = 0) -> Mat[od]:
    """Correlate with the column kernel ky and row kernel kx. Saturates and rounds like cv2 when od is integral.
    Each worker filters bands of 8 rows: the row pass of the rows a band needs lands in a small scratch that stays
    in cache, then the column pass reads it; the source is read in place, edges go through bidx."""
    var nx = len(kx)
    var ny = len(ky)
    var o = Mat[od](src.rows, src.cols, src.ch, False)
    var n = src.cols * src.ch
    var rows = src.rows
    var ps = src.ptr()
    var po = o.ptr()
    var pkx = ptr_of(kx)
    var pky = ptr_of(ky)
    comptime B = 32
    comptime NT = 8

    def task(t: Int) {imm}:
        var scratch = List[Float32](unsafe_uninit_length=min(rows, B + ny - 1) * n)
        var sc = ptr_of(scratch)
        var offs = List[Int](length=ny, fill=0)
        var po_ = ptr_of(offs)
        var y = rows * t // NT
        var y_end = rows * (t + 1) // NT
        while y < y_end:
            var ye = min(y + B, y_end)
            var lo = rows
            var hi = -1
            for yy in range(y, ye):
                for q in range(ny):
                    var r = bidx(yy + q - ny // 2, rows, mode)
                    if r >= 0:
                        lo = min(lo, r)
                        hi = max(hi, r)
            for r in range(lo, hi + 1):
                _hrow_f[sd](ps + r * src.cols * src.ch, sc + (r - lo) * n, pkx, nx, src.cols, src.ch, mode)
            for yy in range(y, ye):
                for q in range(ny):
                    var r = bidx(yy + q - ny // 2, rows, mode)
                    po_[unsafe_offset=q] = (r - lo) * n if r >= 0 else -1
                var i = 0
                while i + WF <= n:
                    var acc = SIMD[F32, WF](0)
                    for q in range(ny):
                        if po_[unsafe_offset=q] >= 0:
                            acc += sc.unsafe_load[width=WF](po_[unsafe_offset=q] + i) * pky[q]
                    po.unsafe_store(yy * n + i, _fin[od, WF](acc))
                    i += WF
                for j in range(i, n):
                    var acc = Float32(0)
                    for q in range(ny):
                        if po_[unsafe_offset=q] >= 0:
                            acc += sc[unsafe_offset=po_[unsafe_offset=q] + j] * pky[q]
                    po.unsafe_store(yy * n + j, _fin[od, 1](SIMD[F32, 1](acc)))
            y = ye
        _ = (len(scratch), len(offs))

    par(task, NT)
    _ = (nx, ny)
    return o^


def gauss_kernel(n: Int, sigma_in: Float64) -> List[Float32]:
    """cv2.getGaussianKernel (the fixed small-kernel tables when sigma <= 0 and n <= 7)."""
    var k = List[Float32](length=n, fill=0)
    if sigma_in <= 0 and n <= 7 and n % 2 == 1:
        var t: List[Float32]
        if n == 1:
            t = [1.0]
        elif n == 3:
            t = [0.25, 0.5, 0.25]
        elif n == 5:
            t = [0.0625, 0.25, 0.375, 0.25, 0.0625]
        else:
            t = [0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125]
        return t^
    var sigma = sigma_in if sigma_in > 0 else 0.3 * (Float64(n - 1) * 0.5 - 1) + 0.8
    var s = Float64(0)
    var w = List[Float64](length=n, fill=0)
    for i in range(n):
        var x = Float64(i) - Float64(n - 1) * 0.5
        w[i] = exp(-x * x / (2 * sigma * sigma))
        s += w[i]
    for i in range(n):
        k[i] = Float32(w[i] / s)
    return k^


def deriv_kernel(n: Int, order: Int) -> List[Float32]:
    """cv2.getDerivKernels row: (n - 1 - order) smoothing steps [1 1], then `order` difference steps [-1 1]."""
    var k = List[Float32]()
    k.append(1)
    for step in range(n - 1):
        var nk = List[Float32](length=len(k) + 1, fill=0)
        var a: Float32 = -1 if step >= n - 1 - order else 1
        for i in range(len(k)):
            nk[i] += a * k[i]
            nk[i + 1] += k[i]
        k = nk^
    return k^


def _binom5_u8[dt: DType](src: Mat[dt]) -> Mat[dt]:
    """GaussianBlur 5x5 with the 1-4-6-4-1 kernel on uint8 in integers like cv2 (reflect101): per output row the column
    pass sums five source rows in uint16 into a padded scratch row (4080 at most), the row pass sums five shifted
    loads (65280 at most, no overflow) and (v + 128) >> 8 rounds half up."""
    comptime WU = simd_width_of[DType.uint16]()
    comptime U16 = DType.uint16
    var o = Mat[dt](src.rows, src.cols, src.ch, False)
    var n = src.cols * src.ch
    var ch = src.ch
    var rows = src.rows
    var cols = src.cols
    var ps = src.ptr()
    var po = o.ptr()
    comptime NT = 8

    def task(t: Int) {imm}:
        var scratch = List[UInt16](unsafe_uninit_length=n + 4 * ch + WU)
        var sc = ptr_of(scratch) + 2 * ch
        for y in range(rows * t // NT, rows * (t + 1) // NT):
            var r0 = ps + bidx(y - 2, rows, 0) * n
            var r1 = ps + bidx(y - 1, rows, 0) * n
            var r2 = ps + y * n
            var r3 = ps + bidx(y + 1, rows, 0) * n
            var r4 = ps + bidx(y + 2, rows, 0) * n
            var i = 0
            while i + WU <= n:
                var v = r0.unsafe_load[width=WU](i).cast[U16]() + r4.unsafe_load[width=WU](i).cast[U16]() + 4 * (r1.unsafe_load[width=WU](i).cast[U16]() + r3.unsafe_load[width=WU](i).cast[U16]()) + 6 * r2.unsafe_load[width=WU](i).cast[U16]()
                sc.unsafe_store(i, v)
                i += WU
            for j in range(i, n):
                sc[unsafe_offset=j] = UInt16(r0[unsafe_offset=j]) + UInt16(r4[unsafe_offset=j]) + 4 * (UInt16(r1[unsafe_offset=j]) + UInt16(r3[unsafe_offset=j])) + 6 * UInt16(r2[unsafe_offset=j])
            for k in range(ch):
                sc[unsafe_offset=k - ch] = sc[unsafe_offset=ch + k]
                sc[unsafe_offset=k - 2 * ch] = sc[unsafe_offset=2 * ch + k]
                sc[unsafe_offset=n + k] = sc[unsafe_offset=n - 2 * ch + k]
                sc[unsafe_offset=n + ch + k] = sc[unsafe_offset=n - 3 * ch + k]
            var dst = po + y * n
            i = 0
            while i + WU <= n:
                var v = sc.unsafe_load[width=WU](i - 2 * ch) + sc.unsafe_load[width=WU](i + 2 * ch) + 4 * (sc.unsafe_load[width=WU](i - ch) + sc.unsafe_load[width=WU](i + ch)) + 6 * sc.unsafe_load[width=WU](i)
                dst.unsafe_store(i, ((v + 128) >> 8).cast[dt]())
                i += WU
            for j in range(i, n):
                var v = sc[unsafe_offset=j - 2 * ch] + sc[unsafe_offset=j + 2 * ch] + 4 * (sc[unsafe_offset=j - ch] + sc[unsafe_offset=j + ch]) + 6 * sc[unsafe_offset=j]
                dst[unsafe_offset=j] = Scalar[dt]((v + 128) >> 8)
        _ = len(scratch)

    par(task, NT)
    _ = (rows, cols)
    return o^


def gaussian_blur[dt: DType](src: Mat[dt], ksize: Int, sigma: Float64) -> Mat[dt]:
    var n = ksize
    if n <= 0:
        n = Int(sigma * (3.0 if dt == DType.uint8 else 4.0) * 2 + 1.0 + 0.5) | 1
    comptime if dt == DType.uint8:
        if n == 5 and sigma <= 0 and src.cols >= 3 and src.rows >= 3:
            return _binom5_u8(src)
    var k = gauss_kernel(n, sigma)
    return sepfilter[dt, dt](src, k, k)


def sobel[sd: DType, od: DType](src: Mat[sd], dx: Int, dy: Int, ksize: Int, scale: Float32, mode: Int = 0) -> Mat[od]:
    """ksize -1 selects Scharr."""
    var kx: List[Float32]
    var ky: List[Float32]
    if ksize < 0:
        var s: List[Float32] = [3.0, 10.0, 3.0]
        var d: List[Float32] = [-1.0, 0.0, 1.0]
        kx = d.copy() if dx > 0 else s.copy()
        ky = d^ if dy > 0 else s^
    else:
        kx = deriv_kernel(ksize, dx)
        ky = deriv_kernel(ksize, dy)
    if scale != 1:
        for i in range(len(kx)):
            kx[i] *= scale
    return sepfilter[sd, od](src, kx, ky, mode)


# ---- pyrDown ----------------------------------------------------------------------------------------------


def _tap_mask[n: Int, ch: Int, t: Int]() -> IndexList[n]:
    """Shuffle mask for pyrDown tap t: output pixel j reads pixel 2j + t of the window (lanes past the (n // ch - 3) // 2
    pixels that fit read lane 0)."""
    var m = IndexList[n]()
    for j in range(n):
        m[j] = (j // ch * 2 + t) * ch + j % ch if j < (n // ch - 3) // 2 * ch else 0
    return m


def _pyr_u8[dt: DType, ch: Int](src: Mat[dt]) -> Mat[dt]:
    """pyrDown on uint8 in integers: the column pass sums five rows in uint16 into a padded scratch row, then 64-lane
    windows of it give every output pixel its five taps with five shuffles ((v + 128) >> 8, half up like cv2)."""
    comptime SW = 64
    comptime PO = (SW // ch - 3) // 2
    comptime M0 = _tap_mask[SW, ch, 0]()
    comptime M1 = _tap_mask[SW, ch, 1]()
    comptime M2 = _tap_mask[SW, ch, 2]()
    comptime M3 = _tap_mask[SW, ch, 3]()
    comptime M4 = _tap_mask[SW, ch, 4]()
    comptime WU = simd_width_of[DType.uint16]()
    comptime U16 = DType.uint16
    var oc = (src.cols + 1) // 2
    var orow = (src.rows + 1) // 2
    var rows = src.rows
    var n = src.cols * ch
    var o = Mat[dt](orow, oc, ch, False)
    var ps = src.ptr()
    var po = o.ptr()
    comptime NT = 8

    def task(t: Int) {imm}:
        var scratch = List[UInt16](unsafe_uninit_length=n + 4 * ch + SW)
        var sc = ptr_of(scratch) + 2 * ch
        for r in range(orow * t // NT, orow * (t + 1) // NT):
            var r0 = ps + bidx(2 * r - 2, rows, 0) * n
            var r1 = ps + bidx(2 * r - 1, rows, 0) * n
            var r2 = ps + 2 * r * n
            var r3 = ps + bidx(2 * r + 1, rows, 0) * n
            var r4 = ps + bidx(2 * r + 2, rows, 0) * n
            var i = 0
            while i + WU <= n:
                var v = r0.unsafe_load[width=WU](i).cast[U16]() + r4.unsafe_load[width=WU](i).cast[U16]() + 4 * (r1.unsafe_load[width=WU](i).cast[U16]() + r3.unsafe_load[width=WU](i).cast[U16]()) + 6 * r2.unsafe_load[width=WU](i).cast[U16]()
                sc.unsafe_store(i, v)
                i += WU
            for j in range(i, n):
                sc[unsafe_offset=j] = UInt16(r0[unsafe_offset=j]) + UInt16(r4[unsafe_offset=j]) + 4 * (UInt16(r1[unsafe_offset=j]) + UInt16(r3[unsafe_offset=j])) + 6 * UInt16(r2[unsafe_offset=j])
            for k in range(ch):
                sc[unsafe_offset=k - ch] = sc[unsafe_offset=ch + k]
                sc[unsafe_offset=k - 2 * ch] = sc[unsafe_offset=2 * ch + k]
                sc[unsafe_offset=n + k] = sc[unsafe_offset=n - 2 * ch + k]
                sc[unsafe_offset=n + ch + k] = sc[unsafe_offset=n - 3 * ch + k]
            var dst = po + r * oc * ch
            var c = 0
            while c * ch + SW // 2 <= oc * ch:
                var w = sc.unsafe_load[width=SW]((2 * c - 2) * ch)
                var v = w.shuffle[M0]() + w.shuffle[M4]() + 4 * (w.shuffle[M1]() + w.shuffle[M3]()) + 6 * w.shuffle[M2]()
                dst.unsafe_store(c * ch, ((v.slice[SW // 2]() + 128) >> 8).cast[dt]())
                c += PO
            for q in range(c, oc):
                for k in range(ch):
                    var b = 2 * q * ch + k
                    var v = sc[unsafe_offset=b - 2 * ch] + sc[unsafe_offset=b + 2 * ch] + 4 * (sc[unsafe_offset=b - ch] + sc[unsafe_offset=b + ch]) + 6 * sc[unsafe_offset=b]
                    dst[unsafe_offset=q * ch + k] = Scalar[dt]((v + 128) >> 8)
        _ = len(scratch)

    par(task, NT)
    return o^


def pyr_down[dt: DType](src: Mat[dt]) -> Mat[dt]:
    """5-tap binomial blur (reflect101), then every second row and column; output ceil(n / 2). Per output row the
    column pass runs on contiguous SIMD lanes into one cached row, then the row pass reads every second pixel."""
    comptime if dt == DType.uint8:
        if src.cols >= 3 and src.rows >= 3:
            if src.ch == 1:
                return _pyr_u8[dt, 1](src)
            elif src.ch == 3:
                return _pyr_u8[dt, 3](src)
            elif src.ch == 4:
                return _pyr_u8[dt, 4](src)
    var oc = (src.cols + 1) // 2
    var orow = (src.rows + 1) // 2
    var ch = src.ch
    var n = src.cols * ch
    var o = Mat[dt](orow, oc, ch, False)
    var ps = src.ptr()
    var po = o.ptr()
    comptime NT = 8

    def task(t: Int) {imm}:
        var vrow = List[Float32](unsafe_uninit_length=n)
        var v = ptr_of(vrow)
        for r in range(orow * t // NT, orow * (t + 1) // NT):
            var p0 = ps + bidx(2 * r - 2, src.rows, 0) * n
            var p1 = ps + bidx(2 * r - 1, src.rows, 0) * n
            var p2 = ps + (2 * r) * n
            var p3 = ps + bidx(2 * r + 1, src.rows, 0) * n
            var p4 = ps + bidx(2 * r + 2, src.rows, 0) * n
            var i = 0
            while i + WF <= n:
                var s = _ld[dt, WF](p0, i) + _ld[dt, WF](p1, i) * 4 + _ld[dt, WF](p2, i) * 6 + _ld[dt, WF](p3, i) * 4 + _ld[dt, WF](p4, i)
                v.unsafe_store(i, s)
                i += WF
            for j in range(i, n):
                v[unsafe_offset=j] = Float32(p0[unsafe_offset=j]) + Float32(p1[unsafe_offset=j]) * 4 + Float32(p2[unsafe_offset=j]) * 6 + Float32(p3[unsafe_offset=j]) * 4 + Float32(p4[unsafe_offset=j])
            var hi = max((src.cols - 3) // 2, 0)  # columns 1..hi need no reflection
            for c in range(oc):
                var c0 = (2 * c - 2) * ch
                var c1 = (2 * c - 1) * ch
                var c3 = (2 * c + 1) * ch
                var c4 = (2 * c + 2) * ch
                if c < 1 or c > hi:
                    c0 = bidx(2 * c - 2, src.cols, 0) * ch
                    c1 = bidx(2 * c - 1, src.cols, 0) * ch
                    c3 = bidx(2 * c + 1, src.cols, 0) * ch
                    c4 = bidx(2 * c + 2, src.cols, 0) * ch
                for k in range(ch):
                    var s = v[unsafe_offset=c0 + k] + v[unsafe_offset=c1 + k] * 4 + v[unsafe_offset=2 * c * ch + k] * 6 + v[unsafe_offset=c3 + k] * 4 + v[unsafe_offset=c4 + k]
                    comptime if dt.is_integral():
                        s = floor(s / 256 + 0.5)
                    else:
                        s = s / 256
                    po.unsafe_store(r * oc * ch + c * ch + k, _fin[dt, 1](SIMD[F32, 1](s)))
        _ = len(vrow)

    par(task, NT)
    return o^


# ---- threshold --------------------------------------------------------------------------------------------


def otsu[dt: DType](m: Mat[dt]) -> Int:
    """The threshold cv2 picks for THRESH_OTSU on one 8-bit channel (first maximum of the between-class variance)."""
    var h = List[Float64](length=256, fill=0)
    for v in m.data:
        h[Int(v)] += 1
    var scale = 1.0 / Float64(m.size())
    var mu = Float64(0)
    for i in range(256):
        mu += Float64(i) * h[i] * scale
    var q1 = Float64(0)
    var mu1 = Float64(0)
    var best = Float64(0)
    var val = 0
    for i in range(256):
        var p = h[i] * scale
        mu1 *= q1
        q1 += p
        var q2 = 1.0 - q1
        if min(q1, q2) < 1.1920929e-07 or max(q1, q2) > 1.0 - 1.1920929e-07:
            continue
        mu1 = (mu1 + Float64(i) * p) / q1
        var mu2 = (mu - q1 * mu1) / q2
        var sigma = q1 * q2 * (mu1 - mu2) * (mu1 - mu2)
        if sigma > best:
            best = sigma
            val = i
    return val


def triangle[dt: DType](m: Mat[dt]) -> Int:
    """The threshold cv2 picks for THRESH_TRIANGLE: the histogram point farthest from the line between its peak and
    the far end of the longer tail (cv2 compares a * i + b * h[i], dropping the constant factors)."""
    var h = List[Int](length=256, fill=0)
    for v in m.data:
        h[Int(v)] += 1
    var left = 0
    var right = 0
    var peak = 0
    var top = 0
    for i in range(256):
        if h[i] > 0:
            left = i
            break
    if left > 0:
        left -= 1
    for i in range(255, 0, -1):
        if h[i] > 0:
            right = i
            break
    if right < 255:
        right += 1
    for i in range(256):
        if h[i] > top:
            top = h[i]
            peak = i
    var flipped = False
    if peak - left < right - peak:
        flipped = True
        for i in range(128):
            var t = h[i]
            h[i] = h[255 - i]
            h[255 - i] = t
        left = 255 - right
        peak = 255 - peak
    var thresh = left
    var dist = 0
    for i in range(left + 1, peak + 1):
        var d = top * i + (left - peak) * h[i]
        if d > dist:
            dist = d
            thresh = i
    thresh -= 1
    return 255 - thresh if flipped else thresh


def threshold[dt: DType](src: Mat[dt], thresh: Float64, maxval: Float64, kind: Int) -> Mat[dt]:
    """kind: 0 binary, 1 binary_inv, 2 trunc, 3 tozero, 4 tozero_inv (cv2 THRESH_*)."""
    var o = Mat[dt](src.rows, src.cols, src.ch)
    comptime W = simd_width_of[dt]()
    var th: Scalar[dt]
    comptime if dt.is_integral():
        th = Scalar[dt](floor(thresh))
    else:
        th = Scalar[dt](thresh)
    var mx = Scalar[dt](maxval)
    var n = src.cols * src.ch
    var pa = src.ptr()
    var po = o.ptr()

    def row(r: Int) {imm}:
        var base = r * n
        var i = 0
        while i + W <= n:
            var x = pa.unsafe_load[width=W](base + i)
            var gt = x.gt(th)
            var z = SIMD[dt, W](0)
            var v: SIMD[dt, W]
            if kind == 0:
                v = gt.select(SIMD[dt, W](mx), z)
            elif kind == 1:
                v = gt.select(z, SIMD[dt, W](mx))
            elif kind == 2:
                v = gt.select(SIMD[dt, W](th), x)
            elif kind == 3:
                v = gt.select(x, z)
            else:
                v = gt.select(z, x)
            po.unsafe_store(base + i, v)
            i += W
        for j in range(i, n):
            var x = pa[unsafe_offset=base + j]
            var gt = x > th
            var v: Scalar[dt]
            if kind == 0:
                v = mx if gt else Scalar[dt](0)
            elif kind == 1:
                v = Scalar[dt](0) if gt else mx
            elif kind == 2:
                v = th if gt else x
            elif kind == 3:
                v = x if gt else Scalar[dt](0)
            else:
                v = Scalar[dt](0) if gt else x
            po[unsafe_offset=base + j] = v

    par(row, src.rows)
    return o^


# ---- integral ---------------------------------------------------------------------------------------------


def integral[sd: DType, od: DType](src: Mat[sd]) -> Mat[od]:
    """(rows + 1) x (cols + 1) summed-area table, zero first row and column, per channel."""
    var ch = src.ch
    var o = Mat[od](src.rows + 1, src.cols + 1, ch, False)
    var pa = src.ptr()
    var po = o.ptr()
    var stride = o.cols * ch
    # zero first row and first column (rest is overwritten)
    for i in range(stride):
        po[unsafe_offset=i] = Scalar[od](0)
    for r in range(o.rows):
        for k in range(ch):
            po[unsafe_offset=r * stride + k] = Scalar[od](0)

    def hrow(r: Int) {imm}:
        for k in range(ch):
            var acc = Scalar[od](0)
            var psrc = pa + (r * src.cols) * ch + k
            var pdst = po + (r + 1) * stride + ch + k
            for c in range(src.cols):
                acc += Scalar[od](psrc[unsafe_offset=c * ch])
                pdst[unsafe_offset=c * ch] = acc

    par(hrow, src.rows)

    comptime W = simd_width_of[od]()
    for r in range(1, o.rows):
        var base = r * stride
        var prev = (r - 1) * stride
        var i = 0
        while i + W <= stride:
            var cur = po.unsafe_load[width=W](base + i)
            var pr = po.unsafe_load[width=W](prev + i)
            po.unsafe_store(base + i, cur + pr)
            i += W
        for j in range(i, stride):
            po[unsafe_offset=base + j] += po[unsafe_offset=prev + j]

    return o^


# ---- morphology -------------------------------------------------------------------------------------------


def strel(shape: Int, k: Int) -> List[Int]:
    """cv2.getStructuringElement runs as (dy, x1, x2) triples relative to the anchor k // 2.
    shape 0 rect, 1 cross, 2 ellipse."""
    var out = List[Int]()
    var a = k // 2
    for i in range(k):
        var x1 = 0
        var x2 = k - 1
        if shape == 1:
            x1 = a if i != a else 0
            x2 = a if i != a else k - 1
        elif shape == 2:
            var r = a
            var dy = i - r
            if abs(dy) <= r:
                var inv = 1.0 / Float64(r * r) if r > 0 else 0.0
                var dx = Int(Float64(a) * sqrt(Float64(r * r - dy * dy) * inv) + 0.5) if r > 0 else 0
                x1 = max(a - dx, 0)
                x2 = min(a + dx + 1, k) - 1
            else:
                x1 = 1
                x2 = 0
        if x1 <= x2:
            out.append(i - a)
            out.append(x1 - a)
            out.append(x2 - a)
    return out^


@always_inline
def _rect_hrow[is_min: Bool, dt: DType](s: Ptr[Scalar[dt]], d: Ptr[Scalar[dt]], n: Int, ch: Int, r: Int):
    """d[i] = min (max) of s[i + dx * ch] for dx in -r..r; taps outside the row are skipped (the identity is neutral)."""
    comptime W = simd_width_of[dt]()
    comptime ident = Scalar[dt].MAX if is_min else Scalar[dt].MIN
    var e = r * ch
    var i = e
    while i + W <= n - e:
        var acc = s.unsafe_load[width=W](i - e)
        for dx in range(-r + 1, r + 1):
            var v = s.unsafe_load[width=W](i + dx * ch)
            comptime if is_min:
                acc = min(acc, v)
            else:
                acc = max(acc, v)
        d.unsafe_store(i, acc)
        i += W
    for side in range(2):
        for ii in range(0 if side == 0 else max(i, e), min(e, n) if side == 0 else n):
            var acc = ident
            for dx in range(-r, r + 1):
                var j = ii + dx * ch
                if j >= 0 and j < n:
                    var v = s[unsafe_offset=j]
                    comptime if is_min:
                        if v < acc:
                            acc = v
                    else:
                        if v > acc:
                            acc = v
            d[unsafe_offset=ii] = acc


def morph[is_min: Bool, dt: DType](src: Mat[dt], shape: Int, k: Int, iters: Int) -> Mat[dt]:
    """erode (is_min) or dilate with a k x k structuring element, borders ignored as cv2 does."""
    var runs = strel(shape, k)
    var iters2 = iters
    var kk = k
    if shape == 0 and iters2 > 1:
        # iterating a rect k times equals one pass of (k - 1) * iters + 1
        kk = (k - 1) * iters2 + 1
        iters2 = 1
    var r = kk // 2
    var ch = src.ch
    comptime W = simd_width_of[dt]()
    comptime ident = Scalar[dt].MAX if is_min else Scalar[dt].MIN
    var cur = Mat[dt](0, 0)
    var n = src.cols * ch
    for it in range(iters2):
        var o = Mat[dt](src.rows, src.cols, ch, False)
        var po = o.ptr()
        if shape == 0:
            # a rectangle is separable: min over the row window, then over the column window; the source is
            # read in place and only the r edge columns and rows are clipped
            var tmp = Mat[dt](src.rows, src.cols, ch, False)
            var pt = tmp.ptr()
            var ps = src.ptr() if it == 0 else cur.ptr()

            def hband(b: Int) {imm}:
                for y in range(b * 8, min(b * 8 + 8, src.rows)):
                    _rect_hrow[is_min, dt](ps + y * n, pt + y * n, n, ch, r)

            par(hband, (src.rows + 7) // 8)

            def vband(b: Int) {imm}:
                for y in range(b * 8, min(b * 8 + 8, src.rows)):
                    var y0 = max(y - r, 0)
                    var y1 = min(y + r, src.rows - 1)
                    var i = 0
                    while i + W <= n:
                        var acc = pt.unsafe_load[width=W](y0 * n + i)
                        for yy in range(y0 + 1, y1 + 1):
                            var v = pt.unsafe_load[width=W](yy * n + i)
                            comptime if is_min:
                                acc = min(acc, v)
                            else:
                                acc = max(acc, v)
                        po.unsafe_store(y * n + i, acc)
                        i += W
                    for ii in range(i, n):
                        var acc = pt[unsafe_offset=y0 * n + ii]
                        for yy in range(y0 + 1, y1 + 1):
                            var v = pt[unsafe_offset=yy * n + ii]
                            comptime if is_min:
                                if v < acc:
                                    acc = v
                            else:
                                if v > acc:
                                    acc = v
                        po[unsafe_offset=y * n + ii] = acc

            par(vband, (src.rows + 7) // 8)
            _ = (tmp.rows, cur.rows)
        else:
            var p = pad(src, r, r, 2, ident) if it == 0 else pad(cur, r, r, 2, ident)
            var pp = p.ptr()
            var pstride = p.cols * ch
            var pr = ptr_of(runs)
            var nr = len(runs) // 3

            def row(y: Int) {imm}:
                var i = 0
                while i < n:
                    var w = W if i + W <= n else 1
                    var acc = SIMD[dt, W](ident)
                    for q in range(nr):
                        var base = (y + r + pr[unsafe_offset=3 * q]) * pstride + r * ch + i
                        for dx in range(pr[unsafe_offset=3 * q + 1], pr[unsafe_offset=3 * q + 2] + 1):
                            var v = pp.unsafe_load[width=W](base + dx * ch) if w == W else SIMD[dt, W](pp[unsafe_offset=base + dx * ch])
                            comptime if is_min:
                                acc = min(acc, v)
                            else:
                                acc = max(acc, v)
                    if w == W:
                        po.unsafe_store(y * n + i, acc)
                    else:
                        po[unsafe_offset=y * n + i] = acc[0]
                    i += w

            par(row, src.rows)
            _ = (p.rows, len(runs))
        cur = o^
    return cur^


# ---- cvtColor ---------------------------------------------------------------------------------------------

def cvt_code(name: String) raises -> Int:
    var names: List[String] = ["bgr2gray", "rgb2gray", "gray2bgr", "bgr2rgb", "bgr2hsv", "hsv2bgr", "bgr2ycrcb", "ycrcb2bgr", "bgr2lab", "lab2bgr", "bgr2luv", "luv2bgr"]
    for i in range(len(names)):
        if names[i] == name:
            return i
    raise Error("mojocv: cvtColor code " + name)


def _sat8(v: Float32) -> UInt8:
    return UInt8(min(max(floor(v + 0.5), 0), 255))


def _srgb_gamma(v: Float32) -> Float32:
    return v / 12.92 if v <= 0.04045 else Float32(((Float64(v) + 0.055) / 1.055) ** 2.4)


def _srgb_ungamma(v: Float32) -> Float32:
    return 12.92 * v if v <= 0.0031308 else Float32(1.055 * Float64(v) ** (1.0 / 2.4) - 0.055)


comptime LUT_N = 4096


def _lut[w: Int](p: Ptr[Float32], v: SIMD[F32, w]) -> SIMD[F32, w]:
    """Linear interpolation in a LUT_N + 1 entry table over [0, 1]."""
    var t = min(max(v, 0.0), 1.0) * Float32(LUT_N)
    var i = min(floor(t).cast[DType.int32](), Int32(LUT_N - 1))
    var f = t - i.cast[F32]()
    var lo = p.unsafe_gather(i)
    return lo + (p.unsafe_gather(i + 1) - lo) * f


def _lab_luv_vec[dt: DType, w: Int, luv: Bool, inverse: Bool](ps: Ptr[Scalar[dt]], po: Ptr[Scalar[dt]], gam8: Ptr[Float32], gam: Ptr[Float32], cbrt_t: Ptr[Float32], ungam: Ptr[Float32]):
    """BGR <-> Lab / Luv for w pixels (3 interleaved channels each) at ps, written to po."""
    comptime is8 = dt == DType.uint8
    comptime XN = Float32(0.950456)
    comptime ZN = Float32(1.088754)
    comptime UN = Float32(4.0 * 0.950456 / (0.950456 + 15.0 + 3.0 * 1.088754))
    comptime VN = Float32(9.0 / (0.950456 + 15.0 + 3.0 * 1.088754))
    comptime LT = Float32(0.008856)
    comptime V = SIMD[F32, w]
    comptime if not inverse:
        var b: V
        var g: V
        var r: V
        comptime if is8:
            b = gam8.unsafe_gather(ps.strided_load[width=w](3).cast[DType.int32]())
            g = gam8.unsafe_gather((ps + 1).strided_load[width=w](3).cast[DType.int32]())
            r = gam8.unsafe_gather((ps + 2).strided_load[width=w](3).cast[DType.int32]())
        else:
            b = _lut(gam, ps.strided_load[width=w](3).cast[F32]())
            g = _lut(gam, (ps + 1).strided_load[width=w](3).cast[F32]())
            r = _lut(gam, (ps + 2).strided_load[width=w](3).cast[F32]())
        var xx = 0.412453 * r + 0.357580 * g + 0.180423 * b
        var yy = 0.212671 * r + 0.715160 * g + 0.072169 * b
        var zz = 0.019334 * r + 0.119193 * g + 0.950227 * b
        var fy = _lut(cbrt_t, yy)
        var l = yy.gt(LT).select(116.0 * fy - 16.0, 903.3 * yy)
        var c1: V
        var c2: V
        comptime if luv:
            var d = Float32(52.0) / max(xx + 15.0 * yy + 3.0 * zz, V(1.1920929e-07))
            c1 = l * (xx * d - 13.0 * UN)
            c2 = l * (2.25 * yy * d - 13.0 * VN)
        else:
            c1 = 500.0 * (_lut(cbrt_t, xx / XN) - fy)
            c2 = 200.0 * (fy - _lut(cbrt_t, zz / ZN))
        comptime if is8:
            var o0 = l * 2.55
            var o1 = (c1 + 134.0) * (255.0 / 354.0) if luv else c1 + 128.0
            var o2 = (c2 + 140.0) * (255.0 / 262.0) if luv else c2 + 128.0
            po.strided_store[width=w](min(max(floor(o0 + 0.5), 0.0), 255.0).cast[dt](), 3)
            (po + 1).strided_store[width=w](min(max(floor(o1 + 0.5), 0.0), 255.0).cast[dt](), 3)
            (po + 2).strided_store[width=w](min(max(floor(o2 + 0.5), 0.0), 255.0).cast[dt](), 3)
        else:
            po.strided_store[width=w](l.cast[dt](), 3)
            (po + 1).strided_store[width=w](c1.cast[dt](), 3)
            (po + 2).strided_store[width=w](c2.cast[dt](), 3)
    else:
        var l = ps.strided_load[width=w](3).cast[F32]()
        var c1 = (ps + 1).strided_load[width=w](3).cast[F32]()
        var c2 = (ps + 2).strided_load[width=w](3).cast[F32]()
        comptime if is8:
            l = l * (100.0 / 255.0)
            c1 = c1 * (354.0 / 255.0) - 134.0 if luv else c1 - 128.0
            c2 = c2 * (262.0 / 255.0) - 140.0 if luv else c2 - 128.0
        var xx: V
        var yy: V
        var zz: V
        comptime if luv:
            var f = (l + 16.0) * (1.0 / 116.0)
            yy = l.gt(7.9996248).select(f * f * f, l * (1.0 / 903.3))
            var s = 13.0 * max(l, V(1e-6))
            var up = c1 / s + UN
            var vp = c2 / s + VN
            xx = l.gt(0).select(yy * 2.25 * up / vp, V(0))
            zz = l.gt(0).select(yy * (12.0 - 3.0 * up - 20.0 * vp) / (4.0 * vp), V(0))
        else:
            var hi = l.gt(7.9996248)
            var fy = hi.select((l + 16.0) * (1.0 / 116.0), 7.787 * (l * (1.0 / 903.3)) + 16.0 / 116.0)
            yy = hi.select(fy * fy * fy, l * (1.0 / 903.3))
            var fx = c1 * (1.0 / 500.0) + fy
            var fz = fy - c2 * (1.0 / 200.0)
            xx = XN * fx.le(0.20689303).select((fx - 16.0 / 116.0) * (1.0 / 7.787), fx * fx * fx)
            zz = ZN * fz.le(0.20689303).select((fz - 16.0 / 116.0) * (1.0 / 7.787), fz * fz * fz)
        var r = _lut(ungam, 3.240479 * xx - 1.53715 * yy - 0.498535 * zz)
        var g = _lut(ungam, -0.969256 * xx + 1.875991 * yy + 0.041556 * zz)
        var b = _lut(ungam, 0.055648 * xx - 0.204043 * yy + 1.057311 * zz)
        comptime if is8:
            po.strided_store[width=w](min(max(floor(b * 255.0 + 0.5), 0.0), 255.0).cast[dt](), 3)
            (po + 1).strided_store[width=w](min(max(floor(g * 255.0 + 0.5), 0.0), 255.0).cast[dt](), 3)
            (po + 2).strided_store[width=w](min(max(floor(r * 255.0 + 0.5), 0.0), 255.0).cast[dt](), 3)
        else:
            po.strided_store[width=w](b.cast[dt](), 3)
            (po + 1).strided_store[width=w](g.cast[dt](), 3)
            (po + 2).strided_store[width=w](r.cast[dt](), 3)


def cvt_lab_luv[dt: DType](src: Mat[dt], code: Int) raises -> Mat[dt]:
    """BGR <-> CIE Lab / Luv (D65, sRGB gamma) with cv2's float formulas, WF pixels at a time. Gamma, cube root and
    inverse gamma are interpolated from 4096-cell tables (8-bit input uses an exact 256-entry gamma table). 8-bit
    images are scaled the way cv2 does (L * 255 / 100; a, b + 128; u + 134 and v + 140 stretched to 255); cv2 reaches
    the 8-bit result through fixed-point tables, so those agree to within 1 to 2."""
    if src.ch != 3:
        raise Error("mojocv: cvtColor Lab/Luv wants 3 channels")
    var o = Mat[dt](src.rows, src.cols, 3, False)
    var ps = src.ptr()
    var po = o.ptr()
    var n = src.cols
    var gam8 = List[Float32](length=256, fill=0)
    for i in range(256):
        gam8[i] = _srgb_gamma(Float32(i) / 255.0)
    var gam = List[Float32](length=LUT_N + 1, fill=0)
    var cbrt_t = List[Float32](length=LUT_N + 1, fill=0)
    var ungam = List[Float32](length=LUT_N + 1, fill=0)
    for i in range(LUT_N + 1):
        var t = Float32(i) / Float32(LUT_N)
        gam[i] = _srgb_gamma(t)
        cbrt_t[i] = Float32(cbrt(Float64(t))) if t > 0.008856 else 7.787 * t + 16.0 / 116.0
        ungam[i] = _srgb_ungamma(t)
    var p8 = ptr_of(gam8)
    var pg = ptr_of(gam)
    var pc = ptr_of(cbrt_t)
    var pu = ptr_of(ungam)
    var inverse = code == 9 or code == 11
    var luv = code == 10 or code == 11

    def row(y: Int) {imm}:
        var x = 0
        while x < n:
            var i = (y * n + x) * 3
            var vec = x + WF <= n
            # code is a runtime value: pick the comptime variant once per chunk
            if inverse:
                if luv:
                    if vec:
                        _lab_luv_vec[dt, WF, True, True](ps + i, po + i, p8, pg, pc, pu)
                    else:
                        _lab_luv_vec[dt, 1, True, True](ps + i, po + i, p8, pg, pc, pu)
                else:
                    if vec:
                        _lab_luv_vec[dt, WF, False, True](ps + i, po + i, p8, pg, pc, pu)
                    else:
                        _lab_luv_vec[dt, 1, False, True](ps + i, po + i, p8, pg, pc, pu)
            else:
                if luv:
                    if vec:
                        _lab_luv_vec[dt, WF, True, False](ps + i, po + i, p8, pg, pc, pu)
                    else:
                        _lab_luv_vec[dt, 1, True, False](ps + i, po + i, p8, pg, pc, pu)
                else:
                    if vec:
                        _lab_luv_vec[dt, WF, False, False](ps + i, po + i, p8, pg, pc, pu)
                    else:
                        _lab_luv_vec[dt, 1, False, False](ps + i, po + i, p8, pg, pc, pu)
            x += WF if vec else 1

    par(row, src.rows)
    _ = (len(gam8), len(gam), len(cbrt_t), len(ungam))
    return o^


def cvt_color[dt: DType](src: Mat[dt], code: Int) raises -> Mat[dt]:
    """code indexes CODES. 8-bit images use cv2's integer arithmetic; float images use its float formulas."""
    if code >= 8:
        return cvt_lab_luv(src, code)
    var oc = 1 if (code == 0 or code == 1) else 3
    var ic = 1 if code == 2 else 3
    if src.ch != ic:
        raise Error("mojocv: cvtColor code " + String(code) + " wants " + String(ic) + " channels")
    var o = Mat[dt](src.rows, src.cols, oc)
    var pa = src.ptr()
    var po = o.ptr()
    var npx = src.cols

    def row(r: Int) {imm}:
        comptime if dt == DType.uint8:
            if code == 3:
                var base = r * npx * 3
                var qa = pa.unsafe_bitcast[UInt8]()
                var qo = po.unsafe_bitcast[UInt8]()
                var c = 0
                while c + 8 < npx:
                    var v = (qa + base + c * 3).unsafe_load[width=32]()
                    var s = v.shuffle[2,1,0,5,4,3,8,7,6,11,10,9,14,13,12,17,16,15,20,19,18,23,22,21,24,25,26,27,28,29,30,31]()
                    (qo + base + c * 3).unsafe_store[width=32](s)
                    c += 8
                for cc in range(c, npx):
                    var i = base + cc * 3
                    qo[i] = qa[i + 2]
                    qo[i + 1] = qa[i + 1]
                    qo[i + 2] = qa[i]
                return
        comptime if dt == DType.uint8:
            if code == 0 or code == 1:
                var base = r * npx * 3
                var qa = pa.unsafe_bitcast[UInt8]()
                var qo = po.unsafe_bitcast[UInt8]()
                var c = 0
                if code == 0:
                    while c + 16 <= npx:
                        var v0 = (qa + base + c * 3).unsafe_load[width=32]()
                        var blue0 = SIMD[DType.uint8, 8](v0[0], v0[3], v0[6], v0[9], v0[12], v0[15], v0[18], v0[21])
                        var green0 = SIMD[DType.uint8, 8](v0[1], v0[4], v0[7], v0[10], v0[13], v0[16], v0[19], v0[22])
                        var red0 = SIMD[DType.uint8, 8](v0[2], v0[5], v0[8], v0[11], v0[14], v0[17], v0[20], v0[23])
                        var y0 = (blue0.cast[DType.int32]() * 3735 + green0.cast[DType.int32]() * 19235 + red0.cast[DType.int32]() * 9798 + 16384) >> 15
                        qo.unsafe_store(r * npx + c, y0.cast[DType.uint8]())
                        var v1 = (qa + base + (c + 8) * 3).unsafe_load[width=32]()
                        var blue1 = SIMD[DType.uint8, 8](v1[0], v1[3], v1[6], v1[9], v1[12], v1[15], v1[18], v1[21])
                        var green1 = SIMD[DType.uint8, 8](v1[1], v1[4], v1[7], v1[10], v1[13], v1[16], v1[19], v1[22])
                        var red1 = SIMD[DType.uint8, 8](v1[2], v1[5], v1[8], v1[11], v1[14], v1[17], v1[20], v1[23])
                        var y1 = (blue1.cast[DType.int32]() * 3735 + green1.cast[DType.int32]() * 19235 + red1.cast[DType.int32]() * 9798 + 16384) >> 15
                        qo.unsafe_store(r * npx + c + 8, y1.cast[DType.uint8]())
                        c += 16
                    while c + 8 <= npx:
                        var v = (qa + base + c * 3).unsafe_load[width=32]()
                        var blue = SIMD[DType.uint8, 8](v[0], v[3], v[6], v[9], v[12], v[15], v[18], v[21])
                        var green = SIMD[DType.uint8, 8](v[1], v[4], v[7], v[10], v[13], v[16], v[19], v[22])
                        var red = SIMD[DType.uint8, 8](v[2], v[5], v[8], v[11], v[14], v[17], v[20], v[23])
                        var y = (blue.cast[DType.int32]() * 3735 + green.cast[DType.int32]() * 19235 + red.cast[DType.int32]() * 9798 + 16384) >> 15
                        qo.unsafe_store(r * npx + c, y.cast[DType.uint8]())
                        c += 8
                else:
                    while c + 16 <= npx:
                        var v0 = (qa + base + c * 3).unsafe_load[width=32]()
                        var blue0 = SIMD[DType.uint8, 8](v0[2], v0[5], v0[8], v0[11], v0[14], v0[17], v0[20], v0[23])
                        var green0 = SIMD[DType.uint8, 8](v0[1], v0[4], v0[7], v0[10], v0[13], v0[16], v0[19], v0[22])
                        var red0 = SIMD[DType.uint8, 8](v0[0], v0[3], v0[6], v0[9], v0[12], v0[15], v0[18], v0[21])
                        var y0 = (blue0.cast[DType.int32]() * 3735 + green0.cast[DType.int32]() * 19235 + red0.cast[DType.int32]() * 9798 + 16384) >> 15
                        qo.unsafe_store(r * npx + c, y0.cast[DType.uint8]())
                        var v1 = (qa + base + (c + 8) * 3).unsafe_load[width=32]()
                        var blue1 = SIMD[DType.uint8, 8](v1[2], v1[5], v1[8], v1[11], v1[14], v1[17], v1[20], v1[23])
                        var green1 = SIMD[DType.uint8, 8](v1[1], v1[4], v1[7], v1[10], v1[13], v1[16], v1[19], v1[22])
                        var red1 = SIMD[DType.uint8, 8](v1[0], v1[3], v1[6], v1[9], v1[12], v1[15], v1[18], v1[21])
                        var y1 = (blue1.cast[DType.int32]() * 3735 + green1.cast[DType.int32]() * 19235 + red1.cast[DType.int32]() * 9798 + 16384) >> 15
                        qo.unsafe_store(r * npx + c + 8, y1.cast[DType.uint8]())
                        c += 16
                    while c + 8 <= npx:
                        var v = (qa + base + c * 3).unsafe_load[width=32]()
                        var blue = SIMD[DType.uint8, 8](v[2], v[5], v[8], v[11], v[14], v[17], v[20], v[23])
                        var green = SIMD[DType.uint8, 8](v[1], v[4], v[7], v[10], v[13], v[16], v[19], v[22])
                        var red = SIMD[DType.uint8, 8](v[0], v[3], v[6], v[9], v[12], v[15], v[18], v[21])
                        var y = (blue.cast[DType.int32]() * 3735 + green.cast[DType.int32]() * 19235 + red.cast[DType.int32]() * 9798 + 16384) >> 15
                        qo.unsafe_store(r * npx + c, y.cast[DType.uint8]())
                        c += 8
                for cc in range(c, npx):
                    var i = base + cc * 3
                    var x0 = Int(qa[unsafe_offset=i])
                    var x1 = Int(qa[unsafe_offset=i + 1])
                    var x2 = Int(qa[unsafe_offset=i + 2])
                    var b = x0 if code == 0 else x2
                    var rr = x2 if code == 0 else x0
                    qo[unsafe_offset=r * npx + cc] = UInt8((b * 3735 + x1 * 19235 + rr * 9798 + 16384) >> 15)
                return
        comptime if dt == DType.uint8:
            if code == 6:
                var base = r * npx * 3
                var qa = pa.unsafe_bitcast[UInt8]()
                var qo = po.unsafe_bitcast[UInt8]()
                var c = 0
                while c + 16 <= npx:
                    var v0 = (qa + base + c * 3).unsafe_load[width=32]()
                    var v1 = (qa + base + c * 3 + 32).unsafe_load[width=16]()
                    var b16 = SIMD[DType.uint8, 16](v0[0], v0[3], v0[6], v0[9], v0[12], v0[15], v0[18], v0[21], v0[24], v0[27], v0[30], v1[1], v1[4], v1[7], v1[10], v1[13]).cast[DType.int32]()
                    var g16 = SIMD[DType.uint8, 16](v0[1], v0[4], v0[7], v0[10], v0[13], v0[16], v0[19], v0[22], v0[25], v0[28], v0[31], v1[2], v1[5], v1[8], v1[11], v1[14]).cast[DType.int32]()
                    var r16 = SIMD[DType.uint8, 16](v0[2], v0[5], v0[8], v0[11], v0[14], v0[17], v0[20], v0[23], v0[26], v0[29], v1[0], v1[3], v1[6], v1[9], v1[12], v1[15]).cast[DType.int32]()
                    var y16 = (b16 * 3735 + g16 * 19235 + r16 * 9798 + 16384) >> 15
                    var dl = SIMD[DType.int32, 16]((128 << 14) + (1 << 13))
                    var cb16 = ((b16 - y16) * 9241 + dl) >> 14
                    var cr16 = ((r16 - y16) * 11682 + dl) >> 14
                    var yi = y16.clamp(0, 255).cast[DType.uint8]()
                    var cbi = cb16.clamp(0, 255).cast[DType.uint8]()
                    var cri = cr16.clamp(0, 255).cast[DType.uint8]()
                    (qo + base + c * 3).unsafe_strided_store[width=16](yi, 3)
                    (qo + base + c * 3 + 1).unsafe_strided_store[width=16](cri, 3)
                    (qo + base + c * 3 + 2).unsafe_strided_store[width=16](cbi, 3)
                    c += 16
                while c + 8 <= npx:
                    var v = (qa + base + c * 3).unsafe_load[width=32]()
                    var b8 = SIMD[DType.uint8, 8](v[0], v[3], v[6], v[9], v[12], v[15], v[18], v[21]).cast[DType.int32]()
                    var g8 = SIMD[DType.uint8, 8](v[1], v[4], v[7], v[10], v[13], v[16], v[19], v[22]).cast[DType.int32]()
                    var r8 = SIMD[DType.uint8, 8](v[2], v[5], v[8], v[11], v[14], v[17], v[20], v[23]).cast[DType.int32]()
                    var y8 = (b8 * 3735 + g8 * 19235 + r8 * 9798 + 16384) >> 15
                    var dl8 = SIMD[DType.int32, 8]((128 << 14) + (1 << 13))
                    var cb8 = ((b8 - y8) * 9241 + dl8) >> 14
                    var cr8 = ((r8 - y8) * 11682 + dl8) >> 14
                    var yi8 = y8.clamp(0, 255).cast[DType.uint8]()
                    var cbi8 = cb8.clamp(0, 255).cast[DType.uint8]()
                    var cri8 = cr8.clamp(0, 255).cast[DType.uint8]()
                    (qo + base + c * 3).unsafe_strided_store[width=8](yi8, 3)
                    (qo + base + c * 3 + 1).unsafe_strided_store[width=8](cri8, 3)
                    (qo + base + c * 3 + 2).unsafe_strided_store[width=8](cbi8, 3)
                    c += 8
                for cc in range(c, npx):
                    var i = base + cc * 3
                    var x0 = Int(qa[unsafe_offset=i])
                    var x1 = Int(qa[unsafe_offset=i + 1])
                    var x2 = Int(qa[unsafe_offset=i + 2])
                    var y = (x0 * 3735 + x1 * 19235 + x2 * 9798 + 16384) >> 15
                    var dl = (128 << 14) + (1 << 13)
                    qo[unsafe_offset=i] = UInt8(y)
                    qo[unsafe_offset=i + 1] = UInt8(max(0, min(255, ((x2 - y) * 11682 + dl) >> 14)))
                    qo[unsafe_offset=i + 2] = UInt8(max(0, min(255, ((x0 - y) * 9241 + dl) >> 14)))
                return
        var c0 = 0
        comptime if dt == DType.uint8:

            # 16 pixels at a time for the channel-shuffle and gray conversions
            if code == 0 or code == 1 or code == 3:
                var qa = pa.unsafe_bitcast[UInt8]()
                var qo = po.unsafe_bitcast[UInt8]()
                while c0 + 16 <= npx:
                    var i = (r * npx + c0) * 3
                    var b = (qa + i).unsafe_strided_load[width=16](3)
                    var g = (qa + i + 1).unsafe_strided_load[width=16](3)
                    var rr = (qa + i + 2).unsafe_strided_load[width=16](3)
                    if code == 3:
                        (qo + i).unsafe_strided_store[width=16](rr, 3)
                        (qo + i + 1).unsafe_strided_store[width=16](g, 3)
                        (qo + i + 2).unsafe_strided_store[width=16](b, 3)
                    else:
                        var bb = b if code == 0 else rr
                        var r2 = rr if code == 0 else b
                        var y = (bb.cast[DType.int32]() * 3735 + g.cast[DType.int32]() * 19235 + r2.cast[DType.int32]() * 9798 + 16384) >> 15
                        qo.unsafe_store(r * npx + c0, y.cast[DType.uint8]())
                    c0 += 16
        for c in range(c0, npx):
            var i = (r * npx + c) * ic
            var j = (r * npx + c) * oc
            comptime if dt == DType.uint8:
                var qa = pa.unsafe_bitcast[UInt8]()
                var qo = po.unsafe_bitcast[UInt8]()
                var x0 = Int(qa[unsafe_offset=i])
                var x1 = Int(qa[unsafe_offset=i + 1]) if ic == 3 else 0
                var x2 = Int(qa[unsafe_offset=i + 2]) if ic == 3 else 0
                if code == 0 or code == 1:
                    var b = x0 if code == 0 else x2
                    var rr = x2 if code == 0 else x0
                    qo[unsafe_offset=j] = UInt8((b * 3735 + x1 * 19235 + rr * 9798 + 16384) >> 15)
                elif code == 2:
                    qo[unsafe_offset=j] = UInt8(x0)
                    qo[unsafe_offset=j + 1] = UInt8(x0)
                    qo[unsafe_offset=j + 2] = UInt8(x0)
                elif code == 3:
                    qo[unsafe_offset=j] = UInt8(x2)
                    qo[unsafe_offset=j + 1] = UInt8(x1)
                    qo[unsafe_offset=j + 2] = UInt8(x0)
                elif code == 4:
                    var v = max(x0, max(x1, x2))
                    var diff = v - min(x0, min(x1, x2))
                    var sdiv = Int(floor(Float64(255 << 12) / Float64(v) + 0.5)) if v > 0 else 0
                    var hdiv = Int(floor(Float64(180 << 12) / Float64(6 * diff) + 0.5)) if diff > 0 else 0
                    var h: Int
                    if v == x2:
                        h = x1 - x0
                    elif v == x1:
                        h = x0 - x2 + 2 * diff
                    else:
                        h = x2 - x1 + 4 * diff
                    h = (h * hdiv + (1 << 11)) >> 12
                    h += 180 if h < 0 else 0
                    qo[unsafe_offset=j] = UInt8(h)
                    qo[unsafe_offset=j + 1] = UInt8((diff * sdiv + (1 << 11)) >> 12)
                    qo[unsafe_offset=j + 2] = UInt8(v)
                elif code == 5:
                    var h = Float32(x0) * (6.0 / 180.0)
                    var s = Float32(x1) * (1.0 / 255.0)
                    var v = Float32(x2) * (1.0 / 255.0)
                    var sec = Int(floor(h))
                    var f = h - Float32(sec)
                    sec = sec % 6
                    var t0 = v
                    var t1 = v * (1 - s)
                    var t2 = v * (1 - s * f)
                    var t3 = v * (1 - s * (1 - f))
                    var tab = SIMD[F32, 4](t0, t1, t2, t3)
                    var bi = 0
                    var gi = 0
                    var ri = 0
                    if sec == 0:
                        bi, gi, ri = 1, 3, 0
                    elif sec == 1:
                        bi, gi, ri = 1, 0, 2
                    elif sec == 2:
                        bi, gi, ri = 3, 0, 1
                    elif sec == 3:
                        bi, gi, ri = 0, 2, 1
                    elif sec == 4:
                        bi, gi, ri = 0, 1, 3
                    else:
                        bi, gi, ri = 2, 1, 0
                    qo[unsafe_offset=j] = _sat8(tab[bi] * 255)
                    qo[unsafe_offset=j + 1] = _sat8(tab[gi] * 255)
                    qo[unsafe_offset=j + 2] = _sat8(tab[ri] * 255)
                elif code == 6:
                    var y = (x0 * 3735 + x1 * 19235 + x2 * 9798 + 16384) >> 15
                    var dl = (128 << 14) + (1 << 13)
                    qo[unsafe_offset=j] = UInt8(y)
                    qo[unsafe_offset=j + 1] = UInt8(max(0, min(255, ((x2 - y) * 11682 + dl) >> 14)))
                    qo[unsafe_offset=j + 2] = UInt8(max(0, min(255, ((x0 - y) * 9241 + dl) >> 14)))
                else:
                    var cr = Float32(x1 - 128)
                    var cb = Float32(x2 - 128)
                    qo[unsafe_offset=j + 2] = _sat8(Float32(x0) + 1.403 * cr)
                    qo[unsafe_offset=j + 1] = _sat8(Float32(x0) - 0.714 * cr - 0.344 * cb)
                    qo[unsafe_offset=j] = _sat8(Float32(x0) + 1.773 * cb)
            else:
                var x0 = pa[unsafe_offset=i]
                if code == 0 or code == 1:
                    var b = x0 if code == 0 else pa[unsafe_offset=i + 2]
                    var rr = pa[unsafe_offset=i + 2] if code == 0 else x0
                    po[unsafe_offset=j] = Scalar[dt](0.114) * b + Scalar[dt](0.587) * pa[unsafe_offset=i + 1] + Scalar[dt](0.299) * rr
                elif code == 2:
                    for k in range(3):
                        po[unsafe_offset=j + k] = x0
                elif code == 3:
                    po[unsafe_offset=j] = pa[unsafe_offset=i + 2]
                    po[unsafe_offset=j + 1] = pa[unsafe_offset=i + 1]
                    po[unsafe_offset=j + 2] = x0

    par(row, src.rows)
    return o^


# ---- resize -----------------------------------------------------------------------------------------------


def cubic_w(x: Float64) -> SIMD[DType.float64, 4]:
    """cv2 bicubic weights for the taps -1, 0, 1, 2 around a sample at fraction x, A = -0.75."""
    comptime A = -0.75
    var w0 = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A
    var w1 = ((A + 2) * x - (A + 3)) * x * x + 1
    var w2 = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1
    return SIMD[DType.float64, 4](w0, w1, w2, 1.0 - w0 - w1 - w2)


def lanczos_w(x: Float64) -> SIMD[DType.float64, 8]:
    """cv2 Lanczos4 weights for the taps -3 .. 4 around a sample at fraction x."""
    var c = SIMD[DType.float64, 8](0)
    if x < 1.1920929e-07:
        c[3] = 1
        return c
    comptime s45 = 0.70710678118654752440084436210485
    var cs = SIMD[DType.float64, 16](1, 0, -s45, -s45, 0, 1, s45, -s45, -1, 0, s45, s45, 0, -1, -s45, s45)
    var pi = 3.14159265358979323846
    var y0 = -(x + 3) * pi * 0.25
    var s0 = sin(y0)
    var c0 = cos(y0)
    var sum = Float64(0)
    for i in range(8):
        var y = -(x + 3 - Float64(i)) * pi * 0.25
        c[i] = Float64(Float32((cs[2 * i] * s0 + cs[2 * i + 1] * c0) / (y * y)))
        sum += c[i]
    return c * Float64(Float32(1.0 / sum))


struct Taps(Movable):
    """For each output index: `n` source indices (clamped) and weights."""

    var n: Int
    var idx: List[Int]
    var w: List[Float32]

    def __init__(out self, ssize: Int, dsize: Int, scale: Float64, kind: Int):
        """kind 0 nearest, 1 linear, 2 cubic, 3 lanczos4, 4 area (scale >= 1: every source pixel weighted by the
        part of it inside the destination cell, zero-weight padding up to n taps; when enlarging, a destination pixel sees one or two source pixels)"""
        self.n = 1 if kind == 0 else (2 if kind == 1 else (4 if kind == 2 else (8 if kind == 3 else Int(ceil(scale)) + 2)))
        self.idx = List[Int](length=dsize * self.n, fill=0)
        self.w = List[Float32](length=dsize * self.n, fill=0)
        for d in range(dsize):
            if kind == 0:
                self.idx[d] = min(Int(floor(Float64(d) * scale)), ssize - 1)
                self.w[d] = 1
                continue
            if kind == 4:
                var fs1 = Float64(d) * scale
                var fs2 = fs1 + scale
                var cell = min(scale, Float64(ssize) - fs1)
                var s2 = min(Int(floor(fs2)), ssize - 1)
                var s1 = min(Int(ceil(fs1)), s2)
                var t = 0
                if Float64(s1) - fs1 > 1e-3:
                    self.idx[d * self.n] = s1 - 1
                    self.w[d * self.n] = Float32((Float64(s1) - fs1) / cell)
                    t = 1
                for sx in range(s1, s2):
                    self.idx[d * self.n + t] = sx
                    self.w[d * self.n + t] = Float32(1.0 / cell)
                    t += 1
                if fs2 - Float64(s2) > 1e-3:
                    self.idx[d * self.n + t] = s2
                    self.w[d * self.n + t] = Float32(min(min(fs2 - Float64(s2), 1.0), cell) / cell)
                continue
            var f = (Float64(d) + 0.5) * scale - 0.5
            var s = Int(floor(f))
            f -= Float64(s)
            if kind == 1 and s < 0:
                f = 0
                s = 0
            var half = self.n // 2 - 1
            var ww = SIMD[DType.float64, 8](0)
            if kind == 1:
                ww[0] = 1 - f
                ww[1] = f
            elif kind == 2:
                var c = cubic_w(f)
                for t in range(4):
                    ww[t] = c[t]
            else:
                ww = lanczos_w(f)
            for t in range(self.n):
                self.idx[d * self.n + t] = min(max(s - half + t, 0), ssize - 1)
                self.w[d * self.n + t] = Float32(ww[t])


def _pair_mask[n: Int, ch: Int, odd: Int]() -> IndexList[n]:
    """Shuffle mask taking pixel 2p + odd of interleaved pixels for output pixel p, for the n // (2 * ch) pixels a
    vector of n bytes holds (the rest of the lanes read lane 0)."""
    var m = IndexList[n]()
    for j in range(n):
        m[j] = (j // ch) * 2 * ch + odd * ch + j % ch if j < n // (2 * ch) * ch else 0
    return m


def _half_rows[dt: DType, ch: Int](pa: Ptr[Scalar[dt]], po: Ptr[Scalar[dt]], dw: Int, r0: Int, r1: Int):
    """Output rows r0..r1 of the exact 2x linear shrink: the mean of each 2x2 block, rounded half up (cv2's INTER_AREA
    fast path). One 64-byte load per source row, added in uint16, even and odd pixels picked with one shuffle each;
    the store is n // 2 bytes wide, so a block may write up to ch - 1 bytes past its pixels (the next block or the
    scalar tail overwrites them)."""
    comptime SW = 64
    comptime P = SW // (2 * ch)
    comptime G = SW // 2 - P * ch
    comptime me = _pair_mask[SW, ch, 0]()
    comptime mo = _pair_mask[SW, ch, 1]()
    var sn = 2 * dw * ch
    for r in range(r0, r1):
        var a = pa + 2 * r * sn
        var b = a + sn
        var o = po + r * dw * ch
        var p = 0
        while p + P + (1 if G > 0 else 0) <= dw:
            var s = a.unsafe_load[width=SW](2 * p * ch).cast[DType.uint16]() + b.unsafe_load[width=SW](2 * p * ch).cast[DType.uint16]()
            var v = (s.shuffle[me]().slice[SW // 2]() + s.shuffle[mo]().slice[SW // 2]() + 2) >> 2
            o.unsafe_store(p * ch, v.cast[dt]())
            p += P
        for q in range(p, dw):
            for k in range(ch):
                var i = 2 * q * ch + k
                o[unsafe_offset=q * ch + k] = Scalar[dt]((UInt16(a[unsafe_offset=i]) + UInt16(a[unsafe_offset=i + ch]) + UInt16(b[unsafe_offset=i]) + UInt16(b[unsafe_offset=i + ch]) + 2) >> 2)


def _resize_lin_u8[dt: DType, ch: Int](src: Mat[dt], dw: Int, dh: Int) -> Mat[dt]:
    """uint8 bilinear resize, ch 1 or 3, any ratio. Horizontal pass first, once per source row (a two-slot cache per
    worker: output rows ask for source rows y0 and y1, and the next output row mostly re-asks one of them): one
    8-byte (ch 3) or 4-byte (ch 1) gather per output pixel holds both taps of all channels, the row lands in a float32
    buffer laid out in blocks of WF pixels per channel. Vertical blend of two such rows, then the channels are
    interleaved by one shuffle and stored. The source row that ends the image is copied to a padded scratch first, so
    the wide gathers never read past the buffer."""
    var tx = Taps(src.cols, dw, Float64(src.cols) / Float64(dw), 1)
    var ty = Taps(src.rows, dh, Float64(src.rows) / Float64(dh), 1)
    var nb = (dw + WF - 1) // WF
    var o = Mat[dt](dh, dw, ch, False)
    var cols = src.cols
    var rows = src.rows
    var pa = src.ptr()
    var po = o.ptr()
    # per output column (padded to whole blocks): byte offset of the left tap, weights of the left and right pixel;
    # a right tap clamped onto the left one is folded into the left weight so the second pixel is never read
    var off = List[Int](length=nb * WF, fill=0)
    var cw0 = List[Float32](length=nb * WF, fill=0)
    var cw1 = List[Float32](length=nb * WF, fill=0)
    for d in range(nb * WF):
        var q = min(d, dw - 1)
        var i0 = tx.idx[q * 2]
        var i1 = tx.idx[q * 2 + 1]
        off[d] = i0 * ch
        cw0[d] = tx.w[q * 2] + (tx.w[q * 2 + 1] if i1 == i0 else 0)
        cw1[d] = tx.w[q * 2 + 1] if i1 != i0 else 0
    var poff = ptr_of(off)
    var pw0 = ptr_of(cw0)
    var pw1 = ptr_of(cw1)
    var pyi = ptr_of(ty.idx)
    var pyw = ptr_of(ty.w)
    var rn = cols * ch
    comptime NT = 8

    def task(t: Int) {imm}:
        var hb0 = List[Float32](unsafe_uninit_length=nb * WF * ch)
        var hb1 = List[Float32](unsafe_uninit_length=nb * WF * ch)
        var last = List[UInt8](length=rn + 16, fill=0)
        var h0 = ptr_of(hb0)
        var h1 = ptr_of(hb1)
        var id0 = -1
        var id1 = -1
        var plast = Ptr[Scalar[dt]](unsafe_from_address=Int(last.unsafe_ptr()))

        def hrow(y: Int, dst: Ptr[Float32]) {imm}:
            var rp = pa + y * rn
            if y == rows - 1:
                for j in range(rn):
                    plast[unsafe_offset=j] = rp[unsafe_offset=j]
                rp = plast
            for b in range(nb):
                var a = SIMD[DType.int, WF](Int(rp)) + poff.unsafe_load[width=WF](b * WF)
                var w0 = pw0.unsafe_load[width=WF](b * WF)
                var w1 = pw1.unsafe_load[width=WF](b * WF)
                comptime if ch == 3:
                    var wd = _words3(a)
                    comptime for k in range(3):
                        var p = _ab3[k](wd[0], wd[1])
                        dst.unsafe_store((b * 3 + k) * WF, w0 * p[0] + w1 * p[1])
                else:
                    var wd = gather[alignment=1](a, SIMD[DType.bool, WF](fill=True), SIMD[DType.uint32, WF](0))
                    dst.unsafe_store(b * WF, w0 * (wd & 255).cast[F32]() + w1 * ((wd >> 8) & 255).cast[F32]())

        for r in range(dh * t // NT, dh * (t + 1) // NT):
            var y0 = pyi[unsafe_offset=r * 2]
            var y1 = pyi[unsafe_offset=r * 2 + 1]
            if id0 != y0 and id1 != y0:
                if id0 == y1:
                    hrow(y0, h1)
                    id1 = y0
                else:
                    hrow(y0, h0)
                    id0 = y0
            if id0 != y1 and id1 != y1:
                if id0 == y0:
                    hrow(y1, h1)
                    id1 = y1
                else:
                    hrow(y1, h0)
                    id0 = y1
            var a = h0 if id0 == y0 else h1
            var c = h0 if id0 == y1 else h1
            var wy0 = SIMD[F32, WF](pyw[unsafe_offset=r * 2])
            var wy1 = SIMD[F32, WF](pyw[unsafe_offset=r * 2 + 1])
            var dst = po + r * dw * ch
            for b in range(nb):
                var valid = min(WF, dw - b * WF)
                comptime if ch == 3:
                    var res = SIMD[dt, 4 * WF](0)
                    comptime for k in range(3):
                        var v = wy0 * a.unsafe_load[width=WF]((b * 3 + k) * WF) + wy1 * c.unsafe_load[width=WF]((b * 3 + k) * WF)
                        res = res.insert[offset = k * WF](_fin[dt, WF](v))
                    if b * WF + WF + 6 <= dw:
                        (dst + b * WF * 3).unsafe_store(0, res.shuffle[_ilv_mask[3]()]())
                    else:
                        comptime for k in range(3):
                            (dst + b * WF * 3 + k).unsafe_scatter(SIMD[DType.int32, WF](0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45), res.slice[WF, offset = k * WF](), SIMD[DType.int32, WF](0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15).lt(SIMD[DType.int32, WF](valid)))
                else:
                    var v = _fin[dt, WF](wy0 * a.unsafe_load[width=WF](b * WF) + wy1 * c.unsafe_load[width=WF](b * WF))
                    if valid == WF:
                        (dst + b * WF).unsafe_store(0, v)
                    else:
                        for j in range(valid):
                            dst[unsafe_offset=b * WF + j] = v[j]
        _ = (len(hb0), len(hb1), len(last))

    par(task, NT)
    _ = (len(off), len(cw0), len(cw1), len(ty.idx), len(ty.w), len(tx.idx), len(tx.w))
    return o^


def resize[dt: DType](src: Mat[dt], dw_in: Int, dh_in: Int, kind_in: Int) -> Mat[dt]:
    """Column taps first, on contiguous SIMD over the source width into one cached row per output row; then the row
    taps pick (dw) pixels from that row."""
    var dw = dw_in
    var dh = dh_in
    var kind = kind_in
    comptime if dt == DType.uint8:
        if kind == 1 and src.cols == 2 * dw and src.rows == 2 * dh and (src.ch == 1 or src.ch == 3 or src.ch == 4):
            var o = Mat[dt](dh, dw, src.ch, False)
            var pa = src.ptr()
            var po = o.ptr()
            var ch = src.ch

            def half(t: Int) {imm}:
                if ch == 1:
                    _half_rows[dt, 1](pa, po, dw, dh * t // 8, dh * (t + 1) // 8)
                elif ch == 3:
                    _half_rows[dt, 3](pa, po, dw, dh * t // 8, dh * (t + 1) // 8)
                else:
                    _half_rows[dt, 4](pa, po, dw, dh * t // 8, dh * (t + 1) // 8)

            par(half, 8)
            _ = len(src.data)
            return o^
    comptime if dt == DType.uint8:
        if kind == 1 and src.cols >= 2 and src.rows >= 2:
            if src.ch == 3:
                return _resize_lin_u8[dt, 3](src, dw, dh)
            elif src.ch == 1:
                return _resize_lin_u8[dt, 1](src, dw, dh)
    var tx = Taps(src.cols, dw, Float64(src.cols) / Float64(dw), kind)
    var ty = Taps(src.rows, dh, Float64(src.rows) / Float64(dh), kind)
    var ch = src.ch
    var o = Mat[dt](dh, dw, ch, False)
    var pa = src.ptr()
    var po = o.ptr()
    var pix = ptr_of(tx.idx)
    var pwx = ptr_of(tx.w)
    var piy = ptr_of(ty.idx)
    var pwy = ptr_of(ty.w)
    var nx = tx.n
    var ny = ty.n
    var sn = src.cols * ch
    var dn = dw * ch
    # per output element and tap: the index into the cached column-pass row and its weight
    var ex = List[Int32](length=nx * dn, fill=0)
    var ew = List[Float32](length=nx * dn, fill=0)
    for q in range(nx):
        for d in range(dw):
            for k in range(ch):
                ex[q * dn + d * ch + k] = Int32(tx.idx[d * nx + q] * ch + k)
                ew[q * dn + d * ch + k] = tx.w[d * nx + q]
    var pex = ptr_of(ex)
    var pew = ptr_of(ew)
    comptime NT = 8

    def task(t: Int) {imm}:
        var vrow = List[Float32](unsafe_uninit_length=sn)
        var v = ptr_of(vrow)
        for r in range(dh * t // NT, dh * (t + 1) // NT):
            if kind == 0:
                var srow = pa + piy[unsafe_offset=r * ny] * sn
                for d in range(dw):
                    for k in range(ch):
                        po[unsafe_offset=(r * dw + d) * ch + k] = srow[unsafe_offset=pix[unsafe_offset=d] * ch + k]
                continue
            var i = 0
            while i + WF <= sn:
                var acc = SIMD[F32, WF](0)
                for q in range(ny):
                    acc += _ld[dt, WF](pa, piy[unsafe_offset=r * ny + q] * sn + i) * pwy[unsafe_offset=r * ny + q]
                v.unsafe_store(i, acc)
                i += WF
            for j in range(i, sn):
                var acc = Float32(0)
                for q in range(ny):
                    acc += Float32(pa[unsafe_offset=piy[unsafe_offset=r * ny + q] * sn + j]) * pwy[unsafe_offset=r * ny + q]
                v[unsafe_offset=j] = acc
            var dst = po + r * dn
            var e = 0
            while e + WF <= dn:
                var acc = SIMD[F32, WF](0)
                for q in range(nx):
                    acc += v.unsafe_gather(pex.unsafe_load[width=WF](q * dn + e)) * pew.unsafe_load[width=WF](q * dn + e)
                dst.unsafe_store(e, _fin[dt, WF](acc))
                e += WF
            for j in range(e, dn):
                var acc = Float32(0)
                for q in range(nx):
                    acc += v[unsafe_offset=Int(pex[unsafe_offset=q * dn + j])] * pew[unsafe_offset=q * dn + j]
                dst.unsafe_store(j, _fin[dt, 1](SIMD[F32, 1](acc)))
        _ = len(vrow)

    par(task, NT)
    _ = (tx.n, ty.n, len(ex), len(ew))
    return o^


# ---- warpAffine / remap -----------------------------------------------------------------------------------
# cv2 5.0 interpolates at the exact float position (the 1/32 fixed-point tables are gone), so one sampler serves both.


def border_px[dt: DType](src: Mat[dt], y: Int, x: Int, k: Int, mode: Int, bval: Scalar[dt]) -> Scalar[dt]:
    var yy = bidx(y, src.rows, mode)
    var xx = bidx(x, src.cols, mode)
    if yy < 0 or xx < 0:
        return bval
    return src.data[(yy * src.cols + xx) * src.ch + k]


def sample[dt: DType](src: Mat[dt], ps: Ptr[Scalar[dt]], fx: Float64, fy: Float64, kind: Int, mode: Int, bval: Scalar[dt], dst: Ptr[Scalar[dt]]):
    """Write the interpolated pixel at (fx, fy) (kind 0 nearest, 1 linear, 2 cubic) to dst[0 .. ch)."""
    var n = 1 if kind == 0 else (2 if kind == 1 else 4)
    var off = 0 if kind != 2 else 1
    var sx = Int(floor(fx + 0.5)) if kind == 0 else Int(floor(fx))
    var sy = Int(floor(fy + 0.5)) if kind == 0 else Int(floor(fy))
    var wx = SIMD[DType.float64, 4](1, 0, 0, 0)
    var wy = SIMD[DType.float64, 4](1, 0, 0, 0)
    if kind == 1:
        wx = SIMD[DType.float64, 4](1 - (fx - Float64(sx)), fx - Float64(sx), 0, 0)
        wy = SIMD[DType.float64, 4](1 - (fy - Float64(sy)), fy - Float64(sy), 0, 0)
    elif kind == 2:
        wx = cubic_w(fx - Float64(sx))
        wy = cubic_w(fy - Float64(sy))
    var ch = src.ch
    var inside = sx - off >= 0 and sy - off >= 0 and sx - off + n <= src.cols and sy - off + n <= src.rows
    if inside and kind == 1:
        var w00 = wy[0] * wx[0]
        var w01 = wy[0] * wx[1]
        var w10 = wy[1] * wx[0]
        var w11 = wy[1] * wx[1]
        var r0 = (sy * src.cols + sx) * ch
        var r1 = r0 + src.cols * ch
        for k in range(ch):
            var v = w00 * Float64(ps[unsafe_offset=r0 + k]) + w01 * Float64(ps[unsafe_offset=r0 + ch + k]) + w10 * Float64(ps[unsafe_offset=r1 + k]) + w11 * Float64(ps[unsafe_offset=r1 + ch + k])
            comptime if dt.is_integral():
                v = max(Float64(Scalar[dt].MIN), min(Float64(Scalar[dt].MAX), v.__round__()))
            dst[unsafe_offset=k] = Scalar[dt](v)
        return
    for k in range(ch):
        var acc = Float64(0)
        if inside:
            for a in range(n):
                var row = ((sy + a - off) * src.cols + sx - off) * ch + k
                for b in range(n):
                    acc += wy[a] * wx[b] * Float64(ps[unsafe_offset=row + b * ch])
        else:
            for a in range(n):
                for b in range(n):
                    acc += wy[a] * wx[b] * Float64(border_px(src, sy + a - off, sx + b - off, k, mode, bval))
        comptime if dt.is_integral():
            acc = max(Float64(Scalar[dt].MIN), min(Float64(Scalar[dt].MAX), acc.__round__()))
        dst[unsafe_offset=k] = Scalar[dt](acc)


def _ilv_mask[ch: Int]() -> IndexList[4 * WF]:
    var m = IndexList[4 * WF]()
    for i in range(4 * WF):
        m[i] = (i % ch) * WF + i // ch if i < ch * WF else 0
    return m


def _words3(t: SIMD[DType.int, WF]) -> Tuple[SIMD[DType.uint32, WF], SIMD[DType.uint32, WF]]:
    """Two pixels of 3 channels (6 bytes, read as 8) at each of WF byte addresses, as the low and high 32 bits."""
    var m = SIMD[DType.bool, WF // 2](fill=True)
    var z = SIMD[DType.uint64, WF // 2](0)
    var a = gather[alignment=1](t.slice[WF // 2](), m, z)
    var b = gather[alignment=1](t.slice[WF // 2, offset = WF // 2](), m, z)
    var lo = rebind[SIMD[DType.uint32, WF]](a.cast[DType.uint32]().join(b.cast[DType.uint32]()))
    var hi = rebind[SIMD[DType.uint32, WF]]((a >> 32).cast[DType.uint32]().join((b >> 32).cast[DType.uint32]()))
    return (lo, hi)


def _ab3[k: Int](lo: SIMD[DType.uint32, WF], hi: SIMD[DType.uint32, WF]) -> Tuple[SIMD[F32, WF], SIMD[F32, WF]]:
    """Channel k of the left and the right pixel of _words3."""
    comptime if k == 0:
        return ((lo & 255).cast[F32](), (lo >> 24).cast[F32]())
    elif k == 1:
        return (((lo >> 8) & 255).cast[F32](), (hi & 255).cast[F32]())
    else:
        return (((lo >> 16) & 255).cast[F32](), ((hi >> 8) & 255).cast[F32]())


def _lin_u8[dt: DType, ch: Int](ps: Ptr[Scalar[dt]], size: Int, cols: Int, sx: SIMD[DType.int32, WF], sy: SIMD[DType.int32, WF], wx1: SIMD[F32, WF], wy1: SIMD[F32, WF], dst: Ptr[Scalar[dt]], wide: Bool, safe: Bool) -> Bool:
    """uint8 bilinear, ch 1 or 3. ch 3: one 8-byte gather per source row and pixel holds both columns of all three
    channels (4 gathers of 8 lanes per WF pixels, 12 before); ch 1: two 4-byte gathers hold both columns. The
    channels are interleaved with one shuffle and stored as one 4 * WF-byte write when `wide` (the 16 bytes past the
    pixels belong to the next block or are the caller's to spare). False, nothing written, when a wide read could
    pass the end of the source."""
    var a = (sy.cast[DType.int]() * cols + sx.cast[DType.int]()) * ch + Int(ps)
    if not safe and Int(a.reduce_max()) - Int(ps) + (7 if ch == 3 else 3) + cols * ch >= size:
        return False
    var wx0 = 1 - wx1
    var wy0 = 1 - wy1
    var row = SIMD[DType.int, WF](cols * ch)
    comptime if ch == 3:
        var res = SIMD[dt, 4 * WF](0)
        var r0 = _words3(a)
        var r1 = _words3(a + row)
        comptime for k in range(3):
            var p0 = _ab3[k](r0[0], r0[1])
            var p1 = _ab3[k](r1[0], r1[1])
            res = res.insert[offset = k * WF](_fin[dt, WF](wy0 * (wx0 * p0[0] + wx1 * p0[1]) + wy1 * (wx0 * p1[0] + wx1 * p1[1])))
        if wide:
            dst.unsafe_store(0, res.shuffle[_ilv_mask[3]()]())
        else:
            comptime for k in range(3):
                (dst + k).unsafe_scatter(SIMD[DType.int32, WF](0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45), res.slice[WF, offset = k * WF]())
    else:
        var m = SIMD[DType.bool, WF](fill=True)
        var w0 = gather[alignment=1](a, m, SIMD[DType.uint32, WF](0))
        var w1 = gather[alignment=1](a + row, m, SIMD[DType.uint32, WF](0))
        var v = wy0 * (wx0 * (w0 & 255).cast[F32]() + wx1 * ((w0 >> 8) & 255).cast[F32]()) + wy1 * (wx0 * (w1 & 255).cast[F32]() + wx1 * ((w1 >> 8) & 255).cast[F32]())
        dst.unsafe_store(0, _fin[dt, WF](v))
    return True


def _lin_vec[dt: DType, ct: DType](src: Mat[dt], ps: Ptr[Scalar[dt]], fx: SIMD[ct, WF], fy: SIMD[ct, WF], mode: Int, bv: Scalar[dt], dst: Ptr[Scalar[dt]], wide: Bool = False, safe: Bool = False) -> Bool:
    """Bilinear samples at WF positions straight into dst (WF pixels), or bv when all of them lie outside under a
    constant border. False, with dst untouched, when any position needs the general border path. Coordinates in
    float32 (remap: the fraction of a float32 is exact) or float64 (warp). `safe`: the caller
    proved 0 <= x < cols - 1 and 0 <= y < rows - 2 for every lane, so none of the checks run."""
    var ffx = floor(fx)
    var ffy = floor(fy)
    var sx = ffx.cast[DType.int32]()
    var sy = ffy.cast[DType.int32]()
    var bad = sx.lt(0) | sy.lt(0) | (sx + 2).gt(Int32(src.cols)) | (sy + 2).gt(Int32(src.rows))
    if not safe and bad.reduce_or():
        if mode == 2 and (Int(sx.reduce_max()) < -1 or Int(sx.reduce_min()) >= src.cols or Int(sy.reduce_max()) < -1 or Int(sy.reduce_min()) >= src.rows):
            for i in range(WF * src.ch):
                dst[unsafe_offset=i] = bv
            return True
        comptime if dt == DType.uint8:
            # an edge block: clamped coordinates through the vector sampler, then only the lanes that really
            # touch the border are redone one by one
            if (src.ch == 3 or src.ch == 1) and src.cols >= 2 and src.rows >= 3:
                var sxc = sx.clamp(0, Int32(src.cols - 2))
                var syc = sy.clamp(0, Int32(src.rows - 3))
                var ok = False
                if src.ch == 3:
                    ok = _lin_u8[dt, 3](ps, src.rows * src.cols * 3, src.cols, sxc, syc, (fx - ffx).cast[F32](), (fy - ffy).cast[F32](), dst, False, False)
                else:
                    ok = _lin_u8[dt, 1](ps, src.rows * src.cols, src.cols, sxc, syc, (fx - ffx).cast[F32](), (fy - ffy).cast[F32](), dst, False, False)
                if ok:
                    var gone = sx.lt(-1) | sx.ge(Int32(src.cols)) | sy.lt(-1) | sy.ge(Int32(src.rows))
                    var redo = bad | sy.gt(Int32(src.rows - 3))  # sy was clamped to rows - 3 above
                    for i in range(WF):
                        if mode == 2 and gone[i]:
                            for k in range(src.ch):
                                dst[unsafe_offset=i * src.ch + k] = bv
                        elif redo[i]:
                            sample(src, ps, Float64(fx[i]), Float64(fy[i]), 1, mode, bv, dst + i * src.ch)
                    return True
        return False
    var ch = src.ch
    var wx1 = (fx - ffx).cast[F32]()
    var wy1 = (fy - ffy).cast[F32]()
    comptime if dt == DType.uint8:
        if ch == 3:
            if _lin_u8[dt, 3](ps, src.rows * src.cols * ch, src.cols, sx, sy, wx1, wy1, dst, wide, safe):
                return True
        elif ch == 1:
            if _lin_u8[dt, 1](ps, src.rows * src.cols, src.cols, sx, sy, wx1, wy1, dst, wide, safe):
                return True
    var wx0 = 1 - wx1
    var wy0 = 1 - wy1
    var base = (sy * Int32(src.cols) + sx) * Int32(ch)
    var lane = SIMD[DType.int32, WF](0)
    comptime for j in range(WF):
        lane[j] = Int32(j * ch)
    for k in range(ch):
        var b = base + Int32(k)
        var p00 = ps.unsafe_gather(b).cast[F32]()
        var p01 = ps.unsafe_gather(b + Int32(ch)).cast[F32]()
        var p10 = ps.unsafe_gather(b + Int32(src.cols * ch)).cast[F32]()
        var p11 = ps.unsafe_gather(b + Int32(src.cols * ch + ch)).cast[F32]()
        var v = wy0 * (wx0 * p00 + wx1 * p01) + wy1 * (wx0 * p10 + wx1 * p11)
        (dst + k).unsafe_scatter(lane, _fin[dt, WF](v))
    return True


def _span(a: Float64, b: Float64, lo: Float64, hi: Float64, n: Int) -> Tuple[Int, Int]:
    """The x in [0, n) with lo <= a * x + b < hi, one pixel narrower on each side to absorb rounding."""
    if abs(a) < 1e-9:
        return (0, n) if (b >= lo + 1e-6 and b < hi - 1e-6) else (0, 0)
    var t1 = (lo - b) / a
    var t2 = (hi - b) / a
    if a < 0:
        var t = t1
        t1 = t2
        t2 = t
    return (max(Int(ceil(max(t1, -1.0))) + 1, 0), min(Int(floor(min(t2, Float64(n) + 1))) - 1, n))


def warp_affine[dt: DType](src: Mat[dt], m: List[Float64], dw: Int, dh: Int, kind: Int, mode: Int, bval: Float64) -> Mat[dt]:
    """m is the inverse map (dst -> src) as a 2 x 3 matrix."""
    var o = Mat[dt](dh, dw, src.ch, False)
    var pm = ptr_of(m)
    var po = o.ptr()
    var ps = src.ptr()
    var bv = Scalar[dt](bval)

    def row(y: Int) {imm}:
        var x = 0
        var iota = SIMD[DType.float64, WF](0)
        comptime for j in range(WF):
            iota[j] = Float64(j)
        var bx = pm[unsafe_offset=1] * Float64(y) + pm[unsafe_offset=2]
        var by = pm[unsafe_offset=4] * Float64(y) + pm[unsafe_offset=5]
        var xlo = 0
        var xhi = 0
        if kind == 1:
            var sx_ = _span(pm[unsafe_offset=0], bx, 0.0, Float64(src.cols - 1), dw)
            var sy_ = _span(pm[unsafe_offset=3], by, 0.0, Float64(src.rows - 2), dw)
            xlo = max(sx_[0], sy_[0])
            xhi = min(sx_[1], sy_[1])
        while x < dw:
            if kind == 1 and x + WF <= dw:
                var xv = iota + Float64(x)
                if _lin_vec(src, ps, pm[unsafe_offset=0] * xv + bx, pm[unsafe_offset=3] * xv + by, mode, bv, po + (y * dw + x) * src.ch, x + WF + 6 <= dw, x >= xlo and x + WF <= xhi):
                    x += WF
                    continue
            var fx = pm[unsafe_offset=0] * Float64(x) + bx
            var fy = pm[unsafe_offset=3] * Float64(x) + by
            sample(src, ps, fx, fy, kind, mode, bv, po + (y * dw + x) * src.ch)
            x += 1

    def band(b: Int) {imm}:
        for y in range(b * 4, min(b * 4 + 4, dh)):
            row(y)

    par_dyn(band, (dh + 3) // 4)
    _ = len(m)
    return o^


def remap[dt: DType](src: Mat[dt], mx: Mat[F32], my: Mat[F32], kind: Int, mode: Int, bval: Float64) -> Mat[dt]:
    var o = Mat[dt](mx.rows, mx.cols, src.ch, False)
    var po = o.ptr()
    var ps = src.ptr()
    var bv = Scalar[dt](bval)
    var pmx = mx.ptr()
    var pmy = my.ptr()

    def row(y: Int) {imm}:
        var x = 0
        while x < mx.cols:
            var i = y * mx.cols + x
            if kind == 1 and x + WF <= mx.cols:
                if _lin_vec(src, ps, pmx.unsafe_load[width=WF](i), pmy.unsafe_load[width=WF](i), mode, bv, po + i * src.ch, x + WF + 6 <= mx.cols):
                    x += WF
                    continue
            sample(src, ps, Float64(pmx[unsafe_offset=i]), Float64(pmy[unsafe_offset=i]), kind, mode, bv, po + i * src.ch)
            x += 1

    def band(b: Int) {imm}:
        for y in range(b * 4, min(b * 4 + 4, mx.rows)):
            row(y)

    par_dyn(band, (mx.rows + 3) // 4)
    _ = (mx.rows, my.rows)
    return o^


def blob[dt: DType](src: Mat[dt], dw: Int, dh: Int, scale: Float32, mean: List[Float32], swap: Bool) -> Mat[F32]:
    """cv2.dnn.blobFromImage with a plain resize: planes laid out as (channel * h, w), float32."""
    var r = resize(src, dw, dh, 1) if (dw != src.cols or dh != src.rows) else Mat[dt](copy=src)
    var o = Mat[F32](r.ch * dh, dw, 1, False)
    var ch = r.ch
    var pr = r.ptr()
    var po = o.ptr()
    var pm = ptr_of(mean)

    def plane_row(i: Int) {imm}:
        var c = i // dh
        var sc = (ch - 1 - c) if (swap and ch == 3) else c
        var src_row = pr + (i % dh) * dw * ch + sc
        for x in range(dw):
            po.unsafe_store(i * dw + x, (Float32(src_row[unsafe_offset=x * ch]) - pm[unsafe_offset=c]) * scale)

    par(plane_row, ch * dh)
    _ = (len(mean), r.rows)
    return o^


# ---- connected components ---------------------------------------------------------------------------------


def find_root(parent: Ptr[Int32], i: Int32) -> Int32:
    var r = i
    while parent[unsafe_offset=Int(r)] != r:
        r = parent[unsafe_offset=Int(r)]
    return r


def components(src: Mat[DType.uint8], conn: Int) -> Tuple[Mat[DType.int32], Mat[DType.int32], Mat[DType.float64]]:
    """connectedComponentsWithStats: label 0 is the background, the others are numbered as cv2 numbers them (4-connected:
    raster order of each component's first pixel; 8-connected: raster order of its first 2 x 2 block);
    stats rows are left, top, width, height, area; centroids are float64 (x, y)."""
    var rows = src.rows
    var cols = src.cols
    var lab = Mat[DType.int32](rows, cols, 1)
    var parent = List[Int32](length=rows * cols // 2 + 2, fill=0)
    var pp = ptr_of(parent)
    var pl = lab.ptr()
    var ps = src.ptr()
    var count: Int32 = 0
    for y in range(rows):
        for x in range(cols):
            if ps[unsafe_offset=y * cols + x] == 0:
                continue
            var best: Int32 = 0
            comptime for k in range(4):
                var nx = x + (-1 if k == 0 else (-1 if k == 1 else (0 if k == 2 else 1)))
                var ny = y - (0 if k == 0 else 1)
                if (k == 0 or conn == 8 or k == 2) and nx >= 0 and nx < cols and ny >= 0:
                    var l = pl[unsafe_offset=ny * cols + nx]
                    if l != 0:
                        if best == 0:
                            best = find_root(pp, l)
                        else:
                            var r = find_root(pp, l)
                            if r != best:
                                if r < best:
                                    pp[unsafe_offset=Int(best)] = r
                                    best = r
                                else:
                                    pp[unsafe_offset=Int(r)] = best
            if best == 0:
                count += 1
                pp[unsafe_offset=Int(count)] = count
                best = count
            pl[unsafe_offset=y * cols + x] = best
    var remap_ = List[Int32](length=Int(count) + 1, fill=0)
    var n: Int32 = 0
    for i in range(1, Int(count) + 1):
        if pp[unsafe_offset=i] == Int32(i):
            n += 1
            remap_[i] = n
        else:
            remap_[i] = remap_[Int(find_root(pp, Int32(i)))]
    var nl = Int(n) + 1
    if conn == 8:
        # cv2 scans 8-connected images in 2 x 2 blocks, so it numbers components by their first block, not first pixel
        var first = List[Int](length=nl, fill=rows * cols + 1)
        for y in range(rows):
            for x in range(cols):
                var l = Int(pl[unsafe_offset=y * cols + x])
                if l != 0:
                    var f = Int(remap_[l])
                    first[f] = min(first[f], (y >> 1) * cols + (x >> 1))
        var order = List[Int]()
        for i in range(1, nl):
            order.append(first[i] * nl + i)
        sort(order)
        var rank = List[Int32](length=nl, fill=0)
        for i in range(len(order)):
            rank[order[i] % nl] = Int32(i + 1)
        for i in range(1, Int(count) + 1):
            remap_[i] = rank[Int(remap_[i])]
    var stats = Mat[DType.int32](nl, 5, 1)
    var cen = Mat[DType.float64](nl, 2, 1)
    var sx = List[Float64](length=nl, fill=0)
    var sy = List[Float64](length=nl, fill=0)
    for i in range(nl):
        stats.set(i, 0, 0, Int32(cols))
        stats.set(i, 1, 0, Int32(rows))
        stats.set(i, 2, 0, -1)
        stats.set(i, 3, 0, -1)
    for y in range(rows):
        for x in range(cols):
            var l = 0
            if pl[unsafe_offset=y * cols + x] != 0:
                l = Int(remap_[Int(pl[unsafe_offset=y * cols + x])])
                pl[unsafe_offset=y * cols + x] = Int32(l)
            stats.set(l, 0, 0, min(stats.at(l, 0), Int32(x)))
            stats.set(l, 1, 0, min(stats.at(l, 1), Int32(y)))
            stats.set(l, 2, 0, max(stats.at(l, 2), Int32(x)))
            stats.set(l, 3, 0, max(stats.at(l, 3), Int32(y)))
            stats.set(l, 4, 0, stats.at(l, 4) + 1)
            sx[l] += Float64(x)
            sy[l] += Float64(y)
    for l in range(nl):
        var area = Float64(stats.at(l, 4))
        stats.set(l, 2, 0, stats.at(l, 2) - stats.at(l, 0) + 1)
        stats.set(l, 3, 0, stats.at(l, 3) - stats.at(l, 1) + 1)
        cen.set(l, 0, 0, sx[l] / area)
        cen.set(l, 1, 0, sy[l] / area)
    return (lab^, stats^, cen^)


# ---- Canny ------------------------------------------------------------------------------------------------


def canny(src: Mat[DType.uint8], t1: Float64, t2: Float64, aperture: Int, l2: Bool) -> Mat[DType.uint8]:
    """cv2.Canny: Sobel (replicate border), non-maximum suppression with cv2's fixed-point tangent tests
    (tan 22.5 deg = 13573 / 2^15), hysteresis by 8-connected flood from the strong pixels."""
    var dx = sobel[DType.uint8, DType.int16](src, 1, 0, aperture, 1.0, 1)
    var dy = sobel[DType.uint8, DType.int16](src, 0, 1, aperture, 1.0, 1)
    var rows = src.rows
    var cols = src.cols
    var lo = min(t1, t2)
    var hi = max(t1, t2)
    if l2:
        lo = min(32767.0, lo)
        hi = min(32767.0, hi)
        lo = lo * lo if lo > 0 else lo
        hi = hi * hi if hi > 0 else hi
    var low = Int(floor(lo))
    var high = Int(floor(hi))
    var mag = List[Int32](unsafe_uninit_length=rows * cols)
    var pmag = ptr_of(mag)
    var pdx = dx.ptr()
    var pdy = dy.ptr()
    comptime WI = simd_width_of[DType.int32]()

    def mrow(y: Int) {imm}:
        var i = y * cols
        var e = i + cols
        while i + WI <= e:
            var a = pdx.unsafe_load[width=WI](i).cast[DType.int32]()
            var b = pdy.unsafe_load[width=WI](i).cast[DType.int32]()
            pmag.unsafe_store(i, (a * a + b * b) if l2 else (abs(a) + abs(b)))
            i += WI
        while i < e:
            var a = Int32(pdx[unsafe_offset=i])
            var b = Int32(pdy[unsafe_offset=i])
            pmag[unsafe_offset=i] = (a * a + b * b) if l2 else (abs(a) + abs(b))
            i += 1

    par(mrow, rows)
    # 1 not an edge, 0 weak candidate, 3 strong; a one-pixel border of 1s lets the flood skip bounds checks
    var ms = cols + 2
    var map = Mat[DType.uint8](rows + 2, ms, 1, False)
    var pm = map.ptr()
    for x in range(ms):
        pm[unsafe_offset=x] = 1
        pm[unsafe_offset=(rows + 1) * ms + x] = 1
    for y in range(rows + 2):
        pm[unsafe_offset=y * ms] = 1
        pm[unsafe_offset=y * ms + cols + 1] = 1

    def nrow(y: Int) {imm}:
        var x = 0
        while x < cols:
            var i = y * cols + x
            var pi = (y + 1) * ms + x + 1
            if x + WI <= cols and not pmag.unsafe_load[width=WI](i).gt(Int32(low)).reduce_or():
                comptime for j in range(WI):
                    pm[unsafe_offset=pi + j] = 1
                x += WI
                continue
            var m = Int(pmag[unsafe_offset=i])
            var xs = Int(pdx[unsafe_offset=i])
            var ys = Int(pdy[unsafe_offset=i])
            var ok = False
            if m > low:
                var ax = abs(xs)
                var ay = abs(ys) << 15
                var tg22x = ax * 13573
                var e = Int(pmag[unsafe_offset=i - 1]) if x > 0 else 0
                var w = Int(pmag[unsafe_offset=i + 1]) if x + 1 < cols else 0
                if ay < tg22x:
                    ok = m > e and m >= w
                else:
                    var tg67x = tg22x + (ax << 16)
                    var up = Int(pmag[unsafe_offset=i - cols]) if y > 0 else 0
                    var dn = Int(pmag[unsafe_offset=i + cols]) if y + 1 < rows else 0
                    if ay > tg67x:
                        ok = m > up and m >= dn
                    else:
                        var sg = -1 if (xs ^ ys) < 0 else 1
                        var xa = x - sg
                        var xb = x + sg
                        var pa = Int(pmag[unsafe_offset=i - cols - sg]) if (y > 0 and xa >= 0 and xa < cols) else 0
                        var pb = Int(pmag[unsafe_offset=i + cols + sg]) if (y + 1 < rows and xb >= 0 and xb < cols) else 0
                        ok = m > pa and m > pb
            pm[unsafe_offset=pi] = UInt8((3 if m > high else 0) if ok else 1)
            x += 1

    par(nrow, rows)
    var stk = List[Int32](unsafe_uninit_length=rows * cols)  # each pixel is pushed at most once
    var stack = ptr_of(stk)
    var sp = 0
    comptime WB = simd_width_of[DType.uint8]()
    for y in range(rows):
        var row = (y + 1) * ms + 1
        var q = 0
        while q < cols:
            if q + WB <= cols and not pm.unsafe_load[width=WB](row + q).eq(3).reduce_or():
                q += WB
                continue
            for qq in range(q, min(q + WB, cols)):
                if pm[unsafe_offset=row + qq] == 3:
                    pm[unsafe_offset=row + qq] = 2
                    stack[unsafe_offset=sp] = Int32(row + qq)
                    sp += 1
            q += WB
    var neigh: List[Int] = [-ms - 1, -ms, -ms + 1, -1, 1, ms - 1, ms, ms + 1]
    var offs = ptr_of(neigh)
    while sp > 0:
        sp -= 1
        var i = Int(stack[unsafe_offset=sp])
        comptime for k in range(8):
            var j = i + offs[unsafe_offset=k]
            if pm[unsafe_offset=j] == 0 or pm[unsafe_offset=j] == 3:
                pm[unsafe_offset=j] = 2
                stack[unsafe_offset=sp] = Int32(j)
                sp += 1
    var o = Mat[DType.uint8](rows, cols, 1, False)
    var po = o.ptr()
    for y in range(rows):
        var row = (y + 1) * ms + 1
        var q = 0
        while q + WB <= cols:
            po.unsafe_store(y * cols + q, pm.unsafe_load[width=WB](row + q).eq(2).select(SIMD[DType.uint8, WB](255), SIMD[DType.uint8, WB](0)))
            q += WB
        while q < cols:
            po[unsafe_offset=y * cols + q] = 255 if pm[unsafe_offset=row + q] == 2 else 0
            q += 1
    _ = (dx.rows, dy.rows, len(mag), map.rows, len(neigh), len(stk))
    return o^


# ---- findContours -----------------------------------------------------------------------------------------


def find_contours(src: Mat[DType.uint8], mode: Int, simple: Bool) -> Tuple[Mat[DType.int32], Mat[DType.int32], Mat[DType.int32]]:
    """Suzuki and Abe border following as cv2 does it: raster scan, outer and hole borders, points emitted in
    follow order starting at the first pixel found. mode 0 external, 1 list, 2 ccomp (outer borders at the top
    level, holes below their outer border), 3 tree. Returns (points (n, 2) int32, lengths (k, 1) int32, hierarchy
    (k, 4) int32 as next, previous, first child, parent). Contours come out in cv2's tree order: depth first, the
    newest sibling first."""
    var external = mode == 0
    var w = src.cols + 2
    var h = src.rows + 2
    var img = List[Int32](length=w * h, fill=0)
    var pi = ptr_of(img)
    for y in range(src.rows):
        for x in range(src.cols):
            if src.data[y * src.cols + x] != 0:
                pi[unsafe_offset=(y + 1) * w + x + 1] = 1
    var dxs = SIMD[DType.int32, 8](1, 1, 0, -1, -1, -1, 0, 1)
    var dys = SIMD[DType.int32, 8](0, -1, -1, -1, 0, 1, 1, 1)
    var pts = List[Int32]()
    var lens = List[Int32]()
    var ctype: List[Int] = [1, 1]  # by border id (nbd): 1 outer, 0 hole; id 1 is the frame
    var cpar: List[Int] = [0, 1]
    var bpar = List[Int]()  # parent border id of every border found (id - 2 indexes it), as the mode shapes the tree
    var nbd = 1
    for y in range(1, h - 1):
        var lnbd = 1
        for x in range(1, w - 1):
            var f = Int(pi[unsafe_offset=y * w + x])
            var is_outer = f == 1 and pi[unsafe_offset=y * w + x - 1] == 0
            var is_hole = f >= 1 and pi[unsafe_offset=y * w + x + 1] == 0
            if is_outer or is_hole:
                if not is_outer and f > 1:
                    lnbd = f
                var outer = is_outer
                nbd += 1
                var tb = ctype[lnbd]
                var parent = (cpar[lnbd] if tb == 1 else lnbd) if outer else (lnbd if tb == 1 else cpar[lnbd])
                ctype.append(1 if outer else 0)
                cpar.append(parent)
                if mode == 1 or (mode == 2 and outer):
                    bpar.append(1)
                else:
                    bpar.append(parent)
                var keep = (not external) or (outer and parent == 1)
                var start = len(pts)
                var cx = x
                var cy = y
                var s = 4 if outer else 0
                var s_end = s
                var found = False
                while True:
                    s = (s - 1) & 7
                    if pi[unsafe_offset=(cy + Int(dys[s])) * w + cx + Int(dxs[s])] != 0:
                        found = True
                        break
                    if s == s_end:
                        break
                if not found:
                    pi[unsafe_offset=y * w + x] = Int32(-nbd)
                    if keep:
                        pts.append(Int32(x - 1))
                        pts.append(Int32(y - 1))
                else:
                    var i3x = cx
                    var i3y = cy
                    var prev_s = s ^ 4
                    var i1x = cx + Int(dxs[s])
                    var i1y = cy + Int(dys[s])
                    while True:
                        s_end = s
                        var i4x = 0
                        var i4y = 0
                        while True:
                            s += 1
                            i4x = i3x + Int(dxs[s & 7])
                            i4y = i3y + Int(dys[s & 7])
                            if pi[unsafe_offset=i4y * w + i4x] != 0:
                                break
                        s &= 7
                        if UInt(s - 1) < UInt(s_end):
                            pi[unsafe_offset=i3y * w + i3x] = Int32(-nbd)
                        elif pi[unsafe_offset=i3y * w + i3x] == 1:
                            pi[unsafe_offset=i3y * w + i3x] = Int32(nbd)
                        if keep and (s != prev_s or not simple):
                            pts.append(Int32(i3x - 1))
                            pts.append(Int32(i3y - 1))
                            prev_s = s
                        if i4x == cx and i4y == cy and i3x == i1x and i3y == i1y:
                            break
                        i3x = i4x
                        i3y = i4y
                        s = (s + 4) & 7
                if keep:
                    lens.append(Int32((len(pts) - start) // 2))
            f = Int(pi[unsafe_offset=y * w + x])
            if f != 1 and f != 0:
                lnbd = abs(f)
    # cv2 stores contours in a tree where a new border becomes the first child of its parent, then lists the tree
    # depth first: so children come newest first, and a node is followed by its subtree
    var n = len(lens)
    var kept = List[Int]()  # discovery index of every kept contour
    for k in range(len(bpar)):
        if not external or (ctype[k + 2] == 1 and bpar[k] == 1):
            kept.append(k)
    var kidx = List[Int](length=len(bpar), fill=-1)  # discovery index -> kept index
    for i in range(len(kept)):
        kidx[kept[i]] = i
    var kids = List[List[Int]](length=n + 1, fill=List[Int]())  # slot n is the frame
    for i in range(n):
        var par = bpar[kept[i]] - 2
        kids[n if par < 0 else kidx[par]].append(i)
    var order = List[Int]()
    var rank = List[Int](length=n, fill=0)
    var hier = List[Int](length=4 * n, fill=-1)
    # depth first from the frame: pushing in discovery order pops the newest sibling first
    var todo = List[Int]()
    for j in range(len(kids[n])):
        todo.append(kids[n][j])
    while len(todo) > 0:
        var c = todo.pop()
        rank[c] = len(order)
        order.append(c)
        for j in range(len(kids[c])):
            todo.append(kids[c][j])
    for p in range(n + 1):
        var m = len(kids[p])
        for j in range(m):
            var c = kids[p][m - 1 - j]  # j-th newest
            var at = rank[c]
            hier[4 * at + 0] = rank[kids[p][m - 2 - j]] if j + 1 < m else -1
            hier[4 * at + 1] = rank[kids[p][m - j]] if j > 0 else -1
            hier[4 * at + 3] = rank[p] if p < n else -1
            if j == 0 and p < n:
                hier[4 * rank[p] + 2] = at
    var op = Mat[DType.int32](len(pts) // 2, 2, 1)
    var ol = Mat[DType.int32](n, 1, 1)
    var oh = Mat[DType.int32](n, 4, 1)
    var starts = List[Int](length=n, fill=0)
    var acc = 0
    for i in range(n):
        starts[i] = acc
        acc += 2 * Int(lens[i])
    var dst_at = 0
    for r in range(n):
        var c = order[r]
        var k = 2 * Int(lens[c])
        for i in range(k):
            op.data[dst_at + i] = pts[starts[c] + i]
        dst_at += k
        ol.data[r] = lens[c]
        for q in range(4):
            oh.data[4 * r + q] = Int32(hier[4 * r + q])
    return (op^, ol^, oh^)


# ---- goodFeaturesToTrack ----------------------------------------------------------------------------------


def good_features(src: Mat[DType.uint8], max_corners: Int, quality: Float64, min_dist: Float64, block: Int, harris: Bool = False, hk: Float64 = 0.04) -> Mat[F32]:
    """Shi-Tomasi (or Harris det - k trace^2 when `harris`): smallest eigenvalue of the windowed gradient covariance, local maxima above quality * max,
    strongest first (ties by position), thinned by a minimum distance on a grid. Returns (n, 2) float32 (x, y)."""
    var scale = Float32(1.0 / (4.0 * Float64(block) * 255.0))
    var dx = sobel[DType.uint8, F32](src, 1, 0, 3, scale)
    var dy = sobel[DType.uint8, F32](src, 0, 1, 3, scale)
    var rows = src.rows
    var cols = src.cols
    var n = rows * cols
    var xx = Mat[F32](rows, cols, 1, False)
    var xy = Mat[F32](rows, cols, 1, False)
    var yy = Mat[F32](rows, cols, 1, False)
    var pdx = dx.ptr()
    var pdy = dy.ptr()
    var pxx = xx.ptr()
    var pxy = xy.ptr()
    var pyy = yy.ptr()

    def cov_rows(y: Int) {imm}:
        var i = y * cols
        var e = i + cols
        while i + WF <= e:
            var a = pdx.unsafe_load[width=WF](i)
            var b = pdy.unsafe_load[width=WF](i)
            pxx.unsafe_store(i, a * a)
            pxy.unsafe_store(i, a * b)
            pyy.unsafe_store(i, b * b)
            i += WF
        while i < e:
            pxx[unsafe_offset=i] = pdx[unsafe_offset=i] * pdx[unsafe_offset=i]
            pxy[unsafe_offset=i] = pdx[unsafe_offset=i] * pdy[unsafe_offset=i]
            pyy[unsafe_offset=i] = pdy[unsafe_offset=i] * pdy[unsafe_offset=i]
            i += 1

    par(cov_rows, rows)
    var box = List[Float32](length=block, fill=1)
    var sxx = sepfilter[F32, F32](xx, box, box)
    var sxy = sepfilter[F32, F32](xy, box, box)
    var syy = sepfilter[F32, F32](yy, box, box)
    var eig = Mat[F32](rows, cols, 1, False)
    var pe = eig.ptr()
    var p0 = sxx.ptr()
    var p1 = sxy.ptr()
    var p2 = syy.ptr()
    var rowmax = List[Float32](length=rows, fill=0)
    var prm = ptr_of(rowmax)
    var hk32 = Float32(hk)

    def eig_rows(y: Int) {imm}:
        var i = y * cols
        var e = i + cols
        var m = SIMD[F32, WF](0)
        var ms = Float32(0)
        while i + WF <= e:
            var a = p0.unsafe_load[width=WF](i) * 0.5
            var b = p1.unsafe_load[width=WF](i)
            var c = p2.unsafe_load[width=WF](i) * 0.5
            var v = (a + c) - sqrt((a - c) * (a - c) + b * b)
            if harris:
                var sa = a * 2
                var sc = c * 2
                v = sa * sc - b * b - hk32 * (sa + sc) * (sa + sc)
            pe.unsafe_store(i, v)
            m = max(m, v)
            i += WF
        while i < e:
            var a = p0[unsafe_offset=i] * 0.5
            var b = p1[unsafe_offset=i]
            var c = p2[unsafe_offset=i] * 0.5
            var v = (a + c) - sqrt((a - c) * (a - c) + b * b)
            if harris:
                var sa = a * 2
                var sc = c * 2
                v = sa * sc - b * b - hk32 * (sa + sc) * (sa + sc)
            pe[unsafe_offset=i] = v
            ms = max(ms, v)
            i += 1
        prm[unsafe_offset=y] = max(ms, m.reduce_max())

    par(eig_rows, rows)
    var best = Float32(0)
    for y in range(rows):
        best = max(best, rowmax[y])
    var thr = Float32(Float64(best) * quality)
    # a pixel is a candidate when above thr and not below any of its (in-image) 8 neighbours
    var mask = Mat[DType.uint8](rows, cols, 1, False)
    var pk = mask.ptr()

    def mask_rows(y: Int) {imm}:
        var x = 0
        while x < cols:
            var i = y * cols + x
            if x == 0 or x + WF >= cols:
                var v = pe[unsafe_offset=i]
                var ok = v > thr
                for ddy in range(max(y - 1, 0), min(y + 2, rows)):
                    for ddx in range(max(x - 1, 0), min(x + 2, cols)):
                        if pe[unsafe_offset=ddy * cols + ddx] > v:
                            ok = False
                pk[unsafe_offset=i] = 1 if ok else 0
                x += 1
                continue
            var v = pe.unsafe_load[width=WF](i)
            var nb = max(pe.unsafe_load[width=WF](i - 1), pe.unsafe_load[width=WF](i + 1))
            if y > 0:
                nb = max(nb, max(pe.unsafe_load[width=WF](i - cols - 1), max(pe.unsafe_load[width=WF](i - cols), pe.unsafe_load[width=WF](i - cols + 1))))
            if y + 1 < rows:
                nb = max(nb, max(pe.unsafe_load[width=WF](i + cols - 1), max(pe.unsafe_load[width=WF](i + cols), pe.unsafe_load[width=WF](i + cols + 1))))
            var ok = v.gt(thr) & v.ge(nb)
            comptime for j in range(WF):
                pk[unsafe_offset=i + j] = 1 if ok[j] else 0
            x += WF

    par(mask_rows, rows)
    var cand = List[Int]()
    comptime WB = simd_width_of[DType.uint8]()
    var q = 0
    while q < n:
        if q + WB <= n and not pk.unsafe_load[width=WB](q).ne(0).reduce_or():
            q += WB
            continue
        for qq in range(q, min(q + WB, n)):
            if pk[unsafe_offset=qq] != 0:
                cand.append(qq)
        q += WB

    def stronger(p: Int, q: Int) {imm} -> Bool:
        return eig.data[p] > eig.data[q] or (eig.data[p] == eig.data[q] and p < q)

    sort(Span(cand), stronger)
    var cell = max(rint(min_dist), 1)
    var gw = (src.cols + cell - 1) // cell
    var gh = (src.rows + cell - 1) // cell
    var grid = List[List[Int]](length=gw * gh, fill=List[Int]())
    var chosen = List[Int]()
    var md2 = min_dist * min_dist
    for c in cand:
        var y = c // src.cols
        var x = c % src.cols
        var good = True
        if min_dist >= 1:
            var xc = x // cell
            var yc = y // cell
            for yy in range(max(yc - 1, 0), min(yc + 2, gh)):
                for xx in range(max(xc - 1, 0), min(xc + 2, gw)):
                    for m in grid[yy * gw + xx]:
                        var ddx = Float64(x - m % src.cols)
                        var ddy = Float64(y - m // src.cols)
                        if ddx * ddx + ddy * ddy < md2:
                            good = False
            if good:
                grid[yc * gw + xc].append(c)
        if good:
            chosen.append(c)
            if max_corners > 0 and len(chosen) == max_corners:
                break
    var o = Mat[F32](len(chosen), 2, 1)
    for i in range(len(chosen)):
        o.data[2 * i] = Float32(chosen[i] % src.cols)
        o.data[2 * i + 1] = Float32(chosen[i] // src.cols)
    _ = (xx.rows, xy.rows, yy.rows, sxx.rows, sxy.rows, syy.rows, mask.rows, len(rowmax))
    return o^


# ---- drawing ----------------------------------------------------------------------------------------------


def clip_line(cols: Int, rows: Int, mut x1: Int, mut y1: Int, mut x2: Int, mut y2: Int) -> Bool:
    """cv2.clipLine, including its integer truncation, so clipped lines rasterize identically."""
    var right = cols - 1
    var bottom = rows - 1
    var c1 = Int(x1 < 0) + Int(x1 > right) * 2 + Int(y1 < 0) * 4 + Int(y1 > bottom) * 8
    var c2 = Int(x2 < 0) + Int(x2 > right) * 2 + Int(y2 < 0) * 4 + Int(y2 > bottom) * 8
    if (c1 & c2) == 0 and (c1 | c2) != 0:
        if (c1 & 12) != 0:
            var a = 0 if c1 < 8 else bottom
            x1 += Int(Float64(a - y1) * Float64(x2 - x1) / Float64(y2 - y1))
            y1 = a
            c1 = Int(x1 < 0) + Int(x1 > right) * 2
        if (c2 & 12) != 0:
            var a = 0 if c2 < 8 else bottom
            x2 += Int(Float64(a - y2) * Float64(x2 - x1) / Float64(y2 - y1))
            y2 = a
            c2 = Int(x2 < 0) + Int(x2 > right) * 2
        if (c1 & c2) == 0 and (c1 | c2) != 0:
            if c1 != 0:
                var a = 0 if c1 == 1 else right
                y1 += Int(Float64(a - x1) * Float64(y2 - y1) / Float64(x2 - x1))
                x1 = a
                c1 = 0
            if c2 != 0:
                var a = 0 if c2 == 1 else right
                y2 += Int(Float64(a - x2) * Float64(y2 - y1) / Float64(x2 - x1))
                x2 = a
                c2 = 0
    return (c1 | c2) == 0


def draw_line[dt: DType](mut m: Mat[dt], px1: Int, py1: Int, px2: Int, py2: Int, color: List[Float64], thickness: Int = 1, flags: Int = 3):
    """cv2.line, LINE_8: thickness 1 clips, then walks the error term of cv2's LineIterator (left to right);
    thicker lines are a filled rectangle with round caps (`flags` 1 start cap, 2 end cap)."""
    if thickness > 1:
        _thick_line(m, px1, py1, px2, py2, color, thickness, flags)
        return
    var x1 = px1
    var y1 = py1
    var x2 = px2
    var y2 = py2
    if not clip_line(m.cols, m.rows, x1, y1, x2, y2):
        return
    if x2 < x1:
        swap(x1, x2)
        swap(y1, y2)
    var dx = x2 - x1
    var dy = abs(y2 - y1)
    var ystep = 1 if y2 >= y1 else -1
    var steep = dy > dx
    var major = dy if steep else dx
    var minor = dx if steep else dy
    var err = major - 2 * minor
    var x = x1
    var y = y1
    for _ in range(major + 1):
        for k in range(m.ch):
            m.set(y, x, k, Scalar[dt](color[k] if k < len(color) else 0.0))
        if steep:
            y += ystep
        else:
            x += 1
        if err < 0:
            if steep:
                x += 1
            else:
                y += ystep
            err += 2 * major - 2 * minor
        else:
            err -= 2 * minor


comptime XY_SHIFT = 16
comptime XY_ONE = 1 << 16


def _tdiv(a: Int, b: Int) -> Int:
    """C integer division (truncates toward zero)."""
    var q = abs(a) // abs(b)
    return q if (a < 0) == (b < 0) else -q


def _put[dt: DType](mut m: Mat[dt], x: Int, y: Int, color: List[Float64]):
    if x < 0 or y < 0 or x >= m.cols or y >= m.rows:
        return
    for k in range(m.ch):
        m.set(y, x, k, Scalar[dt](color[k] if k < len(color) else 0.0))


def _hline[dt: DType](mut m: Mat[dt], y: Int, x1: Int, x2: Int, color: List[Float64]):
    for x in range(x1, x2 + 1):
        _put(m, x, y, color)


def _line2[dt: DType](mut m: Mat[dt], px1: Int, py1: Int, px2: Int, py2: Int, color: List[Float64]):
    """cv2's Line2: a line between 16.16 fixed-point endpoints, clipped in fixed point, walked with a constant step."""
    var x1 = px1
    var y1 = py1
    var x2 = px2
    var y2 = py2
    if not clip_line(m.cols << XY_SHIFT, m.rows << XY_SHIFT, x1, y1, x2, y2):
        return
    var dx = x2 - x1
    var dy = y2 - y1
    var ax = abs(dx)
    var ay = abs(dy)
    var x_step: Int
    var y_step: Int
    var ecount: Int
    if ax > ay:
        if dx < 0:
            swap(x1, x2)
            swap(y1, y2)
            dy = -dy
        x_step = XY_ONE
        y_step = _tdiv(dy << XY_SHIFT, ax | 1)
        ecount = (x2 - x1) >> XY_SHIFT
    else:
        if dy < 0:
            swap(x1, x2)
            swap(y1, y2)
            dx = -dx
        x_step = _tdiv(dx << XY_SHIFT, ay | 1)
        y_step = XY_ONE
        ecount = (y2 - y1) >> XY_SHIFT
    x1 += XY_ONE >> 1
    y1 += XY_ONE >> 1
    while ecount >= 0:
        _put(m, x1 >> XY_SHIFT, y1 >> XY_SHIFT, color)
        x1 += x_step
        y1 += y_step
        ecount -= 1


def _fill_convex[dt: DType](mut m: Mat[dt], vx: List[Int], vy: List[Int], color: List[Float64]):
    """cv2's FillConvexPoly for 16.16 vertices and LINE_8: outline with Line2, then fill between the two active edges."""
    var npts = len(vx)
    var delta = XY_ONE >> 1
    var imin = 0
    var edges = npts
    var xmin = vx[0]
    var xmax = vx[0]
    var ymin = vy[0]
    var ymax = vy[0]
    var p0x = vx[npts - 1]
    var p0y = vy[npts - 1]
    for i in range(npts):
        if vy[i] < ymin:
            ymin = vy[i]
            imin = i
        ymax = max(ymax, vy[i])
        xmax = max(xmax, vx[i])
        xmin = min(xmin, vx[i])
        _line2(m, p0x, p0y, vx[i], vy[i], color)
        p0x = vx[i]
        p0y = vy[i]
    xmin = (xmin + delta) >> XY_SHIFT
    xmax = (xmax + delta) >> XY_SHIFT
    ymin = (ymin + delta) >> XY_SHIFT
    ymax = (ymax + delta) >> XY_SHIFT
    if npts < 3 or xmax < 0 or ymax < 0 or xmin >= m.cols or ymin >= m.rows:
        return
    ymax = min(ymax, m.rows - 1)
    var eidx: List[Int] = [imin, imin]
    var eye: List[Int] = [ymin, ymin]
    var edi: List[Int] = [1, npts - 1]
    var ex: List[Int] = [-XY_ONE, -XY_ONE]
    var edx: List[Int] = [0, 0]
    var y = ymin
    while True:
        for i in range(2):
            if y >= eye[i]:
                var idx0 = eidx[i]
                var idx = idx0 + edi[i]
                if idx >= npts:
                    idx -= npts
                while edges > 0:
                    edges -= 1
                    var ty = (vy[idx] + delta) >> XY_SHIFT
                    if ty > y:
                        var xs = vx[idx0]
                        var xe = vx[idx]
                        eye[i] = ty
                        edx[i] = _tdiv((xe - xs) * 2 + (ty - y), 2 * (ty - y))
                        ex[i] = xs
                        eidx[i] = idx
                        break
                    idx0 = idx
                    idx += edi[i]
                    if idx >= npts:
                        idx -= npts
                else:
                    edges -= 1
        if edges < 0:
            break
        if y >= 0:
            var left = 0
            var right = 1
            if ex[0] > ex[1]:
                left = 1
                right = 0
            var xx1 = (ex[left] + delta) >> XY_SHIFT
            var xx2 = (ex[right] + delta) >> XY_SHIFT
            if xx2 >= 0 and xx1 < m.cols:
                _hline(m, y, max(xx1, 0), min(xx2, m.cols - 1), color)
        ex[0] += edx[0]
        ex[1] += edx[1]
        y += 1
        if y > ymax:
            break


def _circle_fill[dt: DType](mut m: Mat[dt], cx: Int, cy: Int, radius: Int, color: List[Float64]):
    """cv2's filled Circle: Bresenham octant walk, each step fills two rows."""
    var err = 0
    var dx = radius
    var dy = 0
    var plus = 1
    var minus = (radius << 1) - 1
    while dx >= dy:
        for pass_ in range(2):
            var yy = cy - dy if pass_ == 0 else cy - dx
            var yy2 = cy + dy if pass_ == 0 else cy + dx
            var half = dx if pass_ == 0 else dy
            var xa = max(cx - half, 0)
            var xb = min(cx + half, m.cols - 1)
            if xa <= xb:
                if yy >= 0 and yy < m.rows:
                    _hline(m, yy, xa, xb, color)
                if yy2 >= 0 and yy2 < m.rows:
                    _hline(m, yy2, xa, xb, color)
        dy += 1
        err += plus
        plus += 2
        var mask = 0 if err <= 0 else -1
        err -= minus & mask
        dx += mask
        minus -= mask & 2


def _thick_line[dt: DType](mut m: Mat[dt], x1: Int, y1: Int, x2: Int, y2: Int, color: List[Float64], thickness: Int, flags: Int):
    """cv2's ThickLine for LINE_8: a rectangle around the segment, plus a filled circle at the ends flagged in `flags`."""
    var p0x = x1 << XY_SHIFT
    var p0y = y1 << XY_SHIFT
    var p1x = x2 << XY_SHIFT
    var p1y = y2 << XY_SHIFT
    var dx = Float64(p0x - p1x) / Float64(XY_ONE)
    var dy = Float64(p1y - p0y) / Float64(XY_ONE)
    var r = dx * dx + dy * dy
    var odd = thickness & 1
    var th = thickness << (XY_SHIFT - 1)
    if abs(r) > 2.220446049250313e-16:
        r = (Float64(th) + Float64(odd) * Float64(XY_ONE) * 0.5) / sqrt(r)
        var dpx = rint(dy * r)
        var dpy = rint(dx * r)
        var vx: List[Int] = [p0x + dpx, p0x - dpx, p1x - dpx, p1x + dpx]
        var vy: List[Int] = [p0y + dpy, p0y - dpy, p1y - dpy, p1y + dpy]
        _fill_convex(m, vx, vy, color)
    for i in range(2):
        if (flags & (i + 1)) != 0:
            _circle_fill(m, (p0x + (XY_ONE >> 1)) >> XY_SHIFT, (p0y + (XY_ONE >> 1)) >> XY_SHIFT, (th + (XY_ONE >> 1)) >> XY_SHIFT, color)
        p0x = p1x
        p0y = p1y


def draw_rect[dt: DType](mut m: Mat[dt], x1: Int, y1: Int, x2: Int, y2: Int, color: List[Float64], thick: Int):
    """cv2.rectangle: thickness 1 draws the four edges, a negative thickness fills the clipped box."""
    if thick < 0:
        for y in range(max(min(y1, y2), 0), min(max(y1, y2), m.rows - 1) + 1):
            for x in range(max(min(x1, x2), 0), min(max(x1, x2), m.cols - 1) + 1):
                for k in range(m.ch):
                    m.set(y, x, k, Scalar[dt](color[k] if k < len(color) else 0.0))
        return
    # a closed polyline: every segment with a round cap at its end (cv2 PolyLine)
    draw_line(m, x1, y2, x1, y1, color, thick, 2)
    draw_line(m, x1, y1, x2, y1, color, thick, 2)
    draw_line(m, x2, y1, x2, y2, color, thick, 2)
    draw_line(m, x2, y2, x1, y2, color, thick, 2)


# ---- HOG people detector ----------------------------------------------------------------------------------
# Dalal and Triggs 2005 with the OpenCV defaults: 64 x 128 window, 16 x 16 blocks every 8 pixels, 8 x 8 cells, 9
# unsigned orientation bins, square-root gamma, [-1 0 1] gradients (reflect101), Gaussian block window of sigma 4 ((block w + block h) / 8),
# trilinear voting, L2-Hys block normalisation. The linear SVM (3780 weights + bias) is an input file.


def _atan2_pos[w: Int](y: SIMD[F32, w], x: SIMD[F32, w]) -> SIMD[F32, w]:
    """atan2(y, x) folded into [0, pi) (atan2 + pi when negative), Cephes atanf polynomial on min / max."""
    comptime PI = Float32(3.14159265358979323846)
    var ax = abs(x)
    var ay = abs(y)
    var mx = max(ax, ay)
    var t = min(ax, ay) / mx.eq(0).select(SIMD[F32, w](1), mx)
    var big = t.gt(0.41421357)
    var u = big.select((t - 1) / (t + 1), t)
    var z = u * u
    var a = (((8.05374449538e-2 * z - 1.38776856032e-1) * z + 1.99777106478e-1) * z - 3.33329491539e-1) * z * u + u
    a = big.select(a + PI / 4, a)
    a = ay.gt(ax).select(PI / 2 - a, a)
    a = x.lt(0).select(PI - a, a)
    return y.lt(0).select(PI - a, a)  # atan2(y, x) is -a for y < 0; + pi folds it into [0, pi)


def hog_blocks(src: Mat[DType.uint8]) -> Mat[F32]:
    """Normalised 36-vectors for every 16 x 16 block on the 8-pixel grid: (block rows, block cols, 36); cells inside
    a block are ordered column-major (x outer) like cv2, bins fastest.
    The Gaussian window times the trilinear cell weight is separable, so a block's cell histograms are
    sum_i A[cy][i] * R[row + i][cx] with R[row][cx] = sum_j B[cx][j] * V(row, x0 + j), where V(row, x) is the pixel's
    bin vector (16 lanes, 9 used). Every image row is reduced once to R; blocks then only add 16 rows."""
    var h = src.rows
    var w = src.cols
    var nby = (h - 16) // 8 + 1
    var nbx = (w - 16) // 8 + 1
    var o = Mat[F32](max(nby, 0), max(nbx, 0), 36)
    if nby <= 0 or nbx <= 0:
        return o^
    comptime L = 16
    # A[cy][i] and B[cx][j]: window g times the weight of cell c for position i (j) inside the block
    var wa = List[Float32](length=2 * 16, fill=0)
    for i in range(16):
        var f = (Float32(i) + 0.5) / 8.0 - 0.5
        var c0 = Int(floor(f))
        var wf = f - Float32(c0)
        var d = Float32(i - 8)
        var g = Float32(exp(Float64(-(d * d) / 32.0)))
        for c in range(2):
            if c == c0:
                wa[c * 16 + i] = g * (1 - wf)
            elif c == c0 + 1:
                wa[c * 16 + i] = g * wf
    var oh = List[Float32](length=9 * L, fill=0)
    for b in range(9):
        oh[b * L + b] = 1
    var rbuf = List[Float32](unsafe_uninit_length=h * nbx * 2 * L)
    var pr = ptr_of(rbuf)
    var pa = ptr_of(wa)
    var poh = ptr_of(oh)
    var ps = src.ptr()
    var xlim = min(w, 8 * (nbx - 1) + 16)
    comptime NT = 8
    comptime PI = Float32(3.14159265358979323846)

    def grad_rows(t: Int) {imm}:
        var bb = List[Int32](length=w, fill=0)
        var v0 = List[Float32](length=w, fill=0)
        var v1 = List[Float32](length=w, fill=0)
        var pbb = ptr_of(bb)
        var pv0 = ptr_of(v0)
        var pv1 = ptr_of(v1)
        for y in range(h * t // NT, h * (t + 1) // NT):
            var rc = ps + y * w
            var ra = ps + bidx(y - 1, h, 0) * w
            var rb = ps + bidx(y + 1, h, 0) * w
            var x = 0
            while x < xlim:
                var gx: SIMD[F32, L]
                var gy: SIMD[F32, L]
                if x >= 1 and x + L + 1 <= w:
                    gx = sqrt(rc.unsafe_load[width=L](x + 1).cast[F32]()) - sqrt(rc.unsafe_load[width=L](x - 1).cast[F32]())
                    gy = sqrt(rb.unsafe_load[width=L](x).cast[F32]()) - sqrt(ra.unsafe_load[width=L](x).cast[F32]())
                else:
                    gx = SIMD[F32, L](0)
                    gy = SIMD[F32, L](0)
                    for j in range(min(L, w - x)):
                        gx[j] = sqrt(Float32(rc[unsafe_offset=bidx(x + j + 1, w, 0)])) - sqrt(Float32(rc[unsafe_offset=bidx(x + j - 1, w, 0)]))
                        gy[j] = sqrt(Float32(rb[unsafe_offset=x + j])) - sqrt(Float32(ra[unsafe_offset=x + j]))
                var mag = sqrt(gx * gx + gy * gy)
                var ang = _atan2_pos[L](gy, gx)
                var fb = ang * (9.0 / PI) - 0.5
                var b0f = floor(fb)
                var wb = fb - b0f
                var n = min(L, w - x)
                if n == L:
                    pbb.unsafe_store(x, b0f.cast[DType.int32]())
                    pv0.unsafe_store(x, mag * (1 - wb))
                    pv1.unsafe_store(x, mag * wb)
                else:
                    for j in range(n):
                        pbb[unsafe_offset=x + j] = Int32(b0f[j])
                        pv0[unsafe_offset=x + j] = mag[j] * (1 - wb[j])
                        pv1[unsafe_offset=x + j] = mag[j] * wb[j]
                x += L
            var rowr = pr + y * nbx * 2 * L
            for g in range(nbx + 1):
                var a0 = SIMD[F32, L](0)
                var a1 = SIMD[F32, L](0)
                var b0v = SIMD[F32, L](0)
                var b1v = SIMD[F32, L](0)
                for jx in range(8):
                    var xx = g * 8 + jx
                    var b0 = Int(pbb[unsafe_offset=xx])
                    var b1 = (b0 + 1) % 9
                    b0 = ((b0 % 9) + 9) % 9
                    var contrib = pv0[unsafe_offset=xx] * poh.unsafe_load[width=L](b0 * L) + pv1[unsafe_offset=xx] * poh.unsafe_load[width=L](b1 * L)
                    a0 += pa[unsafe_offset=jx] * contrib
                    a1 += pa[unsafe_offset=16 + jx] * contrib
                    b0v += pa[unsafe_offset=jx + 8] * contrib
                    b1v += pa[unsafe_offset=16 + jx + 8] * contrib
                if g < nbx:
                    rowr.unsafe_store((g * 2) * L, a0)
                    rowr.unsafe_store((g * 2 + 1) * L, a1)
                if g >= 1:
                    rowr.unsafe_store(((g - 1) * 2) * L, rowr.unsafe_load[width=L](((g - 1) * 2) * L) + b0v)
                    rowr.unsafe_store(((g - 1) * 2 + 1) * L, rowr.unsafe_load[width=L](((g - 1) * 2 + 1) * L) + b1v)
        _ = (len(bb), len(v0), len(v1))

    par(grad_rows, NT)
    var po = o.ptr()

    def block_row(r: Int) {imm}:
        var scratch = List[Float32](length=64, fill=0)
        var sp = ptr_of(scratch)
        for bx in range(nbx):
            var c00 = SIMD[F32, L](0)
            var c01 = SIMD[F32, L](0)
            var c10 = SIMD[F32, L](0)
            var c11 = SIMD[F32, L](0)
            for i in range(16):
                var at = ((r * 8 + i) * nbx + bx) * 2 * L
                var r0 = pr.unsafe_load[width=L](at)
                var r1 = pr.unsafe_load[width=L](at + L)
                var a0 = pa[unsafe_offset=i]
                var a1 = pa[unsafe_offset=16 + i]
                c00 += a0 * r0
                c01 += a1 * r0
                c10 += a0 * r1
                c11 += a1 * r1
            # four overlapping 16-lane stores: cell k keeps lanes 0..8 of its own store at offset 9k
            sp.unsafe_store(0, c00)
            sp.unsafe_store(9, c01)
            sp.unsafe_store(18, c10)
            sp.unsafe_store(27, c11)
            var v0 = sp.unsafe_load[width=16](0)
            var v1 = sp.unsafe_load[width=16](16)
            var v2 = sp.unsafe_load[width=16](32)  # lanes 4.. are junk, masked below
            var m2 = SIMD[F32, 16](1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            v2 = v2 * m2
            var sc = 1.0 / (sqrt((v0 * v0 + v1 * v1 + v2 * v2).reduce_add()) + 3.6)
            v0 = min(v0 * sc, 0.2)
            v1 = min(v1 * sc, 0.2)
            v2 = min(v2 * sc, 0.2)
            var sc2 = 1.0 / (sqrt((v0 * v0 + v1 * v1 + v2 * v2).reduce_add()) + 1e-3)
            var dst = po + (r * nbx + bx) * 36
            dst.unsafe_store(0, v0 * sc2)
            dst.unsafe_store(16, v1 * sc2)
            dst.unsafe_store(32, (v2 * sc2).slice[4]())
        _ = len(scratch)

    par(block_row, nby)
    _ = (len(rbuf), len(wa), len(oh))
    return o^


def hog_scores(blocks: Mat[F32], svm: Mat[F32]) -> Mat[F32]:
    """SVM score of every 64 x 128 window on the 8-pixel grid: (window rows, window cols). The 7 blocks of one window
    row are contiguous in `blocks`, so each of the 15 rows is one 252-float dot product against a re-laid-out SVM."""
    var ny = blocks.rows - 14
    var nx = blocks.cols - 6
    var o = Mat[F32](max(ny, 0), max(nx, 0), 1, False)
    if ny <= 0 or nx <= 0:
        return o^
    var po = o.ptr()
    var pbk = blocks.ptr()
    var ps = svm.ptr()
    var nbx = blocks.cols
    var sw = List[Float32](length=15 * 252, fill=0)
    for j in range(15):
        for i in range(7):
            for t in range(36):
                sw[j * 252 + i * 36 + t] = ps[unsafe_offset=(i * 15 + j) * 36 + t]
    var psw = ptr_of(sw)

    def row(wy: Int) {imm}:
        for wx in range(nx):
            var acc16 = SIMD[F32, 16](0)
            var acc4 = SIMD[F32, 4](0)
            for j in range(15):
                var bo = ((wy + j) * nbx + wx) * 36
                var so = j * 252
                comptime for q in range(15):
                    acc16 += psw.unsafe_load[width=16](so + q * 16) * pbk.unsafe_load[width=16](bo + q * 16)
                comptime for q in range(3):
                    acc4 += psw.unsafe_load[width=4](so + 240 + q * 4) * pbk.unsafe_load[width=4](bo + 240 + q * 4)
            po[unsafe_offset=wy * nx + wx] = ps[unsafe_offset=3780] + acc16.reduce_add() + acc4.reduce_add()

    par(row, ny)
    _ = len(sw)
    return o^


def group_rects(r: List[Int], thresh: Int) -> List[Int]:
    """cv2.groupRectangles(eps = 0.2): cluster similar boxes, average each cluster, drop clusters of at most
    `thresh` boxes and small boxes sitting inside a larger supported one. r holds x, y, w, h per box."""
    var n = len(r) // 4
    var out = List[Int]()
    if thresh <= 0 or n == 0:
        for v in r:
            out.append(v)
        return out^
    var parent = List[Int](length=n, fill=0)
    for i in range(n):
        parent[i] = i

    def find(mut p: List[Int], i: Int) -> Int:
        var k = i
        while p[k] != k:
            p[k] = p[p[k]]
            k = p[k]
        return k

    for i in range(n):
        for j in range(i + 1, n):
            var delta = 0.2 * Float64(min(r[4 * i + 2], r[4 * j + 2]) + min(r[4 * i + 3], r[4 * j + 3])) * 0.5
            if (Float64(abs(r[4 * i] - r[4 * j])) <= delta and Float64(abs(r[4 * i + 1] - r[4 * j + 1])) <= delta
                    and Float64(abs(r[4 * i] + r[4 * i + 2] - r[4 * j] - r[4 * j + 2])) <= delta
                    and Float64(abs(r[4 * i + 1] + r[4 * i + 3] - r[4 * j + 1] - r[4 * j + 3])) <= delta):
                var a = find(parent, i)
                var b = find(parent, j)
                if a != b:
                    parent[b] = a
    var cls = List[Int](length=n, fill=-1)
    var nc = 0
    var sums = List[Int]()
    var cnt = List[Int]()
    for i in range(n):
        var root = find(parent, i)
        if cls[root] < 0:
            cls[root] = nc
            nc += 1
            for _ in range(4):
                sums.append(0)
            cnt.append(0)
        var c = cls[root]
        for k in range(4):
            sums[4 * c + k] += r[4 * i + k]
        cnt[c] += 1
    var avg = List[Int](length=4 * nc, fill=0)
    for c in range(nc):
        for k in range(4):
            avg[4 * c + k] = rint(Float64(Float32(sums[4 * c + k]) * (1.0 / Float32(cnt[c]))))
    for i in range(nc):
        if cnt[i] <= thresh:
            continue
        var inside = False
        for j in range(nc):
            if j == i or cnt[j] <= thresh:
                continue
            var dx = Int(Float64(avg[4 * j + 2]) * 0.2 + 0.5)
            var dy = Int(Float64(avg[4 * j + 3]) * 0.2 + 0.5)
            if (avg[4 * i] >= avg[4 * j] - dx and avg[4 * i + 1] >= avg[4 * j + 1] - dy
                    and avg[4 * i] + avg[4 * i + 2] <= avg[4 * j] + avg[4 * j + 2] + dx
                    and avg[4 * i + 1] + avg[4 * i + 3] <= avg[4 * j + 1] + avg[4 * j + 3] + dy
                    and (cnt[j] > max(3, cnt[i]) or cnt[i] < 3)):
                inside = True
        if not inside:
            for k in range(4):
                out.append(avg[4 * i + k])
    return out^


def hog_detect(src: Mat[DType.uint8], svm: Mat[F32], hit: Float32, scale0: Float64, group: Int) -> Mat[DType.int32]:
    """HOGDescriptor.detectMultiScale with winStride 8 and no padding: (n, 4) boxes x, y, w, h."""
    var scales = List[Float64]()
    var sc = Float64(1)
    for _ in range(64):
        scales.append(sc)
        if rint(Float64(src.cols) / sc) < 64 or rint(Float64(src.rows) / sc) < 128 or scale0 <= 1.0:
            break
        sc *= scale0
    var found = List[Int]()
    for s in scales:
        var w = rint(Float64(src.cols) / s)
        var h = rint(Float64(src.rows) / s)
        if w < 64 or h < 128:
            continue
        var lvl = Mat[DType.uint8](copy=src) if (w == src.cols and h == src.rows) else resize(src, w, h, 1)
        var sc_ = hog_scores(hog_blocks(lvl), svm)
        for wy in range(sc_.rows):
            for wx in range(sc_.cols):
                if sc_.at(wy, wx) >= hit:
                    found.append(rint(Float64(wx * 8) * s))
                    found.append(rint(Float64(wy * 8) * s))
                    found.append(rint(64.0 * s))
                    found.append(rint(128.0 * s))
    var g = group_rects(found, group)
    var o = Mat[DType.int32](len(g) // 4, 4, 1)
    for i in range(len(g)):
        o.data[i] = Int32(g[i])
    return o^


# ---- GrabCut ----------------------------------------------------------------------------------------------
# Rother, Kolmogorov, Blake 2004: iterate (assign GMM components, refit two 5-component colour GMMs, min-cut on a
# 8-neighbour contrast graph). The cut is Dinic's algorithm; cv2 uses Boykov-Kolmogorov and a different k-means seed,
# so masks agree up to ties and cluster seeding, not bit for bit.

comptime GC_BGD: UInt8 = 0
comptime GC_FGD: UInt8 = 1
comptime GC_PR_BGD: UInt8 = 2
comptime GC_PR_FGD: UInt8 = 3


struct Gmm(Movable):
    var coef: List[Float64]
    var mean: List[Float64]
    var inv: List[Float64]
    var det: List[Float64]

    def __init__(out self):
        self.coef = List[Float64](length=5, fill=0)
        self.mean = List[Float64](length=15, fill=0)
        self.inv = List[Float64](length=45, fill=0)
        self.det = List[Float64](length=5, fill=0)

    def comp(self, k: Int, c0: Float64, c1: Float64, c2: Float64) -> Float64:
        if self.coef[k] <= 0:
            return 0.0
        var d0 = c0 - self.mean[3 * k]
        var d1 = c1 - self.mean[3 * k + 1]
        var d2 = c2 - self.mean[3 * k + 2]
        var b = 9 * k
        var m = d0 * (self.inv[b] * d0 + self.inv[b + 3] * d1 + self.inv[b + 6] * d2) + d1 * (self.inv[b + 1] * d0 + self.inv[b + 4] * d1 + self.inv[b + 7] * d2) + d2 * (self.inv[b + 2] * d0 + self.inv[b + 5] * d1 + self.inv[b + 8] * d2)
        return self.coef[k] * 1.0 / sqrt(self.det[k]) * exp(-0.5 * m)

    def prob(self, c0: Float64, c1: Float64, c2: Float64) -> Float64:
        var s = Float64(0)
        for k in range(5):
            s += self.comp(k, c0, c1, c2)
        return s

    def which(self, c0: Float64, c1: Float64, c2: Float64) -> Int:
        var best = 0
        var bp = -1.0
        for k in range(5):
            var p = self.comp(k, c0, c1, c2)
            if p > bp:
                bp = p
                best = k
        return best

    def fit(mut self, samples: List[Float64], labels: List[Int]):
        """Refit from samples (r, g, b triples) with component labels 0..4."""
        var n = len(labels)
        var cnt = List[Float64](length=5, fill=0)
        var sm = List[Float64](length=15, fill=0)
        var sp = List[Float64](length=45, fill=0)
        for i in range(n):
            var k = labels[i]
            cnt[k] += 1
            for a in range(3):
                sm[3 * k + a] += samples[3 * i + a]
                for b in range(3):
                    sp[9 * k + 3 * a + b] += samples[3 * i + a] * samples[3 * i + b]
        for k in range(5):
            if cnt[k] == 0:
                self.coef[k] = 0
                continue
            self.coef[k] = cnt[k] / Float64(n)
            var c = List[Float64](length=9, fill=0)
            for a in range(3):
                self.mean[3 * k + a] = sm[3 * k + a] / cnt[k]
            for a in range(3):
                for b in range(3):
                    c[3 * a + b] = sp[9 * k + 3 * a + b] / cnt[k] - self.mean[3 * k + a] * self.mean[3 * k + b]
            var det = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6])
            if det <= 2.220446049250313e-16:
                c[0] += 0.01
                c[4] += 0.01
                c[8] += 0.01
                det = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6]) + c[2] * (c[3] * c[7] - c[4] * c[6])
            self.det[k] = det
            var b = 9 * k
            self.inv[b] = (c[4] * c[8] - c[5] * c[7]) / det
            self.inv[b + 3] = -(c[3] * c[8] - c[5] * c[6]) / det
            self.inv[b + 6] = (c[3] * c[7] - c[4] * c[6]) / det
            self.inv[b + 1] = -(c[1] * c[8] - c[2] * c[7]) / det
            self.inv[b + 4] = (c[0] * c[8] - c[2] * c[6]) / det
            self.inv[b + 7] = -(c[0] * c[7] - c[1] * c[6]) / det
            self.inv[b + 2] = (c[1] * c[5] - c[2] * c[4]) / det
            self.inv[b + 5] = -(c[0] * c[5] - c[2] * c[3]) / det
            self.inv[b + 8] = (c[0] * c[4] - c[1] * c[3]) / det


def kmeans5(samples: List[Float64]) -> List[Int]:
    """k-means++ seeding then ten Lloyd steps, five clusters on 3-D points."""
    var n = len(samples) // 3
    var labels = List[Int](length=n, fill=0)
    if n < 5:
        return labels^
    var state = UInt64(0x2545F4914F6CDD1D)
    var cen = List[Float64]()
    var first = Int(state % UInt64(n))
    for a in range(3):
        cen.append(samples[3 * first + a])
    var d2 = List[Float64](length=n, fill=1e300)
    for c in range(1, 5):
        var tot = Float64(0)
        for i in range(n):
            var d = Float64(0)
            for a in range(3):
                d += (samples[3 * i + a] - cen[3 * (c - 1) + a]) ** 2
            d2[i] = min(d2[i], d)
            tot += d2[i]
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        var target = Float64(state % 1000003) / 1000003.0 * tot
        var pick = 0
        var acc = Float64(0)
        for i in range(n):
            acc += d2[i]
            if acc >= target:
                pick = i
                break
        for a in range(3):
            cen.append(samples[3 * pick + a])
    for _ in range(10):
        var sums = List[Float64](length=15, fill=0)
        var cnt = List[Int](length=5, fill=0)
        for i in range(n):
            var best = 0
            var bd = 1e300
            for c in range(5):
                var d = Float64(0)
                for a in range(3):
                    d += (samples[3 * i + a] - cen[3 * c + a]) ** 2
                if d < bd:
                    bd = d
                    best = c
            labels[i] = best
            cnt[best] += 1
            for a in range(3):
                sums[3 * best + a] += samples[3 * i + a]
        for c in range(5):
            if cnt[c] > 0:
                for a in range(3):
                    cen[3 * c + a] = sums[3 * c + a] / Float64(cnt[c])
    return labels^


struct Flow(Movable):
    """Dinic max-flow on arrays; node 0 is the source, node 1 the sink."""

    var head: List[Int32]
    var nxt: List[Int32]
    var to: List[Int32]
    var cap: List[Float64]

    def __init__(out self, nodes: Int):
        self.head = List[Int32](length=nodes, fill=-1)
        self.nxt = List[Int32]()
        self.to = List[Int32]()
        self.cap = List[Float64]()

    def add(mut self, a: Int, b: Int, c_ab: Float64, c_ba: Float64):
        self.to.append(Int32(b))
        self.cap.append(c_ab)
        self.nxt.append(self.head[a])
        self.head[a] = Int32(len(self.to) - 1)
        self.to.append(Int32(a))
        self.cap.append(c_ba)
        self.nxt.append(self.head[b])
        self.head[b] = Int32(len(self.to) - 1)

    def maxflow(mut self, s: Int, t: Int) -> Float64:
        var n = len(self.head)
        var total = Float64(0)
        var level = List[Int32](length=n, fill=0)
        var it = List[Int32](length=n, fill=0)
        var queue = List[Int32](length=n, fill=0)
        while True:
            for i in range(n):
                level[i] = -1
            level[s] = 0
            var qh = 0
            var qt = 1
            queue[0] = Int32(s)
            while qh < qt:
                var u = Int(queue[qh])
                qh += 1
                var e = self.head[u]
                while e >= 0:
                    var v = Int(self.to[Int(e)])
                    if self.cap[Int(e)] > 1e-12 and level[v] < 0:
                        level[v] = level[u] + 1
                        queue[qt] = Int32(v)
                        qt += 1
                    e = self.nxt[Int(e)]
            if level[t] < 0:
                break
            for i in range(n):
                it[i] = self.head[i]
            # iterative blocking-flow DFS
            var path = List[Int32]()
            var u = s
            while True:
                if u == t:
                    var f = Float64(1e300)
                    for e in path:
                        f = min(f, self.cap[Int(e)])
                    for e in path:
                        self.cap[Int(e)] -= f
                        self.cap[Int(e) ^ 1] += f
                    total += f
                    # restart from the first saturated edge
                    var cut = 0
                    for i in range(len(path)):
                        if self.cap[Int(path[i])] <= 1e-12:
                            cut = i
                            break
                    u = s if cut == 0 else Int(self.to[Int(path[cut - 1])])
                    var np_ = List[Int32]()
                    for i in range(cut):
                        np_.append(path[i])
                    path = np_^
                    continue
                var advanced = False
                while it[u] >= 0:
                    var e = Int(it[u])
                    var v = Int(self.to[e])
                    if self.cap[e] > 1e-12 and level[v] == level[u] + 1:
                        path.append(Int32(e))
                        u = v
                        advanced = True
                        break
                    it[u] = self.nxt[e]
                if not advanced:
                    if u == s:
                        break
                    level[u] = -1
                    var last = Int(path.pop())
                    u = Int(self.to[last ^ 1])
        return total


def grab_cut(img: Mat[DType.uint8], rx: Int, ry: Int, rw: Int, rh: Int, iters: Int) -> Mat[DType.uint8]:
    """GC_INIT_WITH_RECT on a 3-channel image: returns the binary foreground mask (255 = foreground)."""
    var h = img.rows
    var w = img.cols
    var mask = Mat[DType.uint8](h, w, 1)
    for y in range(h):
        for x in range(w):
            var inside = x >= rx and x < rx + rw and y >= ry and y < ry + rh
            mask.set(y, x, 0, GC_PR_FGD if inside else GC_BGD)
    # contrast terms
    var beta = Float64(0)
    var cntn = 0
    for y in range(h):
        for x in range(w):
            for d in range(4):
                var nx = x + (-1 if d != 2 else 0) + (2 if d == 3 else 0)
                var ny = y - (0 if d == 0 else 1)
                if d == 3:
                    nx = x + 1
                if nx < 0 or nx >= w or ny < 0:
                    continue
                var dd = Float64(0)
                for c in range(3):
                    var df = Float64(img.at(y, x, c)) - Float64(img.at(ny, nx, c))
                    dd += df * df
                beta += dd
                cntn += 1
    beta = 1.0 / (2 * beta / Float64(max(cntn, 1))) if beta > 2.220446049250313e-16 else 0.0
    var gamma = 50.0
    var lam = 9 * gamma
    var nw = List[Float64](length=4 * h * w, fill=0)
    for y in range(h):
        for x in range(w):
            for d in range(4):
                var nx = x + (-1 if d != 2 else 0) + (2 if d == 3 else 0)
                var ny = y - (0 if d == 0 else 1)
                if d == 3:
                    nx = x + 1
                if nx < 0 or nx >= w or ny < 0:
                    continue
                var dd = Float64(0)
                for c in range(3):
                    var df = Float64(img.at(y, x, c)) - Float64(img.at(ny, nx, c))
                    dd += df * df
                var wgt = gamma * exp(-beta * dd)
                nw[4 * (y * w + x) + d] = wgt if (d == 0 or d == 2) else wgt / 1.4142135623730951
    var bg = Gmm()
    var fg = Gmm()
    var bs = List[Float64]()
    var fs = List[Float64]()
    for y in range(h):
        for x in range(w):
            var m = mask.at(y, x)
            var tgt = fs.copy() if False else List[Float64]()
            if m == GC_BGD or m == GC_PR_BGD:
                for c in range(3):
                    bs.append(Float64(img.at(y, x, c)))
            else:
                for c in range(3):
                    fs.append(Float64(img.at(y, x, c)))
    bg.fit(bs, kmeans5(bs))
    fg.fit(fs, kmeans5(fs))
    for _ in range(iters):
        var bl = List[Int]()
        var fl_ = List[Int]()
        var bs2 = List[Float64]()
        var fs2 = List[Float64]()
        for y in range(h):
            for x in range(w):
                var c0 = Float64(img.at(y, x, 0))
                var c1 = Float64(img.at(y, x, 1))
                var c2 = Float64(img.at(y, x, 2))
                var m = mask.at(y, x)
                if m == GC_BGD or m == GC_PR_BGD:
                    bl.append(bg.which(c0, c1, c2))
                    bs2.append(c0)
                    bs2.append(c1)
                    bs2.append(c2)
                else:
                    fl_.append(fg.which(c0, c1, c2))
                    fs2.append(c0)
                    fs2.append(c1)
                    fs2.append(c2)
        bg.fit(bs2, bl)
        fg.fit(fs2, fl_)
        var g = Flow(h * w + 2)
        for y in range(h):
            for x in range(w):
                var node = 2 + y * w + x
                var m = mask.at(y, x)
                var from_s = Float64(0)
                var to_t = Float64(0)
                if m == GC_PR_BGD or m == GC_PR_FGD:
                    var c0 = Float64(img.at(y, x, 0))
                    var c1 = Float64(img.at(y, x, 1))
                    var c2 = Float64(img.at(y, x, 2))
                    from_s = -log_(max(bg.prob(c0, c1, c2), 1e-300))
                    to_t = -log_(max(fg.prob(c0, c1, c2), 1e-300))
                elif m == GC_BGD:
                    to_t = lam
                else:
                    from_s = lam
                g.add(0, node, from_s, 0.0)
                g.add(node, 1, to_t, 0.0)
                for d in range(4):
                    var wgt = nw[4 * (y * w + x) + d]
                    if wgt > 0:
                        var nx = x + (-1 if d != 2 else 0) + (2 if d == 3 else 0)
                        var ny = y - (0 if d == 0 else 1)
                        if d == 3:
                            nx = x + 1
                        g.add(node, 2 + ny * w + nx, wgt, wgt)
        _ = g.maxflow(0, 1)
        # source side = reachable in the residual graph
        var seen = List[Bool](length=h * w + 2, fill=False)
        var stack = List[Int]()
        stack.append(0)
        seen[0] = True
        while len(stack) > 0:
            var u = stack.pop()
            var e = g.head[u]
            while e >= 0:
                var v = Int(g.to[Int(e)])
                if g.cap[Int(e)] > 1e-12 and not seen[v]:
                    seen[v] = True
                    stack.append(v)
                e = g.nxt[Int(e)]
        for y in range(h):
            for x in range(w):
                var m = mask.at(y, x)
                if m == GC_PR_BGD or m == GC_PR_FGD:
                    mask.set(y, x, 0, GC_PR_FGD if seen[2 + y * w + x] else GC_PR_BGD)
    var o = Mat[DType.uint8](h, w, 1)
    for i in range(h * w):
        o.data[i] = 255 if (mask.data[i] & 1) == 1 else 0
    return o^


# ---- CLI --------------------------------------------------------------------------------------------------


def run_dt[dt: DType](mut a: Args) raises -> Bool:
    var m = load[dt](a.ins[0])
    if a.op == "GaussianBlur":
        var o = Mat[dt](0, 0)
        while a.again():
            o = gaussian_blur(m, a.i("ksize", 5), a.f("sigma", 0.0))
        save(o, a.out)
    elif a.op == "pyrDown":
        var o = Mat[dt](0, 0)
        while a.again():
            o = pyr_down(m)
        save(o, a.out)
    elif a.op == "Sobel":
        comptime if dt == DType.uint8:
            var o = Mat[DType.int16](0, 0)
            while a.again():
                o = sobel[dt, DType.int16](m, a.i("dx", 1), a.i("dy", 0), a.i("ksize", 3), Float32(a.f("scale", 1.0)))
            save(o, a.out)
        else:
            var o = Mat[dt](0, 0)
            while a.again():
                o = sobel[dt, dt](m, a.i("dx", 1), a.i("dy", 0), a.i("ksize", 3), Float32(a.f("scale", 1.0)))
            save(o, a.out)
    elif a.op == "resize":
        var o = Mat[dt](0, 0)
        var kind = a.i("interp", 1)
        while a.again():
            o = resize(m, a.i("w", m.cols), a.i("h", m.rows), kind)
        save(o, a.out)
    elif a.op == "warpAffine":
        var f = a.fl("M", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        var inv = f.copy()
        if a.i("inv", 0) == 0:
            var d = f[0] * f[4] - f[1] * f[3]
            d = 1.0 / d if d != 0 else 0.0
            inv = [f[4] * d, -f[1] * d, 0.0, -f[3] * d, f[0] * d, 0.0]
            inv[2] = -inv[0] * f[2] - inv[1] * f[5]
            inv[5] = -inv[3] * f[2] - inv[4] * f[5]
        var o = Mat[dt](0, 0)
        while a.again():
            o = warp_affine(m, inv, a.i("w", m.cols), a.i("h", m.rows), a.i("interp", 1), a.i("border", 2), a.f("bval", 0.0))
        save(o, a.out)
    elif a.op == "remap":
        var mx = load[F32](a.ins[1])
        var my = load[F32](a.ins[2])
        var o = Mat[dt](0, 0)
        while a.again():
            o = remap(m, mx, my, a.i("interp", 1), a.i("border", 2), a.f("bval", 0.0))
        save(o, a.out)
    elif a.op == "blobFromImage":
        var mean = a.fl("mean", [0.0, 0.0, 0.0])
        var mf = List[Float32]()
        for v in mean:
            mf.append(Float32(v))
        var o = Mat[F32](0, 0)
        while a.again():
            o = blob(m, a.i("w", m.cols), a.i("h", m.rows), Float32(a.f("scale", 1.0)), mf, a.i("swap", 0) == 1)
        save(o, a.out)
    elif a.op == "threshold":
        var th = a.f("thresh", 127.0)
        var o = Mat[dt](0, 0)
        comptime if dt == DType.uint8:
            if a.i("otsu", 0) == 1:
                th = Float64(otsu(m))
            elif a.i("tri", 0) == 1:
                th = Float64(triangle(m))
        while a.again():
            o = threshold(m, th, a.f("maxval", 255.0), a.i("type", 0))
        save(o, a.out)
        var t = Mat[DType.float64](1, 1)
        t.set(0, 0, 0, Float64(floor(th)) if dt.is_integral() else th)
        save(t, a.out + ".1")
    elif a.op == "integral":
        comptime if dt == DType.uint8:
            var o = Mat[DType.int32](0, 0)
            while a.again():
                o = integral[dt, DType.int32](m)
            save(o, a.out)
        else:
            var o = Mat[DType.float64](0, 0)
            while a.again():
                o = integral[dt, DType.float64](m)
            save(o, a.out)
    elif a.op == "erode" or a.op == "dilate":
        var o = Mat[dt](0, 0)
        while a.again():
            if a.op == "erode":
                o = morph[True, dt](m, a.i("shape", 0), a.i("ksize", 3), a.i("iters", 1))
            else:
                o = morph[False, dt](m, a.i("shape", 0), a.i("ksize", 3), a.i("iters", 1))
        save(o, a.out)
    elif a.op == "cvtColor":
        var code = cvt_code(a.s("code", ""))
        var o = Mat[dt](0, 0)
        while a.again():
            o = cvt_color(m, code)
        save(o, a.out)
    else:
        return False
    return True


def run_u8(mut a: Args) raises -> Bool:
    var m = load[DType.uint8](a.ins[0])
    if a.op == "connectedComponentsWithStats":
        var o = (Mat[DType.int32](0, 0), Mat[DType.int32](0, 0), Mat[DType.float64](0, 0))
        while a.again():
            o = components(m, a.i("connectivity", 8))
        save(o[0], a.out)
        save(o[1], a.out + ".1")
        save(o[2], a.out + ".2")
    elif a.op == "Canny":
        var o = Mat[DType.uint8](0, 0)
        while a.again():
            o = canny(m, a.f("t1", 100.0), a.f("t2", 200.0), a.i("aperture", 3), a.i("l2", 0) == 1)
        save(o, a.out)
    elif a.op == "findContours":
        var o = (Mat[DType.int32](0, 0), Mat[DType.int32](0, 0), Mat[DType.int32](0, 0))
        while a.again():
            o = find_contours(m, a.i("mode", 0), a.i("method", 0) == 1)
        save(o[0], a.out)
        save(o[1], a.out + ".1")
        save(o[2], a.out + ".2")
    elif a.op == "goodFeaturesToTrack":
        var o = Mat[F32](0, 0)
        while a.again():
            o = good_features(m, a.i("maxCorners", 100), a.f("quality", 0.01), a.f("minDist", 10.0), a.i("block", 3), a.i("harris", 0) == 1, a.f("k", 0.04))
        save(o, a.out)
    elif a.op == "grabCut":
        var r = a.fl("rect", [0.0, 0.0, Float64(m.cols), Float64(m.rows)])
        var o = Mat[DType.uint8](0, 0)
        while a.again():
            o = grab_cut(m, Int(r[0]), Int(r[1]), Int(r[2]), Int(r[3]), a.i("iters", 5))
        save(o, a.out)
    elif a.op == "HOGScores":
        var svm = load[F32](a.ins[1])
        var o = Mat[F32](0, 0)
        while a.again():
            o = hog_scores(hog_blocks(m), svm)
        save(o, a.out)
    elif a.op == "detectMultiScale":
        var svm = load[F32](a.ins[1])
        var o = Mat[DType.int32](0, 0)
        while a.again():
            o = hog_detect(m, svm, Float32(a.f("hit", 0.0)), a.f("scale", 1.05), a.i("group", 2))
        save(o, a.out)
    elif a.op == "line" or a.op == "rectangle":
        var c = a.fl("color", [255.0, 255.0, 255.0])
        var o = Mat[DType.uint8](copy=m)
        while a.again():
            o = Mat[DType.uint8](copy=m)
            if a.op == "line":
                draw_line(o, a.i("x1", 0), a.i("y1", 0), a.i("x2", 0), a.i("y2", 0), c, a.i("thick", 1))
            else:
                draw_rect(o, a.i("x1", 0), a.i("y1", 0), a.i("x2", 0), a.i("y2", 0), c, a.i("thick", 1))
        save(o, a.out)
    else:
        return False
    return True


def run(mut a: Args) raises -> Bool:
    if a.op == "groupRectangles":
        var m = load[DType.int32](a.ins[0])
        var rl = List[Int]()
        for v in m.data:
            rl.append(Int(v))
        var g = List[Int]()
        while a.again():
            g = group_rects(rl, a.i("thresh", 2))
        var o = Mat[DType.int32](len(g) // 4, 4, 1)
        for i in range(len(g)):
            o.data[i] = Int32(g[i])
        save(o, a.out)
        return True
    if a.op in ["connectedComponentsWithStats", "Canny", "findContours", "goodFeaturesToTrack", "line", "rectangle", "HOGScores", "detectMultiScale", "grabCut"]:
        return run_u8(a)
    if not (a.op in ["GaussianBlur", "pyrDown", "Sobel", "threshold", "integral", "erode", "dilate", "cvtColor", "resize", "warpAffine", "remap", "blobFromImage"]):
        return False
    var code = peek_code(a.ins[0])
    comptime for i in range(2):
        if code == i:
            return run_dt[DTS[i]](a)
    return False
