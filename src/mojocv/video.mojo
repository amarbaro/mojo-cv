# video: sub-pixel corner refinement and sparse pyramidal Lucas-Kanade optical flow (Bouguet 2000) with cv2's
# fixed-point window arithmetic, and dense Farneback flow.
from std.sys.info import simd_width_of
from std.math import sqrt, exp, floor
from mojocv.mat import Mat, load, save
from mojocv.core import Args, par
from mojocv.imgproc import pyr_down, pad, ptr_of, rint, bidx, sepfilter, resize, gaussian_blur, Ptr, F32


def sub_pix(src: Mat[DType.uint8], cx: Float32, cy: Float32, w: Int, h: Int, mut out: List[Float32]):
    """cv2.getRectSubPix on an 8-bit image: a w x h patch centred on (cx, cy), bilinear, replicated border."""
    var x0 = cx - Float32(w - 1) * 0.5
    var y0 = cy - Float32(h - 1) * 0.5
    var ix = Int(floor(x0))
    var iy = Int(floor(y0))
    var a = x0 - Float32(ix)
    var b = y0 - Float32(iy)
    for y in range(h):
        for x in range(w):
            var xa = min(max(ix + x, 0), src.cols - 1)
            var xb = min(max(ix + x + 1, 0), src.cols - 1)
            var ya = min(max(iy + y, 0), src.rows - 1)
            var yb = min(max(iy + y + 1, 0), src.rows - 1)
            var s00 = Float32(src.at(ya, xa))
            var s01 = Float32(src.at(ya, xb))
            var s10 = Float32(src.at(yb, xa))
            var s11 = Float32(src.at(yb, xb))
            out[y * w + x] = (s00 * (1 - a) + s01 * a) * (1 - b) + (s10 * (1 - a) + s11 * a) * b


def corner_sub_pix(src: Mat[DType.uint8], pts: Mat[F32], win: Int, zero: Int, max_iter: Int, eps_in: Float64) -> Mat[F32]:
    """cv2.cornerSubPix: iterate the point to where the weighted gradient field is orthogonal to the offsets."""
    var o = Mat[F32](pts.rows, 2, 1)
    var wn = win * 2 + 1
    var mask = List[Float32](length=wn * wn, fill=0)
    for i in range(wn):
        var y = Float32(i - win) / Float32(win)
        var vy = Float32(exp(-y * y))
        for j in range(wn):
            var x = Float32(j - win) / Float32(win)
            mask[i * wn + j] = Float32(Float64(vy) * exp(Float64(-x * x)))
    if zero >= 0 and zero * 2 + 1 < wn:
        for i in range(win - zero, win + zero + 1):
            for j in range(win - zero, win + zero + 1):
                mask[i * wn + j] = 0
    var eps2 = eps_in * eps_in
    var iters = min(max(max_iter, 1), 100)
    var buf = List[Float32](length=(wn + 2) * (wn + 2), fill=0)
    for p in range(pts.rows):
        var tx = pts.at(p, 0)
        var ty = pts.at(p, 1)
        var cx = tx
        var cy = ty
        var it = 0
        var err = Float64(0)
        while True:
            sub_pix(src, cx, cy, wn + 2, wn + 2, buf)
            var a = Float64(0)
            var b = Float64(0)
            var c = Float64(0)
            var bb1 = Float64(0)
            var bb2 = Float64(0)
            for i in range(wn):
                var py = Float64(i - win)
                for j in range(wn):
                    var m = Float64(mask[i * wn + j])
                    var at = (i + 1) * (wn + 2) + j + 1
                    var tgx = Float64(buf[at + 1] - buf[at - 1])
                    var tgy = Float64(buf[at + wn + 2] - buf[at - wn - 2])
                    var gxx = tgx * tgx * m
                    var gxy = tgx * tgy * m
                    var gyy = tgy * tgy * m
                    var px = Float64(j - win)
                    a += gxx
                    b += gxy
                    c += gyy
                    bb1 += gxx * px + gxy * py
                    bb2 += gxy * px + gyy * py
            var det = a * c - b * b
            if abs(det) <= 2.220446049250313e-16 * 2.220446049250313e-16:
                break
            var scale = 1.0 / det
            var nx = Float32(Float64(cx) + c * scale * bb1 - b * scale * bb2)
            var ny = Float32(Float64(cy) - b * scale * bb1 + a * scale * bb2)
            err = Float64((nx - cx) * (nx - cx) + (ny - cy) * (ny - cy))
            cx = nx
            cy = ny
            if cx < 0 or cx >= Float32(src.cols) or cy < 0 or cy >= Float32(src.rows):
                break
            it += 1
            if not (it < iters and err > eps2):
                break
        if abs(cx - tx) > Float32(win) or abs(cy - ty) > Float32(win):
            cx = tx
            cy = ty
        o.set(p, 0, 0, cx)
        o.set(p, 1, 0, cy)
    return o^


# ---- Lucas-Kanade -----------------------------------------------------------------------------------------


struct Level(Movable):
    """One pyramid level: image and Scharr derivatives, both with a border of `b` pixels (reflect101)."""

    var img: Mat[DType.uint8]
    var dx: Mat[DType.int16]
    var dy: Mat[DType.int16]
    var rows: Int
    var cols: Int

    def __init__(out self, base: Mat[DType.uint8], border: Int):
        self.rows = base.rows
        self.cols = base.cols
        self.img = pad(base, border, border)
        self.dx = Mat[DType.int16](self.img.rows, self.img.cols, 1)
        self.dy = Mat[DType.int16](self.img.rows, self.img.cols, 1)
        var r = base.rows
        var c = base.cols
        # Scharr derivatives of the unpadded image (reflect101); the border of the derivative images stays zero, as in cv2
        for y in range(r):
            for x in range(c):
                var ya = bidx(y - 1, r, 0)
                var yb = bidx(y + 1, r, 0)
                var xa = bidx(x - 1, c, 0)
                var xb = bidx(x + 1, c, 0)
                var vxa = Int(base.at(ya, xa)) * 3 + Int(base.at(y, xa)) * 10 + Int(base.at(yb, xa)) * 3
                var vxb = Int(base.at(ya, xb)) * 3 + Int(base.at(y, xb)) * 10 + Int(base.at(yb, xb)) * 3
                self.dx.set(y + border, x + border, 0, Int16(vxb - vxa))
                var vya = Int(base.at(ya, xa)) * 3 + Int(base.at(ya, x)) * 10 + Int(base.at(ya, xb)) * 3
                var vyb = Int(base.at(yb, xa)) * 3 + Int(base.at(yb, x)) * 10 + Int(base.at(yb, xb)) * 3
                self.dy.set(y + border, x + border, 0, Int16(vyb - vya))


def descale(x: Int, n: Int) -> Int:
    return (x + (1 << (n - 1))) >> n


def lk_flow(prev: Mat[DType.uint8], nxt: Mat[DType.uint8], pts: Mat[F32], win: Int, max_level: Int, max_count: Int, eps: Float32) -> Tuple[Mat[F32], Mat[DType.uint8], Mat[F32]]:
    """cv2.calcOpticalFlowPyrLK: new positions (n, 2), status (n, 1), error (n, 1)."""
    var border = win + 1
    var pa = List[Level]()
    var pb = List[Level]()
    var ia = Mat[DType.uint8](copy=prev)
    var ib = Mat[DType.uint8](copy=nxt)
    var levels = 0
    for l in range(max_level + 1):
        pa.append(Level(ia, border))
        pb.append(Level(ib, border))
        levels = l
        if l == max_level or ia.rows < 2 * win or ia.cols < 2 * win:
            break
        ia = pyr_down(ia)
        ib = pyr_down(ib)
    var n = pts.rows
    var onext = Mat[F32](n, 2, 1)
    var ostat = Mat[DType.uint8](n, 1, 1)
    var oerr = Mat[F32](n, 1, 1)
    var half = Float32(win - 1) * 0.5
    var w2 = win * win
    comptime W_BITS = 14
    comptime FLT_SCALE = Float32(1.0 / Float64(1 << 20))
    for p in range(n):
        ostat.data[p] = 1
        var nx = Float32(0)
        var ny = Float32(0)
        var px0 = pts.at(p, 0)
        var py0 = pts.at(p, 1)
        for level in range(levels, -1, -1):
            var sc = Float32(1.0 / Float64(1 << level))
            var ppx = px0 * sc
            var ppy = py0 * sc
            if level == levels:
                nx = ppx
                ny = ppy
            else:
                nx = nx * 2
                ny = ny * 2
            ppx -= half
            ppy -= half
            var ipx = Int(floor(ppx))
            var ipy = Int(floor(ppy))
            ref lv = pa[level]
            ref lj = pb[level]
            if ipx < -win or ipx >= lv.cols or ipy < -win or ipy >= lv.rows:
                if level == 0:
                    ostat.data[p] = 0
                    oerr.data[p] = 0
                onext.set(p, 0, 0, nx)
                onext.set(p, 1, 0, ny)
                continue
            var a = ppx - Float32(ipx)
            var b = ppy - Float32(ipy)
            var iw00 = rint(Float64((1.0 - a) * (1.0 - b) * Float32(1 << W_BITS)))
            var iw01 = rint(Float64(a * (1.0 - b) * Float32(1 << W_BITS)))
            var iw10 = rint(Float64((1.0 - a) * b * Float32(1 << W_BITS)))
            var iw11 = (1 << W_BITS) - iw00 - iw01 - iw10
            var iwin = List[Int](length=w2, fill=0)
            var gx = List[Int](length=w2, fill=0)
            var gy = List[Int](length=w2, fill=0)
            var a11 = Float32(0)
            var a12 = Float32(0)
            var a22 = Float32(0)
            for y in range(win):
                for x in range(win):
                    var ry = y + ipy + border
                    var rx = x + ipx + border
                    var ival = descale(Int(lv.img.at(ry, rx)) * iw00 + Int(lv.img.at(ry, rx + 1)) * iw01 + Int(lv.img.at(ry + 1, rx)) * iw10 + Int(lv.img.at(ry + 1, rx + 1)) * iw11, W_BITS - 5)
                    var ix = descale(Int(lv.dx.at(ry, rx)) * iw00 + Int(lv.dx.at(ry, rx + 1)) * iw01 + Int(lv.dx.at(ry + 1, rx)) * iw10 + Int(lv.dx.at(ry + 1, rx + 1)) * iw11, W_BITS)
                    var iy = descale(Int(lv.dy.at(ry, rx)) * iw00 + Int(lv.dy.at(ry, rx + 1)) * iw01 + Int(lv.dy.at(ry + 1, rx)) * iw10 + Int(lv.dy.at(ry + 1, rx + 1)) * iw11, W_BITS)
                    iwin[y * win + x] = ival
                    gx[y * win + x] = ix
                    gy[y * win + x] = iy
                    a11 += Float32(ix * ix)
                    a12 += Float32(ix * iy)
                    a22 += Float32(iy * iy)
            a11 *= FLT_SCALE
            a12 *= FLT_SCALE
            a22 *= FLT_SCALE
            var d = a11 * a22 - a12 * a12
            var min_eig = (a22 + a11 - sqrt((a11 - a22) * (a11 - a22) + 4.0 * a12 * a12)) / Float32(2 * w2)
            if min_eig < 1e-4 or d < 1.1920929e-07:
                if level == 0:
                    ostat.data[p] = 0
                onext.set(p, 0, 0, nx)
                onext.set(p, 1, 0, ny)
                continue
            d = 1.0 / d
            nx -= half
            ny -= half
            var pdx = Float32(0)
            var pdy = Float32(0)
            for j in range(max_count):
                var inx = Int(floor(nx))
                var iny = Int(floor(ny))
                if inx < -win or inx >= lj.cols or iny < -win or iny >= lj.rows:
                    if level == 0:
                        ostat.data[p] = 0
                    break
                a = nx - Float32(inx)
                b = ny - Float32(iny)
                iw00 = rint(Float64((1.0 - a) * (1.0 - b) * Float32(1 << W_BITS)))
                iw01 = rint(Float64(a * (1.0 - b) * Float32(1 << W_BITS)))
                iw10 = rint(Float64((1.0 - a) * b * Float32(1 << W_BITS)))
                iw11 = (1 << W_BITS) - iw00 - iw01 - iw10
                var b1 = Float32(0)
                var b2 = Float32(0)
                for y in range(win):
                    for x in range(win):
                        var ry = y + iny + border
                        var rx = x + inx + border
                        var diff = descale(Int(lj.img.at(ry, rx)) * iw00 + Int(lj.img.at(ry, rx + 1)) * iw01 + Int(lj.img.at(ry + 1, rx)) * iw10 + Int(lj.img.at(ry + 1, rx + 1)) * iw11, W_BITS - 5) - iwin[y * win + x]
                        b1 += Float32(diff * gx[y * win + x])
                        b2 += Float32(diff * gy[y * win + x])
                b1 *= FLT_SCALE
                b2 *= FLT_SCALE
                var ddx = (a12 * b2 - a22 * b1) * d
                var ddy = (a12 * b1 - a11 * b2) * d
                nx += ddx
                ny += ddy
                if ddx * ddx + ddy * ddy <= eps:
                    break
                if j > 0 and abs(ddx + pdx) < 0.01 and abs(ddy + pdy) < 0.01:
                    nx -= ddx * 0.5
                    ny -= ddy * 0.5
                    break
                pdx = ddx
                pdy = ddy
            nx += half
            ny += half
            if level == 0 and ostat.data[p] != 0:
                var fx = nx - half
                var fy = ny - half
                var ifx = Int(floor(fx))
                var ify = Int(floor(fy))
                if ifx < -win or ifx >= lj.cols or ify < -win or ify >= lj.rows:
                    ostat.data[p] = 0
                else:
                    var aa = fx - Float32(ifx)
                    var bb = fy - Float32(ify)
                    var w00 = rint(Float64((1.0 - aa) * (1.0 - bb) * Float32(1 << W_BITS)))
                    var w01 = rint(Float64(aa * (1.0 - bb) * Float32(1 << W_BITS)))
                    var w10 = rint(Float64((1.0 - aa) * bb * Float32(1 << W_BITS)))
                    var w11 = (1 << W_BITS) - w00 - w01 - w10
                    var errv = Float32(0)
                    for y in range(win):
                        for x in range(win):
                            var ry = y + ify + border
                            var rx = x + ifx + border
                            var diff = descale(Int(lj.img.at(ry, rx)) * w00 + Int(lj.img.at(ry, rx + 1)) * w01 + Int(lj.img.at(ry + 1, rx)) * w10 + Int(lj.img.at(ry + 1, rx + 1)) * w11, W_BITS - 5) - iwin[y * win + x]
                            errv += Float32(abs(diff))
                    oerr.data[p] = errv / Float32(32 * w2)
            onext.set(p, 0, 0, nx)
            onext.set(p, 1, 0, ny)
    return (onext^, ostat^, oerr^)


# ---- Farneback --------------------------------------------------------------------------------------------


def poly_exp(src: Mat[F32], n: Int, sigma_in: Float64) -> Mat[F32]:
    """Farneback 2003 polynomial expansion: per pixel the weighted least-squares fit
    f ~ c + bx x + by y + axx x^2 + ayy y^2 + axy x y over a (2n+1)^2 Gaussian window, stored as
    (bx, by, axx, ayy, axy) with replicated borders."""
    var sigma = sigma_in if sigma_in >= 1.1920929e-07 else Float64(n) * 0.3
    var g = List[Float64](length=2 * n + 1, fill=0)
    var sum = Float64(0)
    for k in range(-n, n + 1):
        g[k + n] = exp(-Float64(k * k) / (2 * sigma * sigma))
        sum += g[k + n]
    var m2 = Float64(0)
    var m4 = Float64(0)
    for k in range(-n, n + 1):
        g[k + n] /= sum
        m2 += g[k + n] * Float64(k * k)
        m4 += g[k + n] * Float64(k * k * k * k)
    # inverse of [[1, m2, m2], [m2, m4, m2^2], [m2, m2^2, m4]] gives the (const, x^2, y^2) coefficients
    var a00 = Float64(1)
    var det = a00 * (m4 * m4 - m2 * m2 * m2 * m2) - m2 * (m2 * m4 - m2 * m2 * m2) + m2 * (m2 * m2 * m2 - m4 * m2)
    var i10 = -(m2 * m4 - m2 * m2 * m2) / det
    var i11 = (a00 * m4 - m2 * m2) / det
    var i12 = -(a00 * m2 * m2 - m2 * m2) / det
    var o = Mat[F32](src.rows, src.cols, 5)
    var rows = src.rows
    var cols = src.cols
    var tmp = Mat[F32](rows, cols, 3)
    var pt = tmp.ptr()
    var po = o.ptr()

    def vrow(y: Int) {imm}:
        for x in range(cols):
            var c0 = Float32(0)
            var c1 = Float32(0)
            var c2 = Float32(0)
            for k in range(-n, n + 1):
                var v = src.at(min(max(y + k, 0), rows - 1), x, 0)
                var w = Float32(g[k + n])
                c0 += w * v
                c1 += w * Float32(k) * v
                c2 += w * Float32(k * k) * v
            pt[unsafe_offset=(y * cols + x) * 3 + 0] = c0
            pt[unsafe_offset=(y * cols + x) * 3 + 1] = c1
            pt[unsafe_offset=(y * cols + x) * 3 + 2] = c2

    par(vrow, rows, 16)

    def hrow(y: Int) {imm}:
        for x in range(cols):
            var b1 = Float64(0)
            var b2 = Float64(0)
            var b3 = Float64(0)
            var b4 = Float64(0)
            var b5 = Float64(0)
            var b6 = Float64(0)
            for k in range(-n, n + 1):
                var xx = min(max(x + k, 0), cols - 1)
                var w = g[k + n]
                var r0 = Float64(tmp.at(y, xx, 0))
                var r1 = Float64(tmp.at(y, xx, 1))
                var r2 = Float64(tmp.at(y, xx, 2))
                b1 += w * r0
                b2 += w * Float64(k) * r0
                b4 += w * Float64(k * k) * r0
                b3 += w * r1
                b6 += w * Float64(k) * r1
                b5 += w * r2
            po[unsafe_offset=(y * cols + x) * 5 + 0] = Float32(b2 / m2)
            po[unsafe_offset=(y * cols + x) * 5 + 1] = Float32(b3 / m2)
            po[unsafe_offset=(y * cols + x) * 5 + 2] = Float32(i10 * b1 + i11 * b4 + i12 * b5)
            po[unsafe_offset=(y * cols + x) * 5 + 3] = Float32(i10 * b1 + i12 * b4 + i11 * b5)
            po[unsafe_offset=(y * cols + x) * 5 + 4] = Float32(b6 / (m2 * m2))

    par(hrow, rows, 16)
    _ = tmp.rows
    return o^


def fb_matrices(r0: Mat[F32], r1: Mat[F32], flow: Mat[F32]) -> Mat[F32]:
    """Per pixel normal-equation terms (G11, G12, G22, h1, h2) of the displacement fit, with the flow so far
    applied to the second expansion; pixels within 5 of the border are down-weighted as in cv2."""
    var w = flow.cols
    var h = flow.rows
    var m = Mat[F32](h, w, 5)
    var pm = m.ptr()
    var border = SIMD[F32, 8](0.14, 0.14, 0.4472, 0.4472, 0.4472, 1, 1, 1)

    def row(y: Int) {imm}:
        for x in range(w):
            var dx = flow.at(y, x, 0)
            var dy = flow.at(y, x, 1)
            var fx = Float32(x) + dx
            var fy = Float32(y) + dy
            var x1 = Int(floor(fx))
            var y1 = Int(floor(fy))
            var r2 = Float32(0)
            var r3 = Float32(0)
            var r4: Float32
            var r5: Float32
            var r6: Float32
            if x1 >= 0 and x1 < w - 1 and y1 >= 0 and y1 < h - 1:
                var a00 = (Float32(x1 + 1) - fx) * (Float32(y1 + 1) - fy)
                var a01 = (fx - Float32(x1)) * (Float32(y1 + 1) - fy)
                var a10 = (Float32(x1 + 1) - fx) * (fy - Float32(y1))
                var a11 = (fx - Float32(x1)) * (fy - Float32(y1))
                r2 = r1.at(y1, x1, 0) * a00 + r1.at(y1, x1 + 1, 0) * a01 + r1.at(y1 + 1, x1, 0) * a10 + r1.at(y1 + 1, x1 + 1, 0) * a11
                r3 = r1.at(y1, x1, 1) * a00 + r1.at(y1, x1 + 1, 1) * a01 + r1.at(y1 + 1, x1, 1) * a10 + r1.at(y1 + 1, x1 + 1, 1) * a11
                r4 = r1.at(y1, x1, 2) * a00 + r1.at(y1, x1 + 1, 2) * a01 + r1.at(y1 + 1, x1, 2) * a10 + r1.at(y1 + 1, x1 + 1, 2) * a11
                r5 = r1.at(y1, x1, 3) * a00 + r1.at(y1, x1 + 1, 3) * a01 + r1.at(y1 + 1, x1, 3) * a10 + r1.at(y1 + 1, x1 + 1, 3) * a11
                r6 = r1.at(y1, x1, 4) * a00 + r1.at(y1, x1 + 1, 4) * a01 + r1.at(y1 + 1, x1, 4) * a10 + r1.at(y1 + 1, x1 + 1, 4) * a11
                r4 = (r0.at(y, x, 2) + r4) * 0.5
                r5 = (r0.at(y, x, 3) + r5) * 0.5
                r6 = (r0.at(y, x, 4) + r6) * 0.25
            else:
                r4 = r0.at(y, x, 2)
                r5 = r0.at(y, x, 3)
                r6 = r0.at(y, x, 4) * 0.5
            r2 = (r0.at(y, x, 0) - r2) * 0.5
            r3 = (r0.at(y, x, 1) - r3) * 0.5
            r2 += r4 * dx + r6 * dy
            r3 += r6 * dx + r5 * dy
            if x < 5 or y < 5 or x >= w - 5 or y >= h - 5:
                var sc = (border[x] if x < 5 else Float32(1)) * (border[w - x - 1] if x >= w - 5 else Float32(1)) * (border[y] if y < 5 else Float32(1)) * (border[h - y - 1] if y >= h - 5 else Float32(1))
                r2 *= sc
                r3 *= sc
                r4 *= sc
                r5 *= sc
                r6 *= sc
            pm[unsafe_offset=(y * w + x) * 5 + 0] = r4 * r4 + r6 * r6
            pm[unsafe_offset=(y * w + x) * 5 + 1] = (r4 + r5) * r6
            pm[unsafe_offset=(y * w + x) * 5 + 2] = r5 * r5 + r6 * r6
            pm[unsafe_offset=(y * w + x) * 5 + 3] = r4 * r2 + r6 * r3
            pm[unsafe_offset=(y * w + x) * 5 + 4] = r6 * r2 + r5 * r3

    par(row, h, 16)
    return m^


def fb_solve(m: Mat[F32], block: Int, mut flow: Mat[F32]):
    """Box-average the terms over a block x block window (replicated border) and solve the 2x2 system."""
    var box = List[Float32](length=block, fill=1)
    var sm = sepfilter[F32, F32](m, box, box, 1)
    var scale = 1.0 / Float64(block * block)
    for y in range(m.rows):
        for x in range(m.cols):
            var g11 = Float64(sm.at(y, x, 0)) * scale
            var g12 = Float64(sm.at(y, x, 1)) * scale
            var g22 = Float64(sm.at(y, x, 2)) * scale
            var h1 = Float64(sm.at(y, x, 3)) * scale
            var h2 = Float64(sm.at(y, x, 4)) * scale
            var idet = 1.0 / (g11 * g22 - g12 * g12 + 1e-3)
            flow.set(y, x, 0, Float32((g22 * h1 - g12 * h2) * idet))
            flow.set(y, x, 1, Float32((g11 * h2 - g12 * h1) * idet))


def farneback(prev: Mat[DType.uint8], nxt: Mat[DType.uint8], pyr_scale: Float64, levels_in: Int, winsize: Int, iters: Int, poly_n: Int, poly_sigma: Float64) -> Mat[F32]:
    """cv2.calcOpticalFlowFarneback (box window): coarse-to-fine, expansions of the blurred and resized frames."""
    var levels = 0
    var scale = Float64(1)
    for _ in range(levels_in):
        scale *= pyr_scale
        if Float64(prev.cols) * scale < 32 or Float64(prev.rows) * scale < 32:
            break
        levels += 1
    var f0 = Mat[F32](prev.rows, prev.cols, 1)
    var f1 = Mat[F32](prev.rows, prev.cols, 1)
    for i in range(prev.size()):
        f0.data[i] = Float32(prev.data[i])
        f1.data[i] = Float32(nxt.data[i])
    var flow = Mat[F32](0, 0)
    for k in range(levels, -1, -1):
        var sc = Float64(1)
        for _ in range(k):
            sc *= pyr_scale
        var sigma = (1.0 / sc - 1.0) * 0.5
        var sm = max(rint(sigma * 5) | 1, 3)
        var w = rint(Float64(prev.cols) * sc)
        var h = rint(Float64(prev.rows) * sc)
        var nf: Mat[F32]
        if flow.rows == 0:
            nf = Mat[F32](h, w, 2)
        else:
            nf = resize(flow, w, h, 1)
            for i in range(nf.size()):
                nf.data[i] = nf.data[i] * Float32(1.0 / pyr_scale)
        flow = nf^
        var b0 = gaussian_blur(f0, sm, sigma)
        var b1 = gaussian_blur(f1, sm, sigma)
        var r0 = poly_exp(resize(b0, w, h, 1), poly_n, poly_sigma)
        var r1 = poly_exp(resize(b1, w, h, 1), poly_n, poly_sigma)
        var m = fb_matrices(r0, r1, flow)
        for it in range(iters):
            fb_solve(m, winsize, flow)
            if it < iters - 1:
                m = fb_matrices(r0, r1, flow)
    return flow^


def run(mut a: Args) raises -> Bool:
    if a.op == "cornerSubPix":
        var im = load[DType.uint8](a.ins[0])
        var pts = load[F32](a.ins[1])
        var o = Mat[F32](0, 0)
        while a.again():
            o = corner_sub_pix(im, pts, a.i("win", 5), a.i("zero", -1), a.i("maxIter", 30), a.f("eps", 0.01))
        save(o, a.out)
    elif a.op == "calcOpticalFlowFarneback":
        var p0 = load[DType.uint8](a.ins[0])
        var p1 = load[DType.uint8](a.ins[1])
        var o = Mat[F32](0, 0)
        while a.again():
            o = farneback(p0, p1, a.f("pyrScale", 0.5), a.i("levels", 3), a.i("winsize", 15), a.i("iters", 3), a.i("polyN", 5), a.f("polySigma", 1.2))
        save(o, a.out)
    elif a.op == "calcOpticalFlowPyrLK":
        var p0 = load[DType.uint8](a.ins[0])
        var p1 = load[DType.uint8](a.ins[1])
        var pts = load[F32](a.ins[2])
        var r = (Mat[F32](0, 0), Mat[DType.uint8](0, 0), Mat[F32](0, 0))
        while a.again():
            r = lk_flow(p0, p1, pts, a.i("win", 21), a.i("maxLevel", 3), a.i("maxCount", 30), Float32(a.f("eps", 0.01)))
        save(r[0], a.out)
        save(r[1], a.out + ".1")
        save(r[2], a.out + ".2")
    else:
        return False
    return True
