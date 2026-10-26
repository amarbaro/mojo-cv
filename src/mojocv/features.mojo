# features: binary-descriptor matching and ORB. Descriptors are rows of an (n, 32, 1) uint8 Mat.
from std.bit import pop_count
from mojocv.mat import Mat, load, save
from std.math import floor, pow, cos, sin, sqrt
from mojocv.core import Args, par
from mojocv.imgproc import resize, gaussian_blur, rint, Ptr
from std.sys.info import simd_width_of


def hamming(a: Mat[DType.uint8], ia: Int, b: Mat[DType.uint8], ib: Int) -> Int:
    var d = 0
    for k in range(a.cols):
        d += Int(pop_count(a.data[ia * a.cols + k] ^ b.data[ib * b.cols + k]))
    return d


def knn2(q: Mat[DType.uint8], t: Mat[DType.uint8]) -> Mat[DType.int32]:
    """For every query row the two nearest train rows by Hamming distance: (n, 4) = train1, dist1, train2, dist2.
    Equal distances keep the lower train index first, as cv2's brute-force matcher does."""
    var o = Mat[DType.int32](q.rows, 4, 1)
    var po = o.ptr()

    def row(i: Int) {imm}:
        var b1 = 1 << 30
        var b2 = 1 << 30
        var i1 = -1
        var i2 = -1
        for j in range(t.rows):
            var d = hamming(q, i, t, j)
            if d < b1:
                b2 = b1
                i2 = i1
                b1 = d
                i1 = j
            elif d < b2:
                b2 = d
                i2 = j
        po[unsafe_offset=4 * i] = Int32(i1)
        po[unsafe_offset=4 * i + 1] = Int32(b1)
        po[unsafe_offset=4 * i + 2] = Int32(i2)
        po[unsafe_offset=4 * i + 3] = Int32(b2)

    par(row, q.rows)
    return o^


def ratio_test(k: Mat[DType.int32], ratio: Float64) -> Mat[DType.int32]:
    """Lowe's test on knn2 output: keep (query, train, dist) where d1 < ratio * d2."""
    var keep = List[Int32]()
    for i in range(k.rows):
        if Float64(k.at(i, 1)) < ratio * Float64(k.at(i, 3)):
            keep.append(Int32(i))
            keep.append(k.at(i, 0))
            keep.append(k.at(i, 1))
    var o = Mat[DType.int32](len(keep) // 3, 3, 1)
    for i in range(len(keep)):
        o.data[i] = keep[i]
    return o^


# ---- ORB --------------------------------------------------------------------------------------------------
# Rublee et al. 2011: FAST-9 corners on an 8-level pyramid, Harris ranking, intensity-centroid orientation, rotated
# BRIEF. The 256 comparison pairs (ax, ay, bx, by) are the learned rBRIEF pattern of the paper, recovered from
# cv2's ORB by probing it with single-pixel images (each pair was the unique solution), listed here as data.
comptime ORB_PAIRS = "8,-3,9,5,4,2,7,-12,-11,9,-8,2,7,-12,12,-13,2,-13,2,12,1,-7,1,6,-2,-10,-2,-4,-13,-13,-11,-8,-13,-3,-12,-9,10,4,11,9,-13,-8,-8,-9,-11,7,-9,12,7,7,12,6,-4,-5,-3,0,-13,2,-12,-3,-9,0,-7,5,12,-6,12,-1,-3,6,-2,12,-6,-13,-4,-8,11,-13,12,-8,4,7,5,1,5,-3,10,-3,3,-7,6,12,-8,-7,-6,-2,-2,11,-1,-10,-13,12,-8,10,-7,3,-5,-3,-4,2,-3,7,-10,-12,-6,11,5,-12,6,-7,5,-6,7,-1,1,0,4,-5,9,11,11,-13,4,7,4,12,2,-1,4,4,-4,-12,-2,7,-8,-5,-7,-10,4,11,9,12,0,-8,1,-13,-13,-2,-8,2,-3,-2,-2,3,-6,9,-4,-9,8,12,10,7,0,9,1,3,7,-5,11,-10,-13,-6,-11,0,10,7,12,1,-6,-3,-6,12,10,-9,12,-4,-13,8,-8,-12,-13,0,-8,-4,3,3,7,8,5,7,10,-7,-1,7,1,-12,3,-10,5,6,2,-4,3,-10,-13,0,-13,5,-13,-7,-12,12,-13,3,-11,8,-7,12,-4,7,6,-10,12,8,-9,-1,-7,-6,-2,-5,0,12,-12,5,-7,5,3,-10,8,-13,-7,-7,-4,5,-3,-2,-1,-7,2,9,5,-11,-11,-13,-5,-13,-1,6,0,-1,5,-3,5,2,-4,-13,-4,12,-9,-6,-9,6,-12,-10,-8,-4,10,2,12,-3,7,12,12,12,-7,-13,-6,5,-4,9,-3,4,7,-1,12,2,-7,6,-5,1,-13,11,-12,5,-3,7,-2,-6,7,-8,12,-7,-13,-7,-11,-12,1,-3,12,12,2,-6,3,0,-4,3,-2,-13,-1,-13,1,9,7,1,8,-6,1,-1,3,12,9,1,12,6,-1,-9,-1,3,-13,-13,-10,5,7,7,10,12,12,-5,12,9,6,3,7,11,5,-13,6,10,2,-12,2,3,3,8,4,-6,2,6,12,-13,9,-12,10,3,-8,4,-7,9,-11,12,-4,-6,1,12,2,-8,6,-9,7,-4,2,3,3,-2,6,3,11,0,3,-3,8,-8,7,8,9,3,-11,-5,-6,-4,-10,11,-5,10,-5,-8,-3,12,-10,5,-9,0,8,-1,12,-6,4,-6,6,-11,-10,12,-8,7,4,-2,6,7,-2,0,-2,12,-5,-8,-5,2,7,-6,10,12,-9,-13,-8,-8,-5,-13,-5,-2,8,-8,9,-13,-9,-11,-9,0,1,-8,1,-2,7,-4,9,1,-2,1,-1,-4,11,-6,12,-11,-12,-9,-6,4,3,7,7,12,5,5,10,8,0,-4,2,8,-9,12,-5,-13,0,7,2,12,-1,2,1,7,5,11,7,-9,3,5,6,-8,-13,-4,-8,9,-5,9,-3,-3,-4,-7,-3,-12,6,5,8,0,-7,6,-6,12,-13,6,-5,-2,1,-10,3,10,4,1,8,-4,-2,-2,2,-13,2,-12,12,12,-2,-13,0,-6,4,1,9,3,-6,-10,-3,-5,-3,-13,-1,1,7,5,12,-11,4,-2,5,-7,-13,9,-9,-5,7,1,8,6,7,-8,7,6,-7,-4,-7,1,-8,11,-7,-8,-13,6,-12,-8,2,4,3,9,10,-5,12,3,-6,-5,-6,7,8,-3,9,-8,2,-12,2,8,-11,-2,-10,3,-12,-13,-7,-9,-11,0,-10,-5,5,-3,11,8,-2,-13,-1,12,-1,-8,0,9,-13,-11,-12,-5,-10,-2,-10,11,-3,9,-2,-13,2,-3,3,2,-9,-13,-4,0,-4,6,-3,-10,-4,12,-2,-7,-6,-11,-4,9,6,-3,6,11,-13,11,-5,5,11,11,12,6,7,-5,12,-2,-1,12,0,7,-4,-8,-3,-2,-7,1,-6,7,-13,-12,-8,-13,-7,-2,-6,-8,-8,5,-6,-9,-5,-1,-4,5,-13,7,-8,10,1,5,5,-13,1,0,10,-13,9,12,10,-1,5,-8,10,-9,-1,11,1,-13,-9,-3,-6,2,-1,-10,1,12,-13,1,-8,-10,8,-11,10,-6,2,-13,3,-6,7,-13,12,-9,-10,-10,-5,-7,-10,-8,-8,-13,4,-6,8,5,3,12,8,-13,-4,2,-3,-3,5,-13,10,-12,4,-13,5,-1,-9,9,-4,3,0,3,3,-9,-12,1,-6,1,3,2,4,-8,-10,-10,-10,9,8,-13,12,12,-8,-12,-6,-5,2,2,3,7,10,6,11,-8,6,8,8,-12,-7,10,-6,5,-3,-9,-3,9,-1,-13,-1,5,-3,-7,-3,4,-8,-2,-8,3,4,2,12,12,2,-5,3,11,6,-9,11,-13,3,-1,7,12,11,-1,12,4,-3,0,-3,6,4,-11,4,12,2,-4,2,1,-10,-6,-8,1,-13,7,-11,1,-13,12,-11,-13,6,0,11,-13,0,-1,1,4,-13,3,-9,-2,-9,8,-6,-3,-13,-6,-8,-2,5,-9,8,10,2,7,3,-9,-1,-6,-1,-1,9,5,11,-2,11,-3,12,-8,3,0,3,5,-1,4,0,10,3,-6,4,5,-13,0,-10,5,5,8,12,11,8,9,9,-6,7,-4,8,-12,-10,4,-10,9,7,3,12,4,9,-7,10,-2,7,0,12,-2,-1,-6,0,-11"


def orb_pairs() raises -> List[Int]:
    var out = List[Int]()
    for t in ORB_PAIRS.split(","):
        out.append(atol(String(t)))
    return out^


def fast_atan2(y: Float32, x: Float32) -> Float32:
    """Degrees in [0, 360), cv2's polynomial approximation."""
    comptime k = Float32(180.0 / 3.14159265358979323846)
    comptime p1 = Float32(0.9997878412794807) * k
    comptime p3 = Float32(-0.3258083974640975) * k
    comptime p5 = Float32(0.1555786518463281) * k
    comptime p7 = Float32(-0.04432655554792128) * k
    var ax = abs(x)
    var ay = abs(y)
    var a: Float32
    if ax >= ay:
        var c = ay / (ax + 2.220446049250313e-16)
        var c2 = c * c
        a = (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c
    else:
        var c = ax / (ay + 2.220446049250313e-16)
        var c2 = c * c
        a = 90.0 - (((p7 * c2 + p5) * c2 + p3) * c2 + p1) * c
    if x < 0:
        a = 180.0 - a
    if y < 0:
        a = 360.0 - a
    return a


def fast_score(img: Mat[DType.uint8], y: Int, x: Int, t: Int) -> Int:
    """FAST-9 on the 16-pixel circle: the largest threshold at which (y, x) still is a corner, or 0."""
    var dx = SIMD[DType.int32, 16](0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1)
    var dy = SIMD[DType.int32, 16](3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3)
    var c = Int(img.at(y, x))
    # a 9-arc contains at least two of the four compass pixels on one side of the threshold
    var up = 0
    var dn = 0
    for k in range(0, 16, 4):
        var v = Int(img.at(y + Int(dy[k]), x + Int(dx[k]))) - c
        up += Int(v > t)
        dn += Int(v < -t)
    if up < 2 and dn < 2:
        return 0
    var d = SIMD[DType.int32, 16](0)
    for k in range(16):
        d[k] = Int32(Int(img.at(y + Int(dy[k]), x + Int(dx[k]))) - c)
    # quick reject: four compass points
    var best = 0
    for sign in range(2):
        var run_best = 0
        for start in range(16):
            var m = 1 << 20
            for k in range(9):
                var v = Int(d[(start + k) % 16])
                m = min(m, v if sign == 0 else -v)
            run_best = max(run_best, m)
        best = max(best, run_best)
    return best - 1 if best - 1 >= t else 0


def fast_vec[W: Int](ps: Ptr[UInt8], i: Int, stride: Int, t: Int) -> SIMD[DType.int32, W]:
    """fast_score for the W pixels starting at ps[i]: per lane the best of the 16 nine-pixel arcs, all of them
    above (or below) the centre by more than t, found as max over arcs of the arc minimum of the 16 circle diffs."""
    comptime dxs = SIMD[DType.int32, 16](0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1)
    comptime dys = SIMD[DType.int32, 16](3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1, 0, 1, 2, 3)
    comptime V = SIMD[DType.int16, W]
    var c = ps.unsafe_load[width=W](i).cast[DType.int16]()
    var up = V(-32768)
    var dn = V(32767)
    comptime for k in range(16):
        var lo = V(32767)
        var hi = V(-32768)
        comptime for j in range(9):
            comptime q = (k + j) % 16
            var v = ps.unsafe_load[width=W](i + Int(dys[q]) * stride + Int(dxs[q])).cast[DType.int16]() - c
            lo = min(lo, v)
            hi = max(hi, v)
        up = max(up, lo)
        dn = min(dn, hi)
    var best = max(up, -dn).cast[DType.int32]()
    return (best.ge(Int32(t + 1))).select(best - 1, SIMD[DType.int32, W](0))


struct KP(Copyable, Movable):
    var x: Int
    var y: Int
    var resp: Float32
    var angle: Float32

    def __init__(out self, x: Int, y: Int, resp: Float32):
        self.x = x
        self.y = y
        self.resp = resp
        self.angle = 0


def retain_best(mut kps: List[KP], n: Int):
    """cv2 KeyPointsFilter::retainBest: the n strongest plus everything tied with the n-th."""
    if n >= len(kps):
        return
    if n <= 0:
        kps.clear()
        return

    def greater(a: KP, b: KP) -> Bool:
        return a.resp > b.resp

    sort(Span(kps), greater)
    var edge = kps[n - 1].resp
    var cut = n
    while cut < len(kps) and kps[cut].resp >= edge:
        cut += 1
    var kept = List[KP]()
    for i in range(cut):
        kept.append(kps[i].copy())
    kps = kept^


def orb_detect(img: Mat[DType.uint8], nfeatures: Int, fast_t: Int) raises -> Mat[DType.float32]:
    """Rows of (x, y, angle, octave, 32 descriptor bytes) as float32."""
    comptime nlevels = 8
    comptime scale_f = 1.2
    comptime edge = 31
    var pairs = orb_pairs()
    var factor = 1.0 / scale_f
    var per = Float64(nfeatures) * (1.0 - factor) / (1.0 - pow(factor, Float64(nlevels)))
    var counts = List[Int]()
    var tot = 0
    for _ in range(nlevels - 1):
        counts.append(rint(per))
        tot += counts[len(counts) - 1]
        per *= factor
    counts.append(max(nfeatures - tot, 0))
    var levels = List[Mat[DType.uint8]]()
    levels.append(Mat[DType.uint8](copy=img))
    for l in range(1, nlevels):
        var sc = pow(scale_f, Float64(l))
        levels.append(resize(levels[l - 1], rint(Float64(img.cols) / sc), rint(Float64(img.rows) / sc), 1))
    var rows_out = List[List[Float32]]()
    var umax = List[Int](length=16, fill=0)
    var vmax = Int(floor(15.0 * 1.4142135623730951 / 2 + 1))
    var vmin = Int(floor(15.0 * 1.4142135623730951 / 2 + 0.999999))
    for v in range(vmax + 1):
        umax[v] = rint(sqrt(Float64(15 * 15 - v * v)))
    var v0 = 0
    for v in range(15, vmin - 1, -1):
        while umax[v0] == umax[v0 + 1]:
            v0 += 1
        umax[v] = v0
        v0 += 1
    for l in range(nlevels):
        ref im = levels[l]
        var cand = List[KP]()
        var rows = im.rows
        var cols = im.cols
        var sc = Mat[DType.int32](rows, cols, 1)
        var psc = sc.ptr()
        var pim = im.ptr()
        comptime FW = simd_width_of[DType.int16]()

        def score_row(y: Int) {imm}:
            if y < 3 or y >= rows - 3:
                return
            var x = 3
            while x + FW <= cols - 3:
                psc.unsafe_store(y * cols + x, fast_vec[FW](pim, y * cols + x, cols, fast_t))
                x += FW
            for xx in range(x, cols - 3):
                psc[unsafe_offset=y * cols + xx] = Int32(fast_score(im, y, xx, fast_t))

        par(score_row, rows)
        _ = sc.rows
        comptime WI = simd_width_of[DType.int32]()
        for y in range(3, rows - 3):
            var x = 3
            while x < cols - 3:
                if x + WI <= cols - 3 and not psc.unsafe_load[width=WI](y * cols + x).ne(0).reduce_or():
                    x += WI
                    continue
                for xx in range(x, min(x + WI, cols - 3)):
                    var s = psc[unsafe_offset=y * cols + xx]
                    if s == 0:
                        continue
                    var peak = True
                    for ddy in range(-1, 2):
                        for ddx in range(-1, 2):
                            if (ddy != 0 or ddx != 0) and psc[unsafe_offset=(y + ddy) * cols + xx + ddx] >= s:
                                peak = False
                    if peak and xx >= edge and xx < cols - edge and y >= edge and y < rows - edge:
                        cand.append(KP(xx, y, Float32(s)))
                x += WI
        retain_best(cand, 2 * counts[l])
        for i in range(len(cand)):
            var x0 = cand[i].x
            var y0 = cand[i].y
            var a = Float32(0)
            var b = Float32(0)
            var c = Float32(0)
            for yy in range(y0 - 3, y0 + 4):
                for xx in range(x0 - 3, x0 + 4):
                    var ix = Float32((Int(im.at(yy, xx + 1)) - Int(im.at(yy, xx - 1))) * 2 + (Int(im.at(yy - 1, xx + 1)) - Int(im.at(yy - 1, xx - 1))) + (Int(im.at(yy + 1, xx + 1)) - Int(im.at(yy + 1, xx - 1))))
                    var iy = Float32((Int(im.at(yy + 1, xx)) - Int(im.at(yy - 1, xx))) * 2 + (Int(im.at(yy + 1, xx - 1)) - Int(im.at(yy - 1, xx - 1))) + (Int(im.at(yy + 1, xx + 1)) - Int(im.at(yy - 1, xx + 1))))
                    a += ix * ix
                    b += iy * iy
                    c += ix * iy
            var scale = 1.0 / Float32(4 * 7 * 255)
            var ss = scale * scale * scale * scale
            cand[i].resp = (a * b - c * c - 0.04 * (a + b) * (a + b)) * ss
        retain_best(cand, counts[l])
        var blurred = gaussian_blur(im, 7, 2.0)
        var sl = pow(scale_f, Float64(l))
        for i in range(len(cand)):
            var x0 = cand[i].x
            var y0 = cand[i].y
            var m10 = 0
            var m01 = 0
            for u in range(-15, 16):
                m10 += u * Int(im.at(y0, x0 + u))
            for v in range(1, 16):
                var vsum = 0
                var d = umax[v]
                for u in range(-d, d + 1):
                    var vp = Int(im.at(y0 + v, x0 + u))
                    var vm = Int(im.at(y0 - v, x0 + u))
                    vsum += vp - vm
                    m10 += u * (vp + vm)
                m01 += v * vsum
            var ang = fast_atan2(Float32(m01), Float32(m10))
            var rad = ang * Float32(3.14159265358979323846 / 180.0)
            var ca = Float32(cos(Float64(rad)))
            var sa = Float32(sin(Float64(rad)))
            var row = List[Float32]()
            row.append(Float32(Float64(x0) * sl))
            row.append(Float32(Float64(y0) * sl))
            row.append(ang)
            row.append(Float32(l))
            for byte in range(32):
                var val = 0
                for bit in range(8):
                    var k = byte * 8 + bit
                    var ax = Float32(pairs[4 * k])
                    var ay = Float32(pairs[4 * k + 1])
                    var bx = Float32(pairs[4 * k + 2])
                    var by = Float32(pairs[4 * k + 3])
                    var t0 = Int(blurred.at(y0 + rint(Float64(ax * sa + ay * ca)), x0 + rint(Float64(ax * ca - ay * sa))))
                    var t1 = Int(blurred.at(y0 + rint(Float64(bx * sa + by * ca)), x0 + rint(Float64(bx * ca - by * sa))))
                    if t0 < t1:
                        val |= 1 << bit
                row.append(Float32(val))
            rows_out.append(row^)
    var o = Mat[DType.float32](len(rows_out), 36, 1)
    for i in range(len(rows_out)):
        for j in range(36):
            o.set(i, j, 0, rows_out[i][j])
    return o^


def run(mut a: Args) raises -> Bool:
    if a.op == "ORB":
        var im = load[DType.uint8](a.ins[0])
        var o = Mat[DType.float32](0, 0)
        while a.again():
            o = orb_detect(im, a.i("nfeatures", 500), a.i("fastThreshold", 20))
        save(o, a.out)
    elif a.op == "bfmatch":
        var q = load[DType.uint8](a.ins[0])
        var t = load[DType.uint8](a.ins[1])
        var k = Mat[DType.int32](0, 0)
        while a.again():
            k = knn2(q, t)
        var o = Mat[DType.int32](q.rows, 2, 1)
        for i in range(q.rows):
            o.set(i, 0, 0, k.at(i, 0))
            o.set(i, 1, 0, k.at(i, 1))
        save(o, a.out)
    elif a.op == "knnmatch":
        var q = load[DType.uint8](a.ins[0])
        var t = load[DType.uint8](a.ins[1])
        var k = Mat[DType.int32](0, 0)
        var g = Mat[DType.int32](0, 0)
        while a.again():
            k = knn2(q, t)
            g = ratio_test(k, a.f("ratio", 0.75))
        save(k, a.out)
        save(g, a.out + ".1")
    else:
        return False
    return True
