# Binary 3D point descriptors. For a point p take its k nearest neighbours q_0..q_{k-1} (by distance, ties by
# index). Bit j is 1 when the triple product (u_a x u_b) . u_c > 0 for the j-th rank triple (a, b, c) of a fixed
# table, u = q - p. The sign is the handedness of the three neighbours: unbiased (a dot-product sign test came out
# 70% ones and 6 of 100 points shared a descriptor), unchanged by rotation, translation and uniform scale, and
# flipped by a mirror.
# A descriptor is 64 bits, stored as 8 little-endian bytes: describe gives N x 8 uint8, match gives N x 2 int32.
from std.bit import pop_count
from max.algorithm import parallelize
from mojocv.mat import Mat
from mojocv.octree import splitmix

comptime BITS = 64


def triple_table(k: Int) -> List[Int]:
    """Distinct neighbour ranks (a, b, c) for each of the 64 tests, from a fixed splitmix stream."""
    var t = List[Int]()
    for j in range(BITS):
        var a = Int(splitmix(UInt64(3 * j)) % UInt64(k))
        var b = Int(splitmix(UInt64(3 * j + 1)) % UInt64(k))
        var c = Int(splitmix(UInt64(3 * j + 2)) % UInt64(k))
        while b == a:
            b = (b + 1) % k
        while c == a or c == b:
            c = (c + 1) % k
        t.append(a)
        t.append(b)
        t.append(c)
    return t^


def describe(p: Mat[DType.float64], k: Int) raises -> Mat[DType.uint8]:
    if p.cols != 3 or p.ch != 1 or k < 3 or p.rows <= k:
        raise Error("mojocv: describe wants an N x 3 float64 cloud with N > k >= 3")
    var tri = triple_table(k)
    var out = Mat[DType.uint8](p.rows, 8)
    var nd = List[Float64](length=k, fill=0.0)  # k nearest so far, ascending (distance^2, index)
    var ni = List[Int](length=k, fill=0)
    for i in range(p.rows):
        var cnt = 0
        for j in range(p.rows):
            if j == i:
                continue
            var dx = p.data[j * 3] - p.data[i * 3]
            var dy = p.data[j * 3 + 1] - p.data[i * 3 + 1]
            var dz = p.data[j * 3 + 2] - p.data[i * 3 + 2]
            var d = dx * dx + dy * dy + dz * dz
            if cnt == k and d >= nd[k - 1]:
                continue
            var s = cnt if cnt < k else k - 1
            while s > 0 and nd[s - 1] > d:
                nd[s] = nd[s - 1]
                ni[s] = ni[s - 1]
                s -= 1
            nd[s] = d
            ni[s] = j
            if cnt < k:
                cnt += 1
        var bits = UInt64(0)
        for b in range(BITS):
            var u = List[Float64]()
            for t in range(3):
                for a in range(3):
                    u.append(p.data[ni[tri[3 * b + t]] * 3 + a] - p.data[i * 3 + a])
            var cx = u[1] * u[5] - u[2] * u[4]
            var cy = u[2] * u[3] - u[0] * u[5]
            var cz = u[0] * u[4] - u[1] * u[3]
            if (cx * u[6] + cy * u[7]) + cz * u[8] > 0.0:
                bits |= UInt64(1) << UInt64(b)
        for byte in range(8):
            out.data[i * 8 + byte] = UInt8((bits >> UInt64(8 * byte)) & 0xFF)
    return out^


def _word(d: Mat[DType.uint8], i: Int) -> UInt64:
    var w = UInt64(0)
    for byte in range(8):
        w |= UInt64(d.data[i * 8 + byte]) << UInt64(8 * byte)
    return w


def match_desc(a: Mat[DType.uint8], b: Mat[DType.uint8]) raises -> Mat[DType.int32]:
    """For each descriptor of a: [index of the nearest descriptor in b by Hamming distance, that distance]."""
    if a.cols != 8 or b.cols != 8 or a.ch != 1 or b.ch != 1 or b.rows < 1:
        raise Error("mojocv: match wants N x 8 and M x 8 uint8 descriptors with M >= 1")
    var wa = List[UInt64]()
    var wb = List[UInt64]()
    for i in range(a.rows):
        wa.append(_word(a, i))
    for j in range(b.rows):
        wb.append(_word(b, j))
    var out = Mat[DType.int32](a.rows, 2)
    var na = a.rows
    var nb = b.rows
    var po = out.ptr()

    def rows_work(c: Int) {imm}:
        var per = (na + 63) // 64
        for i in range(c * per, min(na, (c + 1) * per)):
            var w = wa[i]
            var best = 65
            var arg = 0
            for j in range(nb):
                var h = Int(pop_count(w ^ wb[j]))
                if h < best:  # strict: the lowest index wins a tie
                    best = h
                    arg = j
            po[i * 2] = Int32(arg)
            po[i * 2 + 1] = Int32(best)

    parallelize(rows_work, 64)
    return out^
