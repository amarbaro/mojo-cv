# core: the CLI argument record, dtype dispatch and the array-level ops (absdiff, bitwise, flip, rotate,
# transpose, convert). Semantics follow cv2 5.0: saturating casts, round half to even.
from std.sys import argv
from std.sys.info import simd_width_of, size_of
from std.time import monotonic
from std.math import clamp
from max.algorithm import parallelize
from std.atomic import Atomic
from mojocv.mat import Mat, load, save

comptime DTS = [DType.uint8, DType.float32, DType.int32, DType.float64, DType.int16]


struct Args(Movable):
    """`mojo-cv OP OUT IN... [k=v]`; `reps=N` repeats the timed part and prints the best time."""

    var op: String
    var out: String
    var ins: List[String]
    var kv: Dict[String, String]
    var reps: Int
    var done: Int
    var t0: Int
    var best: Float64

    def __init__(out self) raises:
        var v = argv()
        if len(v) < 3:
            raise Error("usage: mojo-cv OP OUT IN... [k=v]")
        self.op = String(v[1])
        self.out = String(v[2])
        self.ins = List[String]()
        self.kv = Dict[String, String]()
        for i in range(3, len(v)):
            var s = String(v[i])
            var e = s.find("=")
            if e >= 0:
                self.kv[String(s[byte=:e])] = String(s[byte=e + 1 :])
            else:
                self.ins.append(s^)
        self.reps = 1
        self.done = 0
        self.t0 = 0
        self.best = 1e30
        self.reps = self.i("reps", 1)

    def f(self, k: String, d: Float64) raises -> Float64:
        return atof(self.kv[k]) if k in self.kv else d

    def i(self, k: String, d: Int) raises -> Int:
        return atol(self.kv[k]) if k in self.kv else d

    def fl(self, k: String, d: List[Float64]) raises -> List[Float64]:
        """A comma list such as M=1,0,0,0,1,0."""
        if k not in self.kv:
            return d.copy()
        var out = List[Float64]()
        for t in self.kv[k].split(","):
            out.append(atof(String(t)))
        return out^

    def s(self, k: String, d: String) raises -> String:
        return self.kv[k] if k in self.kv else d

    def again(mut self) -> Bool:
        """`while a.again(): o = op(...)` runs the body `reps` times and keeps the best wall time in ms."""
        var t = monotonic()
        if self.done > 0:
            self.best = min(self.best, Float64(t - self.t0) / 1e6)
        self.t0 = t
        self.done += 1
        return self.done <= self.reps


def par[F: def(Int) -> None](f: F, n: Int, workers: Int = 8):
    """parallelize on the 8 physical cores: with 16 workers every call waits ~215 us for a sleeping SMT sibling
    (measured), with 8 the dispatch takes ~8 us. Pass 16 for regions that run for many milliseconds."""
    parallelize(f, n, workers)


def par_dyn[F: def(Int) -> None](f: F, n: Int, workers: Int = 8):
    """f(0) .. f(n - 1) on `workers` threads, each taking the next unclaimed index from one shared counter, so threads
    that drew cheap items (rows outside a rotated image) take more of them instead of idling at the join."""
    var ctr = List[Int64](length=1, fill=0)
    var pc = MutPointer[Int64, MutAnyOrigin](unsafe_from_address=Int(ctr.unsafe_ptr()))

    def worker(_t: Int) {imm}:
        while True:
            var i = Int(Atomic[Scalar[DType.int64]].fetch_add(pc, 1))
            if i >= n:
                break
            f(i)

    parallelize(worker, workers, workers)
    _ = len(ctr)


def peek_code(path: String) raises -> Int:
    """The dtype code of an .mcv file, read from its header."""
    var h = open(path, "r").read_bytes(16)
    return Int(h[14]) | Int(h[15]) << 8


def _need_same[a: DType, b: DType](x: Mat[a], y: Mat[b]) raises:
    if not x.same_shape(y):
        raise Error("mojocv: input shapes differ")


# ---- elementwise ----------------------------------------------------------------------------------------------


def absdiff[dt: DType](a: Mat[dt], b: Mat[dt]) raises -> Mat[dt]:
    _need_same(a, b)
    var o = Mat[dt](a.rows, a.cols, a.ch)
    comptime W = simd_width_of[dt]()
    var n = a.cols * a.ch
    var pa = a.ptr()
    var pb = b.ptr()
    var po = o.ptr()

    def row(r: Int) {imm}:
        var base = r * n
        comptime if dt.is_floating_point() or dt.is_unsigned():
            var i = 0
            while i + W <= n:
                var x = pa.unsafe_load[width=W](base + i)
                var y = pb.unsafe_load[width=W](base + i)
                po.unsafe_store(base + i, max(x, y) - min(x, y))
                i += W
            for j in range(i, n):
                po[unsafe_offset=base + j] = max(pa[unsafe_offset=base + j], pb[unsafe_offset=base + j]) - min(pa[unsafe_offset=base + j], pb[unsafe_offset=base + j])
        else:
            comptime hi = Int64(Scalar[dt].MAX)
            for j in range(n):
                var d = abs(Int64(pa[unsafe_offset=base + j]) - Int64(pb[unsafe_offset=base + j]))
                po[unsafe_offset=base + j] = Scalar[dt](min(d, hi))

    par(row, a.rows)
    return o^


def _bw[op: Int, w: Int](x: SIMD[DType.uint8, w], y: SIMD[DType.uint8, w]) -> SIMD[DType.uint8, w]:
    comptime if op == 0:
        return x & y
    elif op == 1:
        return x | y
    elif op == 2:
        return x ^ y
    else:
        return ~x


def bitwise[op: Int](a: Mat, b: Mat) raises -> Mat[a.dt]:
    """op 0 and, 1 or, 2 xor, 3 not (b ignored). Works on the raw bytes, as cv2 does for any depth."""
    _need_same(a, b)
    var o = Mat[a.dt](a.rows, a.cols, a.ch)
    comptime W = simd_width_of[DType.uint8]()
    var n = a.cols * a.ch * size_of[Scalar[a.dt]]()
    var pa = a.ptr().unsafe_bitcast[UInt8]()
    var pb = b.ptr().unsafe_bitcast[UInt8]()
    var po = o.ptr().unsafe_bitcast[UInt8]()

    def row(r: Int) {imm}:
        var base = r * n
        var i = 0
        while i + W <= n:
            po.unsafe_store(base + i, _bw[op, W](pa.unsafe_load[width=W](base + i), pb.unsafe_load[width=W](base + i)))
            i += W
        for j in range(i, n):
            po.unsafe_store(base + j, _bw[op, 1](pa.unsafe_load[width=1](base + j), pb.unsafe_load[width=1](base + j)))

    par(row, a.rows)
    return o^


def flip[dt: DType](a: Mat[dt], code: Int) -> Mat[dt]:
    """cv2.flip: code 0 flips around the x axis (rows reversed), > 0 around the y axis, < 0 both."""
    var o = Mat[dt](a.rows, a.cols, a.ch)
    var ch = a.ch
    var pa = a.ptr()
    var po = o.ptr()
    var n = a.cols * ch

    def row(r: Int) {imm}:
        var src = (a.rows - 1 - r if code <= 0 else r) * n
        var dst = r * n
        if code == 0:
            for j in range(n):
                po[unsafe_offset=dst + j] = pa[unsafe_offset=src + j]
        else:
            for c in range(a.cols):
                for k in range(ch):
                    po[unsafe_offset=dst + c * ch + k] = pa[unsafe_offset=src + (a.cols - 1 - c) * ch + k]

    par(row, a.rows)
    return o^


def rotate[dt: DType](a: Mat[dt], code: Int) -> Mat[dt]:
    """cv2.rotate: 0 is 90 clockwise, 1 is 180, 2 is 90 counter-clockwise; code 3 is transpose."""
    comptime if dt == DType.uint8:
        if code == 3 and a.ch == 1:
            var o = Mat[dt](a.cols, a.rows, 1)
            var pa = a.ptr().unsafe_bitcast[UInt8]()
            var po = o.ptr().unsafe_bitcast[UInt8]()
            var rows = a.rows
            var cols = a.cols

            def band(b: Int) {imm}:
                var r0 = b * 8
                var r1 = r0 + 8
                if r1 > rows:
                    r1 = rows
                var cblk = (cols + 7) // 8
                for cb in range(cblk):
                    var c0 = cb * 8
                    var c1 = c0 + 8
                    if c1 > cols:
                        c1 = cols
                    var full = (r1 - r0 == 8) and (c1 - c0 == 8)
                    if full:
                        var rvec0 = (pa + (r0 + 0) * cols + c0).unsafe_load[width=8]()
                        var rvec1 = (pa + (r0 + 1) * cols + c0).unsafe_load[width=8]()
                        var rvec2 = (pa + (r0 + 2) * cols + c0).unsafe_load[width=8]()
                        var rvec3 = (pa + (r0 + 3) * cols + c0).unsafe_load[width=8]()
                        var rvec4 = (pa + (r0 + 4) * cols + c0).unsafe_load[width=8]()
                        var rvec5 = (pa + (r0 + 5) * cols + c0).unsafe_load[width=8]()
                        var rvec6 = (pa + (r0 + 6) * cols + c0).unsafe_load[width=8]()
                        var rvec7 = (pa + (r0 + 7) * cols + c0).unsafe_load[width=8]()
                        for i in range(8):
                            var out = SIMD[DType.uint8, 8](rvec0[i], rvec1[i], rvec2[i], rvec3[i], rvec4[i], rvec5[i], rvec6[i], rvec7[i])
                            (po + (c0 + i) * rows + r0).unsafe_store[width=8](out)
                    else:
                        for r in range(r0, r1):
                            var srow = r * cols
                            for c in range(c0, c1):
                                po[c * rows + r] = pa[srow + c]

            par(band, (rows + 7) // 8)
            return o^

    var swap = code != 1
    var o = Mat[dt](a.cols if swap else a.rows, a.rows if swap else a.cols, a.ch)
    var ch = a.ch
    var pa = a.ptr()
    var po = o.ptr()
    comptime T = 64

    def tile(t: Int) {imm}:
        var tc = (a.cols + T - 1) // T
        var r0 = (t // tc) * T
        var c0 = (t % tc) * T
        for r in range(r0, min(r0 + T, a.rows)):
            for c in range(c0, min(c0 + T, a.cols)):
                var orow: Int
                var ocol: Int
                if code == 0:
                    orow = c
                    ocol = a.rows - 1 - r
                elif code == 1:
                    orow = a.rows - 1 - r
                    ocol = a.cols - 1 - c
                elif code == 2:
                    orow = a.cols - 1 - c
                    ocol = r
                else:
                    orow = c
                    ocol = r
                var s = (r * a.cols + c) * ch
                var d = (orow * o.cols + ocol) * ch
                for k in range(ch):
                    po[unsafe_offset=d + k] = pa[unsafe_offset=s + k]

    par(tile, ((a.rows + T - 1) // T) * ((a.cols + T - 1) // T))
    return o^


def _cv[s: DType, d: DType, wide: DType, w: Int](x: SIMD[s, w], al: Scalar[wide], be: Scalar[wide]) -> SIMD[d, w]:
    var y = x.cast[wide]() * al + be
    comptime if d.is_integral():
        y = y.__round__().clamp(Scalar[wide](Scalar[d].MIN), Scalar[wide](Scalar[d].MAX))
    return y.cast[d]()


def convert[s: DType, d: DType](a: Mat[s], alpha: Float64, beta: Float64) -> Mat[d]:
    """cv2 convertTo: d = saturate(round_half_even(s * alpha + beta))."""
    var o = Mat[d](a.rows, a.cols, a.ch)
    var pa = a.ptr()
    var po = o.ptr()
    var n = a.cols * a.ch
    comptime wide = DType.float64 if (s == DType.float64 or s == DType.int32 or d == DType.int32 or d == DType.float64) else DType.float32
    comptime W = simd_width_of[wide]()
    var al = Scalar[wide](alpha)
    var be = Scalar[wide](beta)

    def row(r: Int) {imm}:
        var base = r * n
        var i = 0
        while i + W <= n:
            po.unsafe_store(base + i, _cv[s, d, wide, W](pa.unsafe_load[width=W](base + i), al, be))
            i += W
        for j in range(i, n):
            po.unsafe_store(base + j, _cv[s, d, wide, 1](pa.unsafe_load[width=1](base + j), al, be))

    par(row, a.rows)
    return o^


# ---- CLI --------------------------------------------------------------------------------------------------


def run_dt[dt: DType](mut a: Args) raises -> Bool:
    var m = load[dt](a.ins[0])
    if a.op == "flip":
        var o = Mat[dt](0, 0)
        while a.again():
            o = flip(m, a.i("code", 1))
        save(o, a.out)
    elif a.op == "rotate" or a.op == "transpose":
        var o = Mat[dt](0, 0)
        while a.again():
            o = rotate(m, a.i("code", 3 if a.op == "transpose" else 0))
        save(o, a.out)
    elif a.op == "absdiff":
        var m2 = load[dt](a.ins[1])
        var o = Mat[dt](0, 0)
        while a.again():
            o = absdiff(m, m2)
        save(o, a.out)
    elif a.op.startswith("bitwise_"):
        var m2 = load[dt](a.ins[1]) if a.op != "bitwise_not" else Mat[dt](copy=m)
        var o = Mat[dt](0, 0)
        while a.again():
            if a.op == "bitwise_and":
                o = bitwise[0](m, m2)
            elif a.op == "bitwise_or":
                o = bitwise[1](m, m2)
            elif a.op == "bitwise_xor":
                o = bitwise[2](m, m2)
            else:
                o = bitwise[3](m, m2)
        save(o, a.out)
    elif a.op == "convert":
        var to = a.i("to", 0)
        comptime for j in range(len(DTS)):
            if to == j:
                var o = Mat[DTS[j]](0, 0)
                while a.again():
                    o = convert[dt, DTS[j]](m, a.f("alpha", 1.0), a.f("beta", 0.0))
                save(o, a.out)
    else:
        return False
    return True


def run(mut a: Args) raises -> Bool:
    if not (a.op in ["flip", "rotate", "transpose", "absdiff", "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "convert"]):
        return False
    var code = peek_code(a.ins[0])
    comptime for i in range(len(DTS)):
        if code == i:
            return run_dt[DTS[i]](a)
    return False
