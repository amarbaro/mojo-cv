# Mat: an owned image or array, rows x cols x ch, row-major and interleaved (the layout cv2 uses), one dtype
# fixed at compile time. Ops are generic over dt; the file format below carries dt so both sides agree.
from std.sys.info import size_of

comptime MAGIC = "MCV1"


def dt_code[dt: DType]() -> Int:
    comptime if dt == DType.uint8:
        return 0
    elif dt == DType.float32:
        return 1
    elif dt == DType.int32:
        return 2
    elif dt == DType.float64:
        return 3
    else:
        comptime assert dt == DType.int16, "mojocv: dtype must be uint8, int16, int32, float32 or float64"
        return 4


struct Mat[dt: DType](Movable):
    var rows: Int
    var cols: Int
    var ch: Int
    var data: List[Scalar[Self.dt]]

    def __init__(out self, rows: Int, cols: Int, ch: Int = 1):
        self.rows = rows
        self.cols = cols
        self.ch = ch
        self.data = List[Scalar[Self.dt]](length=rows * cols * ch, fill=0)

    def __init__(out self, rows: Int, cols: Int, ch: Int, zero: Bool):
        self.rows = rows
        self.cols = cols
        self.ch = ch
        if zero:
            self.data = List[Scalar[Self.dt]](length=rows * cols * ch, fill=0)
        else:
            self.data = List[Scalar[Self.dt]](unsafe_uninit_length=rows * cols * ch)

    def __init__(out self, *, copy: Self):
        self.rows = copy.rows
        self.cols = copy.cols
        self.ch = copy.ch
        self.data = copy.data.copy()

    def size(self) -> Int:
        return self.rows * self.cols * self.ch

    def ptr(self) -> MutPointer[Scalar[Self.dt], MutAnyOrigin]:
        return MutPointer[Scalar[Self.dt], MutAnyOrigin](unsafe_from_address=Int(self.data.unsafe_ptr()))

    def at(self, r: Int, c: Int, k: Int = 0) -> Scalar[Self.dt]:
        return self.data[(r * self.cols + c) * self.ch + k]

    def set(mut self, r: Int, c: Int, k: Int, v: Scalar[Self.dt]):
        self.data[(r * self.cols + c) * self.ch + k] = v

    def same_shape(self, o: Mat) -> Bool:
        return self.rows == o.rows and self.cols == o.cols and self.ch == o.ch


def _u32(b: List[UInt8], at: Int) -> Int:
    return Int(b[at]) | Int(b[at + 1]) << 8 | Int(b[at + 2]) << 16 | Int(b[at + 3]) << 24


def load[dt: DType](path: String) raises -> Mat[dt]:
    """Read an .mcv file: "MCV1", u32 rows, u32 cols, u16 ch, u16 dtype code, then the raw little-endian data."""
    var b = open(path, "r").read_bytes()
    if len(b) < 16 or String(StringSlice(unsafe_from_utf8=Span(b)[:4])) != MAGIC:
        raise Error("mojocv: not an MCV1 file: " + path)
    var m = Mat[dt](_u32(b, 4), _u32(b, 8), Int(b[12]) | Int(b[13]) << 8)
    var code = Int(b[14]) | Int(b[15]) << 8
    if code != dt_code[dt]():
        raise Error(String("mojocv: ", path, " holds dtype code ", code, ", wanted ", dt_code[dt]()))
    var n = m.size() * size_of[Scalar[dt]]()
    if len(b) != 16 + n:
        raise Error(String("mojocv: ", path, " has ", len(b) - 16, " data bytes, wanted ", n))
    var dst = m.ptr().bitcast[UInt8]()
    for i in range(n):
        dst[i] = b[16 + i]
    return m^


def save[dt: DType](m: Mat[dt], path: String) raises:
    var h = List[UInt8](length=16, fill=0)
    for i in range(4):
        h[i] = MAGIC.as_bytes()[i]
    for i in range(4):
        h[4 + i] = UInt8((m.rows >> (8 * i)) & 255)
        h[8 + i] = UInt8((m.cols >> (8 * i)) & 255)
    h[12] = UInt8(m.ch & 255)
    h[13] = UInt8(m.ch >> 8)
    h[14] = UInt8(dt_code[dt]())
    with open(path, "w") as f:
        f.write_bytes(Span(h))
        f.write_bytes(Span(unsafe_ptr=m.ptr().bitcast[UInt8](), length=m.size() * size_of[Scalar[dt]]()))
