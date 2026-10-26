# Point clouds <-> integer cells. A cloud is a Mat[float64] N x 3; cells are Mat[int32] M x 3.
from std.math import floor
from std.builtin.sort import sort
from mojocv.mat import Mat

comptime OFF = 1 << 19  # cell coordinates are packed into 20 bits each: |cell| < 524288


def ckey(x: Int, y: Int, z: Int) raises -> Int:
    """Pack a cell into one Int whose numeric order is the lexicographic (x, y, z) order."""
    if not (-OFF <= x < OFF and -OFF <= y < OFF and -OFF <= z < OFF):
        raise Error(String("mojocv: cell (", x, ",", y, ",", z, ") is outside +-", OFF))
    return ((x + OFF) << 40) | ((y + OFF) << 20) | (z + OFF)


def ckey_x(k: Int) -> Int:
    return (k >> 40) - OFF


def ckey_y(k: Int) -> Int:
    return ((k >> 20) & (2 * OFF - 1)) - OFF


def ckey_z(k: Int) -> Int:
    return (k & (2 * OFF - 1)) - OFF


def voxelize(p: Mat[DType.float64], size: Float64) raises -> Mat[DType.float64]:
    """N x 6: the integer cell of each point, then its residual in that cell, in [0, 1) cell units."""
    if p.cols != 3 or p.ch != 1 or size <= 0:
        raise Error("mojocv: voxelize wants an N x 3 float64 cloud and size > 0")
    var v = Mat[DType.float64](p.rows, 6)
    for i in range(p.rows):
        for a in range(3):
            var s = p.data[i * 3 + a] / size
            var c = floor(s)
            v.data[i * 6 + a] = c
            v.data[i * 6 + 3 + a] = s - c
    return v^


def unvoxelize(v: Mat[DType.float64], size: Float64) raises -> Mat[DType.float64]:
    if v.cols != 6 or v.ch != 1:
        raise Error("mojocv: unvoxelize wants the N x 6 output of voxelize")
    var p = Mat[DType.float64](v.rows, 3)
    for i in range(v.rows):
        for a in range(3):
            p.data[i * 3 + a] = (v.data[i * 6 + a] + v.data[i * 6 + 3 + a]) * size
    return p^


def occupied(p: Mat[DType.float64], size: Float64) raises -> Mat[DType.int32]:
    """The distinct occupied cells of a cloud, sorted by (x, y, z)."""
    var v = voxelize(p, size)
    var keys = List[Int]()
    for i in range(v.rows):
        keys.append(ckey(Int(v.data[i * 6]), Int(v.data[i * 6 + 1]), Int(v.data[i * 6 + 2])))
    sort(keys)
    return keys_to_cells(keys, 0)


def keys_to_cells(keys: List[Int], shift: Int) -> Mat[DType.int32]:
    """Unique sorted packed keys to M x 3 cells; `shift` drops low bits (a label packed under the key)."""
    var out = Mat[DType.int32](0, 3)
    var last = -1
    for k0 in keys:
        var k = k0 >> shift
        if k != last:
            out.data.append(Int32(ckey_x(k)))
            out.data.append(Int32(ckey_y(k)))
            out.data.append(Int32(ckey_z(k)))
            last = k
    out.rows = len(out.data) // 3
    return out^
