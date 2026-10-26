# Zobrist-hashed sparse octree over a 2^depth cube of cells, diff and static/moving split by hash.
# Cell hash = splitmix(Kx[x] ^ Ky[y] ^ Kz[z]) with a keyed 64-bit value per coordinate; a node hash is the
# XOR of its children, so equal hashes mean equal subtrees. The outer mix breaks the XOR-linearity of the
# raw keys: without it, the 4 corners of any axis-aligned rectangle of cells cancel and a change is invisible.
from std.builtin.sort import sort
from mojocv.mat import Mat
from mojocv.voxel import ckey, ckey_x, ckey_y, ckey_z, keys_to_cells

comptime GOLD: UInt64 = 0x9E3779B97F4A7C15


def splitmix(x: UInt64) -> UInt64:
    var z = x + GOLD
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def zkey(axis: Int, v: Int) -> UInt64:
    return splitmix((UInt64(axis + 1) << 40) + UInt64(v))


def cell_hash(x: Int, y: Int, z: Int) -> UInt64:
    return splitmix(zkey(0, x) ^ zkey(1, y) ^ zkey(2, z))


def morton(x: Int, y: Int, z: Int, depth: Int) -> Int:
    var m = 0
    for b in range(depth):
        m |= (((x >> b) & 1) << (3 * b + 2)) | (((y >> b) & 1) << (3 * b + 1)) | (((z >> b) & 1) << (3 * b))
    return m


struct Level(Movable):
    var code: List[Int]  # Morton code of each node, ascending
    var hash: List[UInt64]
    var lo: List[Int]  # children of node i are next-level nodes lo[i] .. hi[i]
    var hi: List[Int]

    def __init__(out self):
        self.code = List[Int]()
        self.hash = List[UInt64]()
        self.lo = List[Int]()
        self.hi = List[Int]()


struct Octree(Movable):
    var depth: Int
    var lv: List[Level]  # lv[0] is the root level, lv[depth] the cells

    def __init__(out self, cells: Mat[DType.int32], depth: Int) raises:
        if cells.cols != 3 or cells.ch != 1 or depth < 1 or depth > 19:
            raise Error("mojocv: octree wants an M x 3 int32 cell list and 1 <= depth <= 19")
        self.depth = depth
        var n = 1 << depth
        var codes = List[Int]()
        for i in range(cells.rows):
            var x = Int(cells.data[i * 3])
            var y = Int(cells.data[i * 3 + 1])
            var z = Int(cells.data[i * 3 + 2])
            if not (0 <= x < n and 0 <= y < n and 0 <= z < n):
                raise Error(String("mojocv: cell (", x, ",", y, ",", z, ") is outside the 2^", depth, " cube"))
            codes.append(morton(x, y, z, depth))
        sort(codes)
        self.lv = List[Level]()
        for _ in range(depth + 1):
            self.lv.append(Level())
        ref leaf = self.lv[depth]
        for c in codes:
            if len(leaf.code) == 0 or leaf.code[len(leaf.code) - 1] != c:
                leaf.code.append(c)
                leaf.hash.append(cell_hash(unmorton(c, 2), unmorton(c, 1), unmorton(c, 0)))
        for l in range(depth - 1, -1, -1):
            ref child = self.lv[l + 1]
            ref par = self.lv[l]
            for i in range(len(child.code)):
                var pc = child.code[i] >> 3
                if len(par.code) == 0 or par.code[len(par.code) - 1] != pc:
                    par.code.append(pc)
                    par.hash.append(0)
                    par.lo.append(i)
                    par.hi.append(i)
                var last = len(par.code) - 1
                par.hash[last] ^= child.hash[i]
                par.hi[last] = i + 1

    def root_hash(self) -> UInt64:
        return self.lv[0].hash[0] if len(self.lv[0].hash) > 0 else 0

    def cells(self) -> Int:
        return len(self.lv[self.depth].code)


def unmorton(m: Int, axis: Int) -> Int:
    """Coordinate of axis (2 = x, 1 = y, 0 = z) from a Morton code."""
    var v = 0
    var b = 0
    while (m >> (3 * b)) != 0:
        v |= ((m >> (3 * b + axis)) & 1) << b
        b += 1
    return v


struct Walk(Movable):
    """Collected result: packed (cell key << 1 | label) entries, label 1 = static, 0 = moving; visited = node
    groups compared."""
    var keys: List[Int]
    var visited: Int

    def __init__(out self):
        self.keys = List[Int]()
        self.visited = 0


def _emit_subtree(t: Octree, lvl: Int, i: Int, label: Int, mut w: Walk) raises:
    if lvl == t.depth:
        var c = t.lv[lvl].code[i]
        w.keys.append((ckey(unmorton(c, 2), unmorton(c, 1), unmorton(c, 0)) << 1) | label)
        return
    for j in range(t.lv[lvl].lo[i], t.lv[lvl].hi[i]):
        _emit_subtree(t, lvl + 1, j, label, w)


def _walk(ts: List[Octree], lvl: Int, nodes: List[Int], emit_static: Bool, mut w: Walk) raises:
    w.visited += 1
    var all_present = True
    var all_equal = True
    var first = -1
    for t in range(len(ts)):
        if nodes[t] < 0:
            all_present = False
        elif first < 0:
            first = t
        elif ts[t].lv[lvl].hash[nodes[t]] != ts[first].lv[lvl].hash[nodes[first]]:
            all_equal = False
    if all_present and all_equal:
        if emit_static:
            _emit_subtree(ts[0], lvl, nodes[0], 1, w)
        return
    var depth = ts[0].depth
    if lvl == depth:
        _emit_subtree(ts[first], lvl, nodes[first], 0, w)
        return
    var base = ts[first].lv[lvl].code[nodes[first]] << 3
    var cur = List[Int]()
    for t in range(len(ts)):
        cur.append(ts[t].lv[lvl].lo[nodes[t]] if nodes[t] >= 0 else 0)
    for c in range(8):
        var kids = List[Int]()
        var any = False
        for t in range(len(ts)):
            var k = -1
            if nodes[t] >= 0 and cur[t] < ts[t].lv[lvl].hi[nodes[t]] and ts[t].lv[lvl + 1].code[cur[t]] == base + c:
                k = cur[t]
                cur[t] += 1
                any = True
            kids.append(k)
        if any:
            _walk(ts, lvl + 1, kids, emit_static, w)


def walk(ts: List[Octree], emit_static: Bool) raises -> Walk:
    """Compare the trees of a frame window top-down, skipping every subtree whose hash is the same in all
    frames. Cells present in all frames are static, cells present in some are moving."""
    var w = Walk()
    var roots = List[Int]()
    var any = False
    for t in range(len(ts)):
        roots.append(0 if len(ts[t].lv[0].code) > 0 else -1)
        any = any or roots[t] >= 0
    if any:
        _walk(ts, 0, roots, emit_static, w)
    sort(w.keys)
    return w^


def walk_cells(w: Walk, labels: Bool) -> Mat[DType.int32]:
    """M x 3 cells, or M x 4 with the static label, in (x, y, z) order."""
    var m = keys_to_cells(w.keys, 1)
    if not labels:
        return m^
    var out = Mat[DType.int32](len(w.keys), 4)
    for i in range(len(w.keys)):
        for a in range(3):
            out.data[i * 4 + a] = m.data[i * 3 + a]
        out.data[i * 4 + 3] = Int32(w.keys[i] & 1)
    return out^
