# Voxel boundary to a watertight triangle mesh. Every face between an occupied and an empty cell becomes a quad,
# oriented outward (counter-clockwise seen from outside), split into two triangles. Lattice corners are welded where
# the surface is one sheet: two quads share a corner copy when they are joined along a lattice edge, and at an edge
# where two cells touch only diagonally (four quads) each cell keeps its own pair. So a corner where two cells meet
# at a point or along an edge becomes two vertices, and the result is a closed 2-manifold: every edge lies on exactly
# two triangles, which `edges_euler` checks and raises on.
# vertices: V x 3 float64 (lattice corners), faces: F x 3 int32.
from std.collections import Dict, Set
from std.builtin.sort import sort
from mojocv.mat import Mat
from mojocv.voxel import ckey, ckey_x, ckey_y, ckey_z


struct Mesh(Movable):
    var verts: Mat[DType.float64]
    var faces: Mat[DType.int32]

    def __init__(out self, var verts: Mat[DType.float64], var faces: Mat[DType.int32]):
        self.verts = verts^
        self.faces = faces^


def _find(mut parent: List[Int], x: Int) -> Int:
    var r = x
    while parent[r] != r:
        r = parent[r]
    var c = x
    while parent[c] != r:
        var nx = parent[c]
        parent[c] = r
        c = nx
    return r


def _union(mut parent: List[Int], a: Int, b: Int):
    var ra = _find(parent, a)
    var rb = _find(parent, b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def voxel_mesh(cells: Mat[DType.int32]) raises -> Mesh:
    if cells.cols != 3 or cells.ch != 1:
        raise Error("mojocv: mesh wants an M x 3 int32 cell list")
    var occ = Set[Int]()
    for i in range(cells.rows):
        occ.add(ckey(Int(cells.data[i * 3]), Int(cells.data[i * 3 + 1]), Int(cells.data[i * 3 + 2])))
    var keys = List[Int]()
    for k in occ:
        keys.append(k)
    sort(keys)
    # quads: 4 corner lattice points each, owner cell, corner order counter-clockwise from outside
    var qpt = List[Int]()  # 3 ints per corner, 12 per quad
    var qown = List[Int]()
    for k in keys:
        var c = List[Int](length=3, fill=0)
        c[0] = ckey_x(k)
        c[1] = ckey_y(k)
        c[2] = ckey_z(k)
        for a in range(3):
            for dirn in range(2):
                var nb = c.copy()
                nb[a] += 1 if dirn == 1 else -1
                if ckey(nb[0], nb[1], nb[2]) in occ:
                    continue
                var b = (a + 1) % 3
                var d = (a + 2) % 3
                for corner in range(4):
                    var t1 = 1 if corner == 1 or corner == 2 else 0  # (0,0) (1,0) (1,1) (0,1) in (b, d) ...
                    var t2 = 1 if corner >= 2 else 0
                    var ub = t1 if dirn == 1 else t2  # ... and the -a face runs the other way round
                    var ud = t2 if dirn == 1 else t1
                    var p = c.copy()
                    p[a] += dirn
                    p[b] += ub
                    p[d] += ud
                    qpt.append(p[0])
                    qpt.append(p[1])
                    qpt.append(p[2])
                qown.append(k)
    var nq = len(qown)
    # edges: key = lattice point of the lower end * 3 + axis; value = quad * 4 + edge index
    var edges = Dict[Int, List[Int]]()
    for q in range(nq):
        for e in range(4):
            var i0 = (q * 4 + e) * 3
            var i1 = (q * 4 + (e + 1) % 4) * 3
            var lo = List[Int](length=3, fill=0)
            var axis = 0
            for a in range(3):
                lo[a] = min(qpt[i0 + a], qpt[i1 + a])
                if qpt[i0 + a] != qpt[i1 + a]:
                    axis = a
            var ek = ckey(lo[0], lo[1], lo[2]) * 3 + axis
            if ek not in edges:
                edges[ek] = List[Int]()
            edges[ek].append(q * 4 + e)
    var cross = Set[Int]()
    var m = Mesh(Mat[DType.float64](0, 3), Mat[DType.int32](0, 3))
    for _ in range(24):
        var bad = _assemble(qpt, qown, edges, cross, m)
        if len(bad) == 0:
            return m^
        for e in bad:
            if e in cross:
                cross.remove(e)
            else:
                cross.add(e)
    raise Error("mojocv: mesh could not resolve a pinched edge into a manifold")


def _assemble(
    qpt: List[Int], qown: List[Int], edges: Dict[Int, List[Int]], cross: Set[Int], mut m: Mesh
) raises -> List[Int]:
    """Weld corner copies along the pairings; the second result lists 4-quad edges still on four triangles."""
    var nq = len(qown)
    var parent = List[Int]()
    for i in range(nq * 4):
        parent.append(i)
    for item in edges.items():
        ref fs = item.value
        if len(fs) != 2 and len(fs) != 4:
            raise Error("mojocv: mesh found a lattice edge with an odd number of boundary faces")
        var pairs = List[Int]()  # indices into fs, two per pair
        if len(fs) == 2:
            pairs.append(0)
            pairs.append(1)
        else:
            # Two cells touch diagonally. Each quad runs along the edge forwards or backwards; a pairing must join a
            # forward quad to a backward one. Same-owner pairing keeps each cell's corner whole; the cross pairing is
            # the other orientable choice and is used where the first leaves two sheets sharing both end vertices.
            var o0 = qown[fs[0] // 4]
            for x in range(4):
                for y in range(x + 1, 4):
                    var same = qown[fs[x] // 4] == qown[fs[y] // 4]
                    if (item.key in cross) != same and _forward(qpt, fs[x]) != _forward(qpt, fs[y]):
                        pairs.append(x)
                        pairs.append(y)
            _ = o0
        for pr in range(len(pairs) // 2):
            var qa = fs[pairs[2 * pr]] // 4
            var ea = fs[pairs[2 * pr]] % 4
            var qb = fs[pairs[2 * pr + 1]] // 4
            var eb = fs[pairs[2 * pr + 1]] % 4
            for s in range(2):  # join the copies of each end point
                var ia = (qa * 4 + (ea + s) % 4) * 3
                for t in range(2):
                    var ib = (qb * 4 + (eb + t) % 4) * 3
                    if qpt[ia] == qpt[ib] and qpt[ia + 1] == qpt[ib + 1] and qpt[ia + 2] == qpt[ib + 2]:
                        _union(parent, qa * 4 + (ea + s) % 4, qb * 4 + (eb + t) % 4)
    var vid = Dict[Int, Int]()
    var verts = Mat[DType.float64](0, 3)
    var faces = Mat[DType.int32](0, 3)
    var corner_vertex = List[Int]()
    for i in range(nq * 4):
        var r = _find(parent, i)
        if r not in vid:
            vid[r] = verts.rows
            for a in range(3):
                verts.data.append(Float64(qpt[i * 3 + a]))
            verts.rows += 1
        corner_vertex.append(vid[r])
    for q in range(nq):
        for tri in range(2):
            faces.data.append(Int32(corner_vertex[q * 4]))
            faces.data.append(Int32(corner_vertex[q * 4 + 1 + tri]))
            faces.data.append(Int32(corner_vertex[q * 4 + 2 + tri]))
            faces.rows += 1
    m = Mesh(verts^, faces^)
    var count = Dict[Int, Int]()
    var nv = m.verts.rows
    for f in range(m.faces.rows):
        for e in range(3):
            var a = Int(m.faces.data[f * 3 + e])
            var b = Int(m.faces.data[f * 3 + (e + 1) % 3])
            var k = min(a, b) * nv + max(a, b)
            count[k] = (count[k] if k in count else 0) + 1
    var bad = List[Int]()
    for item in edges.items():
        if len(item.value) == 4:
            for x in range(4):  # the vertex pair of each of the four quad edges
                var f = item.value[x]
                var a = corner_vertex[f]
                var b = corner_vertex[(f // 4) * 4 + (f % 4 + 1) % 4]
                if count[min(a, b) * nv + max(a, b)] != 2:
                    bad.append(item.key)
                    break
    return bad^


def _forward(qpt: List[Int], qe: Int) -> Bool:
    """True when quad edge qe (quad * 4 + edge) runs towards larger coordinates."""
    var i0 = qe * 3
    var i1 = ((qe // 4) * 4 + (qe % 4 + 1) % 4) * 3
    for a in range(3):
        if qpt[i0 + a] != qpt[i1 + a]:
            return qpt[i1 + a] > qpt[i0 + a]
    return True


def edges_euler(m: Mesh) raises -> Tuple[Int, Int]:
    """(edges, euler characteristic V - E + F); raises unless every edge lies on exactly two triangles."""
    var count = Dict[Int, Int]()
    var nv = m.verts.rows
    for f in range(m.faces.rows):
        for e in range(3):
            var a = Int(m.faces.data[f * 3 + e])
            var b = Int(m.faces.data[f * 3 + (e + 1) % 3])
            var k = min(a, b) * nv + max(a, b)
            count[k] = (count[k] if k in count else 0) + 1
    for item in count.items():
        if item.value != 2:
            raise Error(String("mojocv: mesh is not watertight: an edge lies on ", item.value, " triangles"))
    return (len(count), nv - len(count) + m.faces.rows)
