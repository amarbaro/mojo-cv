# GLB (glTF 2.0 binary): a writer for one triangle mesh and a reader for any single-mesh GLB.
# Layout written: 12-byte header, JSON chunk (space padded to 4), BIN chunk (zero padded to 4) holding
# POSITION (float32 VEC3, min/max set), NORMAL (area-weighted vertex normals), optional COLOR_0 (float32 VEC3 in
# [0, 1]) and uint32 indices; one scene, one node, one mesh, one TRIANGLES primitive.
# The reader takes meshes[0].primitives[0]: POSITION, optional NORMAL and COLOR_0 (float32), indices of uint8, uint16
# or uint32 (none = sequential), byteStride honoured. It validates every length, offset, count, component type and
# index and raises "mojocv: glb: <what>" instead of reading out of range.
from std.math import sqrt
from std.memory import bitcast
from mojocv.mat import Mat


struct Glb(Movable):
    var verts: Mat[DType.float64]
    var faces: Mat[DType.int32]
    var normals: Mat[DType.float64]  # 0 rows when absent
    var colors: Mat[DType.float64]  # 0 rows when absent

    def __init__(out self):
        self.verts = Mat[DType.float64](0, 3)
        self.faces = Mat[DType.int32](0, 3)
        self.normals = Mat[DType.float64](0, 3)
        self.colors = Mat[DType.float64](0, 3)


def _put_u32(mut b: List[UInt8], v: Int):
    for i in range(4):
        b.append(UInt8((v >> (8 * i)) & 255))


def _put_f32(mut b: List[UInt8], v: Float64):
    _put_u32(b, Int(bitcast[DType.uint32, 1](SIMD[DType.float32, 1](Float32(v)))))


def _rd_u32(b: List[UInt8], at: Int) -> Int:
    return Int(b[at]) | Int(b[at + 1]) << 8 | Int(b[at + 2]) << 16 | Int(b[at + 3]) << 24


def _rd_f32(b: List[UInt8], at: Int) -> Float64:
    return Float64(bitcast[DType.float32, 1](SIMD[DType.uint32, 1](UInt32(_rd_u32(b, at))))[0])


def _f32_text(v: Float64) -> String:
    return String(Float64(Float32(v)))


def write_glb(
    v: Mat[DType.float64], f: Mat[DType.int32], colors: Mat[DType.float64], path: String
) raises:
    var nv = v.rows
    var nf = f.rows
    if v.cols != 3 or f.cols != 3 or nv == 0 or nf == 0 or v.ch != 1 or f.ch != 1:
        raise Error("mojocv: glb: write wants V x 3 vertices and F x 3 faces, both non-empty")
    var has_color = colors.rows > 0
    if has_color and (colors.rows != nv or colors.cols != 3):
        raise Error("mojocv: glb: colors must be V x 3, one per vertex")
    var lo = List[Float64](length=3, fill=1e300)
    var hi = List[Float64](length=3, fill=-1e300)
    for i in range(nv):
        for a in range(3):
            var x = Float64(Float32(v.data[i * 3 + a]))
            if x != x or abs(x) > 3.0e38:
                raise Error("mojocv: glb: vertex coordinates must be finite float32 values")
            lo[a] = min(lo[a], x)
            hi[a] = max(hi[a], x)
    for k in range(nf * 3):
        if Int(f.data[k]) < 0 or Int(f.data[k]) >= nv:
            raise Error(String("mojocv: glb: face index ", Int(f.data[k]), " outside 0..", nv - 1))
    var nrm = List[Float64](length=nv * 3, fill=0.0)
    for t in range(nf):
        var i0 = Int(f.data[t * 3])
        var i1 = Int(f.data[t * 3 + 1])
        var i2 = Int(f.data[t * 3 + 2])
        var e1 = List[Float64]()
        var e2 = List[Float64]()
        for a in range(3):
            e1.append(v.data[i1 * 3 + a] - v.data[i0 * 3 + a])
            e2.append(v.data[i2 * 3 + a] - v.data[i0 * 3 + a])
        var c0 = e1[1] * e2[2] - e1[2] * e2[1]  # not normalised: area weighting
        var c1 = e1[2] * e2[0] - e1[0] * e2[2]
        var c2 = e1[0] * e2[1] - e1[1] * e2[0]
        for k in range(3):
            var idx = Int(f.data[t * 3 + k])
            nrm[idx * 3] += c0
            nrm[idx * 3 + 1] += c1
            nrm[idx * 3 + 2] += c2
    var bin = List[UInt8]()
    for i in range(nv * 3):
        _put_f32(bin, v.data[i])
    for i in range(nv):
        var len = sqrt(nrm[i * 3] ** 2 + nrm[i * 3 + 1] ** 2 + nrm[i * 3 + 2] ** 2)
        for a in range(3):
            _put_f32(bin, nrm[i * 3 + a] / len if len > 0.0 else 0.0)
    if has_color:
        for i in range(nv * 3):
            _put_f32(bin, colors.data[i])
    var idx_off = len(bin)
    for k in range(nf * 3):
        _put_u32(bin, Int(f.data[k]))
    var vb = nv * 12
    var views = String('{"buffer":0,"byteOffset":0,"byteLength":', vb, ',"target":34962}')
    views += String(',{"buffer":0,"byteOffset":', vb, ',"byteLength":', vb, ',"target":34962}')
    var attrs = String('"POSITION":0,"NORMAL":1')
    var accs = String(
        '{"bufferView":0,"componentType":5126,"count":', nv, ',"type":"VEC3","min":[', _f32_text(lo[0]), ",",
        _f32_text(lo[1]), ",", _f32_text(lo[2]), '],"max":[', _f32_text(hi[0]), ",", _f32_text(hi[1]), ",",
        _f32_text(hi[2]), "]}",
    )
    accs += String(',{"bufferView":1,"componentType":5126,"count":', nv, ',"type":"VEC3"}')
    var iview = 2
    if has_color:
        views += String(',{"buffer":0,"byteOffset":', 2 * vb, ',"byteLength":', vb, ',"target":34962}')
        accs += String(',{"bufferView":2,"componentType":5126,"count":', nv, ',"type":"VEC3"}')
        attrs += ',"COLOR_0":2'
        iview = 3
    views += String(',{"buffer":0,"byteOffset":', idx_off, ',"byteLength":', nf * 12, ',"target":34963}')
    accs += String(',{"bufferView":', iview, ',"componentType":5125,"count":', nf * 3, ',"type":"SCALAR"}')
    var json = String(
        '{"asset":{"version":"2.0","generator":"mojo-cv"},"scene":0,"scenes":[{"nodes":[0]}],"nodes":[{"mesh":0}],',
        '"meshes":[{"primitives":[{"attributes":{', attrs, '},"indices":', iview, ',"mode":4}]}],',
        '"buffers":[{"byteLength":', len(bin), '}],"bufferViews":[', views, '],"accessors":[', accs, "]}",
    )
    var jb = List[UInt8]()
    for byte in json.as_bytes():
        jb.append(byte)
    while len(jb) % 4 != 0:
        jb.append(32)
    while len(bin) % 4 != 0:
        bin.append(0)
    var out = List[UInt8]()
    _put_u32(out, 0x46546C67)  # "glTF"
    _put_u32(out, 2)
    _put_u32(out, 12 + 8 + len(jb) + 8 + len(bin))
    _put_u32(out, len(jb))
    _put_u32(out, 0x4E4F534A)  # "JSON"
    out.extend(jb^)
    _put_u32(out, len(bin))
    _put_u32(out, 0x004E4942)  # "BIN\0"
    out.extend(bin^)
    with open(path, "w") as fh:
        fh.write_bytes(Span(out))


# ---- a small arena JSON parser, enough for glTF (objects, arrays, strings, numbers, literals) ----
struct Js(Movable):
    var kind: List[Int]  # 0 null, 1 bool, 2 number, 3 string, 4 array, 5 object
    var num: List[Float64]
    var text: List[String]  # string value
    var key: List[String]  # member name when the parent is an object
    var first: List[Int]
    var next: List[Int]

    def __init__(out self):
        self.kind = List[Int]()
        self.num = List[Float64]()
        self.text = List[String]()
        self.key = List[String]()
        self.first = List[Int]()
        self.next = List[Int]()

    def add(mut self, kind: Int) -> Int:
        self.kind.append(kind)
        self.num.append(0.0)
        self.text.append(String(""))
        self.key.append(String(""))
        self.first.append(-1)
        self.next.append(-1)
        return len(self.kind) - 1

    def get(self, node: Int, name: String) -> Int:
        if node < 0 or self.kind[node] != 5:
            return -1
        var c = self.first[node]
        while c >= 0:
            if self.key[c] == name:
                return c
            c = self.next[c]
        return -1

    def at(self, node: Int, i: Int) -> Int:
        if node < 0 or self.kind[node] != 4:
            return -1
        var c = self.first[node]
        var k = 0
        while c >= 0 and k < i:
            c = self.next[c]
            k += 1
        return c

    def count(self, node: Int) -> Int:
        var n = 0
        var c = self.first[node] if node >= 0 else -1
        while c >= 0:
            n += 1
            c = self.next[c]
        return n


def _skip(b: List[UInt8], mut p: Int):
    while p < len(b) and (b[p] == 32 or b[p] == 9 or b[p] == 10 or b[p] == 13):
        p += 1


def _jstr(b: List[UInt8], mut p: Int) raises -> String:
    p += 1  # opening quote
    var out = List[UInt8]()
    while True:
        if p >= len(b):
            raise Error("mojocv: glb: unterminated JSON string")
        var c = b[p]
        p += 1
        if c == 34:
            break
        if c == 92:
            if p >= len(b):
                raise Error("mojocv: glb: bad JSON escape")
            var e = b[p]
            p += 1
            if e == 110:
                out.append(10)
            elif e == 116:
                out.append(9)
            elif e == 117:
                if p + 4 > len(b):
                    raise Error("mojocv: glb: bad JSON \\u escape")
                var cp = 0
                for _ in range(4):
                    var h = Int(b[p])
                    if 48 <= h <= 57:
                        cp = cp * 16 + h - 48
                    elif 97 <= h <= 102:
                        cp = cp * 16 + h - 87
                    elif 65 <= h <= 70:
                        cp = cp * 16 + h - 55
                    else:
                        raise Error("mojocv: glb: bad hex digit in \\u escape")
                    p += 1
                if cp < 128:
                    out.append(UInt8(cp))
                else:
                    out.append(63)
            else:
                out.append(e)
        else:
            out.append(c)
    return String(StringSlice(unsafe_from_utf8=Span(out)))


def _jval(b: List[UInt8], mut p: Int, mut js: Js, depth: Int) raises -> Int:
    if depth > 64:
        raise Error("mojocv: glb: JSON nested deeper than 64")
    _skip(b, p)
    if p >= len(b):
        raise Error("mojocv: glb: JSON ends early")
    var c = b[p]
    if c == 123 or c == 91:  # { [
        var node = js.add(5 if c == 123 else 4)
        p += 1
        var last = -1
        _skip(b, p)
        if p < len(b) and Int(b[p]) == (125 if c == 123 else 93):
            p += 1
            return node
        while True:
            _skip(b, p)
            var name = String("")
            if c == 123:
                if p >= len(b) or b[p] != 34:
                    raise Error(String("mojocv: glb: JSON object key expected at byte ", p))
                name = _jstr(b, p)
                _skip(b, p)
                if p >= len(b) or b[p] != 58:
                    raise Error(String("mojocv: glb: ':' expected at byte ", p))
                p += 1
            var child = _jval(b, p, js, depth + 1)
            js.key[child] = name
            if last < 0:
                js.first[node] = child
            else:
                js.next[last] = child
            last = child
            _skip(b, p)
            if p >= len(b):
                raise Error("mojocv: glb: JSON ends early")
            if b[p] == 44:
                p += 1
                continue
            if Int(b[p]) == (125 if c == 123 else 93):
                p += 1
                return node
            raise Error(String("mojocv: glb: ',' or closing bracket expected at byte ", p))
    if c == 34:
        var n = js.add(3)
        js.text[n] = _jstr(b, p)
        return n
    var q = p
    while q < len(b) and b[q] != 44 and b[q] != 93 and b[q] != 125 and b[q] != 32 and b[q] != 10 and b[q] != 13:
        q += 1
    var tok = String(StringSlice(unsafe_from_utf8=Span(b)[p:q]))
    p = q
    if tok == "true" or tok == "false":
        var n = js.add(1)
        js.num[n] = 1.0 if tok == "true" else 0.0
        return n
    if tok == "null":
        return js.add(0)
    var n = js.add(2)
    try:
        js.num[n] = Float64(tok)
    except:
        raise Error(String("mojocv: glb: bad JSON token '", tok, "' at byte ", p))
    return n


def _need(js: Js, node: Int, what: String) raises -> Int:
    if node < 0:
        raise Error(String("mojocv: glb: missing ", what))
    return node


def _accessor(
    js: Js, bin: List[UInt8], acc: Int, ncomp: Int, mut out: List[Float64], want_float: Bool
) raises -> Int:
    """Decode accessor index `acc` into `out` (count * ncomp values); returns count."""
    var a = _need(js, js.at(js.get(0, "accessors"), acc), String("accessor ") + String(acc))
    var count = Int(js.num[_need(js, js.get(a, "count"), "accessor count")])
    var ctype = Int(js.num[_need(js, js.get(a, "componentType"), "accessor componentType")])
    var type_name = js.text[_need(js, js.get(a, "type"), "accessor type")]
    var bvi = js.get(a, "bufferView")
    var bv = _need(js, js.at(js.get(0, "bufferViews"), Int(js.num[_need(js, bvi, "accessor bufferView")])), "bufferView")
    var csize = 1 if ctype == 5121 else (2 if ctype == 5123 else 4)
    if ctype not in (5121, 5123, 5125, 5126):
        raise Error(String("mojocv: glb: unsupported componentType ", ctype))
    if (want_float and ctype != 5126) or (not want_float and ctype == 5126):
        raise Error(String("mojocv: glb: componentType ", ctype, " not allowed for this attribute"))
    var tcomp = 1 if type_name == "SCALAR" else (3 if type_name == "VEC3" else 0)
    if tcomp != ncomp:
        raise Error(String("mojocv: glb: accessor type ", type_name, " where ", ncomp, " components are needed"))
    var off = Int(js.num[js.get(bv, "byteOffset")]) if js.get(bv, "byteOffset") >= 0 else 0
    off += Int(js.num[js.get(a, "byteOffset")]) if js.get(a, "byteOffset") >= 0 else 0
    var blen = Int(js.num[_need(js, js.get(bv, "byteLength"), "bufferView byteLength")])
    var stride = Int(js.num[js.get(bv, "byteStride")]) if js.get(bv, "byteStride") >= 0 else ncomp * csize
    if count < 1 or stride < ncomp * csize or off < 0 or off + (count - 1) * stride + ncomp * csize > off - (
        Int(js.num[js.get(a, "byteOffset")]) if js.get(a, "byteOffset") >= 0 else 0
    ) + blen or off + (count - 1) * stride + ncomp * csize > len(bin):
        raise Error(String("mojocv: glb: accessor ", acc, " reads outside its bufferView or the BIN chunk"))
    for i in range(count):
        for c in range(ncomp):
            var at = off + i * stride + c * csize
            if ctype == 5126:
                out.append(_rd_f32(bin, at))
            elif ctype == 5125:
                out.append(Float64(_rd_u32(bin, at)))
            elif ctype == 5123:
                out.append(Float64(Int(bin[at]) | Int(bin[at + 1]) << 8))
            else:
                out.append(Float64(bin[at]))
    return count


def read_glb(path: String) raises -> Glb:
    var b = open(path, "r").read_bytes()
    if len(b) < 20 or _rd_u32(b, 0) != 0x46546C67:
        raise Error("mojocv: glb: not a GLB file (magic)")
    if _rd_u32(b, 4) != 2:
        raise Error(String("mojocv: glb: version ", _rd_u32(b, 4), ", only 2 is supported"))
    if _rd_u32(b, 8) != len(b):
        raise Error(String("mojocv: glb: header length ", _rd_u32(b, 8), " != file size ", len(b)))
    var jl = _rd_u32(b, 12)
    if _rd_u32(b, 16) != 0x4E4F534A or jl % 4 != 0 or 20 + jl > len(b):
        raise Error("mojocv: glb: first chunk must be JSON, 4-byte aligned, inside the file")
    var js = Js()
    var json_bytes = List[UInt8]()
    for i in range(jl):
        json_bytes.append(b[20 + i])
    var p = 0
    _ = _jval(json_bytes, p, js, 0)
    var bin = List[UInt8]()
    var at = 20 + jl
    if at + 8 <= len(b):
        var bl = _rd_u32(b, at)
        if _rd_u32(b, at + 4) != 0x004E4942 or at + 8 + bl > len(b):
            raise Error("mojocv: glb: second chunk must be BIN and inside the file")
        for i in range(bl):
            bin.append(b[at + 8 + i])
    var prim = js.at(js.get(js.at(js.get(0, "meshes"), 0), "primitives"), 0)
    var attrs = js.get(_need(js, prim, "meshes[0].primitives[0]"), "attributes")
    var g = Glb()
    var pos = List[Float64]()
    var nv = _accessor(js, bin, Int(js.num[_need(js, js.get(attrs, "POSITION"), "POSITION")]), 3, pos, True)
    g.verts = Mat[DType.float64](nv, 3)
    g.verts.data = pos^
    var nrm_node = js.get(attrs, "NORMAL")
    if nrm_node >= 0:
        var nd = List[Float64]()
        if _accessor(js, bin, Int(js.num[nrm_node]), 3, nd, True) != nv:
            raise Error("mojocv: glb: NORMAL count differs from POSITION")
        g.normals = Mat[DType.float64](nv, 3)
        g.normals.data = nd^
    var col_node = js.get(attrs, "COLOR_0")
    if col_node >= 0:
        var cd = List[Float64]()
        if _accessor(js, bin, Int(js.num[col_node]), 3, cd, True) != nv:
            raise Error("mojocv: glb: COLOR_0 count differs from POSITION")
        g.colors = Mat[DType.float64](nv, 3)
        g.colors.data = cd^
    var idx = List[Float64]()
    var ni = nv
    if js.get(prim, "indices") >= 0:
        ni = _accessor(js, bin, Int(js.num[js.get(prim, "indices")]), 1, idx, False)
    else:
        for i in range(nv):
            idx.append(Float64(i))
    if js.get(prim, "mode") >= 0 and Int(js.num[js.get(prim, "mode")]) != 4:
        raise Error("mojocv: glb: only TRIANGLES (mode 4) primitives are supported")
    if ni % 3 != 0:
        raise Error(String("mojocv: glb: ", ni, " indices is not a multiple of 3"))
    g.faces = Mat[DType.int32](ni // 3, 3)
    for k in range(ni):
        if Int(idx[k]) >= nv:
            raise Error(String("mojocv: glb: index ", Int(idx[k]), " >= vertex count ", nv))
        g.faces.data[k] = Int32(Int(idx[k]))
    return g^
