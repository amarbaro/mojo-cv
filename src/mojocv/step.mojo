# STEP AP242 (ISO 10303-21): a writer and a strict reader for the entities this module writes.
#   write_step_mesh    a closed triangle mesh as one FACETED_BREP: CLOSED_SHELL of FACE / FACE_OUTER_BOUND / POLY_LOOP
#                      over shared CARTESIAN_POINTs, inside the AP242 product structure (PRODUCT .. SHAPE_REPRESENTATION)
#   write_step_surface a B-spline or NURBS surface on clamped uniform knots as B_SPLINE_SURFACE_WITH_KNOTS (plain) or the
#                      complex RATIONAL_B_SPLINE_SURFACE instance, as an item of the same shape representation
# The reader parses any Part 21 data section, then accepts only the entity types listed in KNOWN (the product
# structure, units and the geometry above). Any other entity raises "mojocv: step: unsupported entity NAME (#id)"
# so that nothing is dropped silently; every #reference must resolve.
from std.collections import Dict, Set
from mojocv.mat import Mat
from mojocv.nurbs import knots


def _real(x: Float64) raises -> String:
    if x != x or abs(x) > 1.0e300:
        raise Error("mojocv: step: coordinates must be finite")
    var s = String(x)
    var e = s.find("e")
    if e >= 0:
        var mant = String(s[byte=:e])
        if mant.find(".") < 0:
            mant += "."
        return mant + "E" + String(s[byte = e + 1 :])
    if s.find(".") < 0:
        s += "."
    return s


def _point(id: Int, x: Float64, y: Float64, z: Float64) raises -> String:
    return String("#", id, "=CARTESIAN_POINT('',(", _real(x), ",", _real(y), ",", _real(z), "));\n")


def _header_and_context() -> String:
    # fixed entity ids 1..16; geometry starts at 17
    return String(
        "ISO-10303-21;\nHEADER;\nFILE_DESCRIPTION(('mojo-cv'),'2;1');\n",
        "FILE_NAME('part.stp','2026-01-01T00:00:00',('mojo-cv'),(''),'mojo-cv','mojo-cv','');\n",
        "FILE_SCHEMA(('AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF { 1 0 10303 442 1 1 4 }'));\nENDSEC;\nDATA;\n",
        "#1=APPLICATION_CONTEXT('core data for automotive mechanical design processes');\n",
        "#2=APPLICATION_PROTOCOL_DEFINITION('international standard','ap242_managed_model_based_3d_engineering',2014,#1);\n",
        "#3=PRODUCT_CONTEXT('',#1,'mechanical');\n",
        "#4=PRODUCT('part','part','',(#3));\n",
        "#5=PRODUCT_DEFINITION_FORMATION('','',#4);\n",
        "#6=PRODUCT_DEFINITION_CONTEXT('part definition',#1,'design');\n",
        "#7=PRODUCT_DEFINITION('design','',#5,#6);\n",
        "#8=PRODUCT_DEFINITION_SHAPE('','',#7);\n",
        "#10=(LENGTH_UNIT()NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\n",
        "#11=(NAMED_UNIT(*)PLANE_ANGLE_UNIT()SI_UNIT($,.RADIAN.));\n",
        "#12=(NAMED_UNIT(*)SI_UNIT($,.STERADIAN.)SOLID_ANGLE_UNIT());\n",
        "#13=UNCERTAINTY_MEASURE_WITH_UNIT(LENGTH_MEASURE(1.E-07),#10,'distance_accuracy_value','');\n",
        "#14=(GEOMETRIC_REPRESENTATION_CONTEXT(3)GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT((#13))",
        "GLOBAL_UNIT_ASSIGNED_CONTEXT((#10,#11,#12))REPRESENTATION_CONTEXT('',''));\n",
    )


def _footer(items: String) -> String:
    return String(
        "#9=SHAPE_DEFINITION_REPRESENTATION(#8,#15);\n",
        "#15=ADVANCED_BREP_SHAPE_REPRESENTATION('',(", items, "),#14);\nENDSEC;\nEND-ISO-10303-21;\n",
    )


def write_step_mesh(v: Mat[DType.float64], f: Mat[DType.int32], path: String) raises:
    if v.cols != 3 or f.cols != 3 or v.rows == 0 or f.rows == 0 or v.ch != 1 or f.ch != 1:
        raise Error("mojocv: step: write wants V x 3 vertices and F x 3 faces, both non-empty")
    var body = String("#16=AXIS2_PLACEMENT_3D('',#17,#18,#19);\n")
    body += "#17=CARTESIAN_POINT('',(0.,0.,0.));\n#18=DIRECTION('',(0.,0.,1.));\n#19=DIRECTION('',(1.,0.,0.));\n"
    var p0 = 20  # point i has id p0 + i
    for i in range(v.rows):
        body += _point(p0 + i, v.data[i * 3], v.data[i * 3 + 1], v.data[i * 3 + 2])
    var q0 = p0 + v.rows  # face t: loop q0 + 3t, outer bound +1, face +2
    var faces = String("")
    for t in range(f.rows):
        var ids = List[Int]()
        for k in range(3):
            var i = Int(f.data[t * 3 + k])
            if i < 0 or i >= v.rows:
                raise Error(String("mojocv: step: face index ", i, " outside 0..", v.rows - 1))
            ids.append(p0 + i)
        var b = q0 + 3 * t
        body += String("#", b, "=POLY_LOOP('',(#", ids[0], ",#", ids[1], ",#", ids[2], "));\n")
        body += String("#", b + 1, "=FACE_OUTER_BOUND('',#", b, ",.T.);\n")
        body += String("#", b + 2, "=FACE('',(#", b + 1, "));\n")
        faces += String("#", b + 2) if t == 0 else String(",#", b + 2)
    var shell = q0 + 3 * f.rows
    body += String("#", shell, "=CLOSED_SHELL('',(", faces, "));\n")
    body += String("#", shell + 1, "=FACETED_BREP('part',#", shell, ");\n")
    with open(path, "w") as fh:
        fh.write(_header_and_context() + body + _footer(String("#16,#", shell + 1)))


def _ilist(xs: List[Int]) -> String:
    var s = String("(")
    for i in range(len(xs)):
        s += String(xs[i]) if i == 0 else String(",", xs[i])
    return s + ")"


def write_step_surface(net: Mat[DType.float64], p: Int, q: Int, path: String) raises:
    var nu = net.rows
    var nv = net.cols
    if net.ch != 4 or nu <= p or nv <= q or p < 1 or q < 1:
        raise Error("mojocv: step: surface wants a nu x nv x 4 net with nu > p >= 1 and nv > q >= 1")
    var body = String("#16=AXIS2_PLACEMENT_3D('',#17,#18,#19);\n")
    body += "#17=CARTESIAN_POINT('',(0.,0.,0.));\n#18=DIRECTION('',(0.,0.,1.));\n#19=DIRECTION('',(1.,0.,0.));\n"
    var rational = False
    for k in range(nu * nv):
        rational = rational or net.data[k * 4 + 3] != 1.0
        body += _point(20 + k, net.data[k * 4], net.data[k * 4 + 1], net.data[k * 4 + 2])
    var grid = String("(")
    var wts = String("(")
    for i in range(nu):
        grid += "(" if i == 0 else ",("
        wts += "(" if i == 0 else ",("
        for j in range(nv):
            grid += String("#", 20 + i * nv + j) if j == 0 else String(",#", 20 + i * nv + j)
            wts += _real(net.data[(i * nv + j) * 4 + 3]) if j == 0 else "," + _real(net.data[(i * nv + j) * 4 + 3])
        grid += ")"
        wts += ")"
    grid += ")"
    wts += ")"
    var mu = List[Int]()  # clamped uniform knots: distinct values with multiplicities
    var ku = String("(0.")
    mu.append(p + 1)
    for j in range(1, nu - p):
        mu.append(1)
        ku += "," + _real(Float64(j) / Float64(nu - p))
    mu.append(p + 1)
    ku += ",1.)"
    var mv = List[Int]()
    var kv = String("(0.")
    mv.append(q + 1)
    for j in range(1, nv - q):
        mv.append(1)
        kv += "," + _real(Float64(j) / Float64(nv - q))
    mv.append(q + 1)
    kv += ",1.)"
    var id = 20 + nu * nv
    var head = String(p, ",", q, ",", grid, ",.UNSPECIFIED.,.F.,.F.,.F.")
    if rational:
        body += String(
            "#", id, "=(BOUNDED_SURFACE()B_SPLINE_SURFACE(", head, ")B_SPLINE_SURFACE_WITH_KNOTS(", _ilist(mu), ",",
            _ilist(mv), ",", ku, ",", kv, ",.UNSPECIFIED.)GEOMETRIC_REPRESENTATION_ITEM()RATIONAL_B_SPLINE_SURFACE(", wts,
            ")REPRESENTATION_ITEM('')SURFACE());\n",
        )
    else:
        body += String(
            "#", id, "=B_SPLINE_SURFACE_WITH_KNOTS('',", head, ",", _ilist(mu), ",", _ilist(mv), ",", ku, ",", kv,
            ",.UNSPECIFIED.);\n",
        )
    with open(path, "w") as fh:
        fh.write(_header_and_context() + body + _footer(String("#16,#", id)))


# ---- Part 21 reader: arena of parsed values ----
comptime K_NULL = 0
comptime K_STAR = 1
comptime K_NUM = 2
comptime K_STR = 3
comptime K_ENUM = 4
comptime K_REF = 5
comptime K_LIST = 6
comptime K_TYPED = 7  # NAME(args): text = NAME, children = args
comptime K_COMPLEX = 8  # (NAME(args) NAME(args) ...): children are K_TYPED

comptime KNOWN = "APPLICATION_CONTEXT APPLICATION_PROTOCOL_DEFINITION PRODUCT_CONTEXT PRODUCT PRODUCT_DEFINITION_FORMATION PRODUCT_DEFINITION_CONTEXT PRODUCT_DEFINITION PRODUCT_DEFINITION_SHAPE SHAPE_DEFINITION_REPRESENTATION ADVANCED_BREP_SHAPE_REPRESENTATION AXIS2_PLACEMENT_3D DIRECTION UNCERTAINTY_MEASURE_WITH_UNIT CARTESIAN_POINT POLY_LOOP FACE_OUTER_BOUND FACE CLOSED_SHELL FACETED_BREP B_SPLINE_SURFACE_WITH_KNOTS LENGTH_UNIT NAMED_UNIT SI_UNIT PLANE_ANGLE_UNIT SOLID_ANGLE_UNIT GEOMETRIC_REPRESENTATION_CONTEXT GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT GLOBAL_UNIT_ASSIGNED_CONTEXT REPRESENTATION_CONTEXT BOUNDED_SURFACE B_SPLINE_SURFACE GEOMETRIC_REPRESENTATION_ITEM RATIONAL_B_SPLINE_SURFACE REPRESENTATION_ITEM SURFACE "


struct P21(Movable):
    var kind: List[Int]
    var num: List[Float64]
    var text: List[String]
    var first: List[Int]
    var next: List[Int]
    var ents: Dict[Int, Int]  # entity id -> root node

    def __init__(out self):
        self.kind = List[Int]()
        self.num = List[Float64]()
        self.text = List[String]()
        self.first = List[Int]()
        self.next = List[Int]()
        self.ents = Dict[Int, Int]()

    def add(mut self, kind: Int) -> Int:
        self.kind.append(kind)
        self.num.append(0.0)
        self.text.append(String(""))
        self.first.append(-1)
        self.next.append(-1)
        return len(self.kind) - 1

    def child(self, node: Int, i: Int) -> Int:
        var c = self.first[node]
        var k = 0
        while c >= 0 and k < i:
            c = self.next[c]
            k += 1
        return c

    def count(self, node: Int) -> Int:
        var n = 0
        var c = self.first[node]
        while c >= 0:
            n += 1
            c = self.next[c]
        return n


def _ws(b: List[UInt8], mut p: Int) raises:
    while p < len(b):
        if b[p] == 32 or b[p] == 9 or b[p] == 10 or b[p] == 13:
            p += 1
        elif b[p] == 47 and p + 1 < len(b) and b[p + 1] == 42:  # /* comment */
            p += 2
            while p + 1 < len(b) and not (b[p] == 42 and b[p + 1] == 47):
                p += 1
            if p + 1 >= len(b):
                raise Error("mojocv: step: unterminated comment")
            p += 2
        else:
            break


def _in(c: UInt8, lo: Int, hi: Int) -> Bool:
    return Int(c) >= lo and Int(c) <= hi


def _is_name(c: UInt8) -> Bool:
    return _in(c, 65, 90) or _in(c, 48, 57) or c == 95 or _in(c, 97, 122) or c == 45


def _val(b: List[UInt8], mut p: Int, mut a: P21, depth: Int) raises -> Int:
    if depth > 32:
        raise Error("mojocv: step: nesting deeper than 32")
    _ws(b, p)
    if p >= len(b):
        raise Error("mojocv: step: data ends early")
    var c = b[p]
    if c == 36:  # $
        p += 1
        return a.add(K_NULL)
    if c == 42:  # *
        p += 1
        return a.add(K_STAR)
    if c == 35:  # #n
        var q = p + 1
        var n = 0
        while q < len(b) and _in(b[q], 48, 57):
            n = n * 10 + Int(b[q]) - 48
            q += 1
        if q == p + 1:
            raise Error(String("mojocv: step: bad reference at byte ", p))
        p = q
        var r = a.add(K_REF)
        a.num[r] = Float64(n)
        return r
    if c == 39:  # 'string'
        var out = List[UInt8]()
        p += 1
        while True:
            if p >= len(b):
                raise Error("mojocv: step: unterminated string")
            if b[p] == 39:
                if p + 1 < len(b) and b[p + 1] == 39:
                    out.append(39)
                    p += 2
                    continue
                p += 1
                break
            out.append(b[p])
            p += 1
        var s = a.add(K_STR)
        a.text[s] = String(StringSlice(unsafe_from_utf8=Span(out)))
        return s
    if c == 46:  # .ENUM.
        var q = p + 1
        while q < len(b) and b[q] != 46:
            q += 1
        if q >= len(b):
            raise Error("mojocv: step: unterminated enumeration")
        var e = a.add(K_ENUM)
        a.text[e] = String(StringSlice(unsafe_from_utf8=Span(b)[p + 1 : q]))
        p = q + 1
        return e
    if c == 40:  # ( list )
        var l = a.add(K_LIST)
        p += 1
        var last = -1
        _ws(b, p)
        if p < len(b) and b[p] == 41:
            p += 1
            return l
        while True:
            var ch = _val(b, p, a, depth + 1)
            if last < 0:
                a.first[l] = ch
            else:
                a.next[last] = ch
            last = ch
            _ws(b, p)
            if p >= len(b):
                raise Error("mojocv: step: list ends early")
            if b[p] == 44:
                p += 1
                continue
            if b[p] == 41:
                p += 1
                return l
            raise Error(String("mojocv: step: ',' or ')' expected at byte ", p))
    if _is_name(c) and not _in(c, 48, 57) and c != 45:  # NAME( args )
        var q = p
        while q < len(b) and _is_name(b[q]):
            q += 1
        var t = a.add(K_TYPED)
        a.text[t] = String(StringSlice(unsafe_from_utf8=Span(b)[p:q]))
        p = q
        _ws(b, p)
        if p >= len(b) or b[p] != 40:
            raise Error(String("mojocv: step: '(' expected after ", a.text[t]))
        var l = _val(b, p, a, depth + 1)
        a.first[t] = a.first[l]
        return t
    var q = p  # number
    while q < len(b) and (_in(b[q], 48, 57) or b[q] == 43 or b[q] == 45 or b[q] == 46 or b[q] == 69 or b[q] == 101):
        q += 1
    if q == p:
        raise Error(String("mojocv: step: unexpected byte ", Int(c), " at ", p))
    var n = a.add(K_NUM)
    try:
        a.num[n] = Float64(String(StringSlice(unsafe_from_utf8=Span(b)[p:q])))
    except:
        raise Error(String("mojocv: step: bad number at byte ", p))
    p = q
    return n


def parse_p21(b: List[UInt8]) raises -> P21:
    var a = P21()
    var p = 0
    _ws(b, p)
    var tag = String("ISO-10303-21;")
    if len(b) < 13 or String(StringSlice(unsafe_from_utf8=Span(b)[:13])) != tag:
        raise Error("mojocv: step: missing ISO-10303-21; header")
    var text = String(StringSlice(unsafe_from_utf8=Span(b)))
    var d = text.find("DATA;")
    var e = text.rfind("ENDSEC;")
    if d < 0 or e < d or text.find("END-ISO-10303-21;") < 0 or text.find("FILE_SCHEMA") < 0:
        raise Error("mojocv: step: missing HEADER FILE_SCHEMA, DATA section or END-ISO-10303-21;")
    p = d + 5
    while True:
        _ws(b, p)
        if p >= e:
            break
        if b[p] != 35:
            raise Error(String("mojocv: step: entity instance expected at byte ", p))
        var id_node = _val(b, p, a, 0)
        var id = Int(a.num[id_node])
        _ws(b, p)
        if p >= len(b) or b[p] != 61:
            raise Error(String("mojocv: step: '=' expected after #", id))
        p += 1
        var root: Int
        _ws(b, p)
        if p < len(b) and b[p] == 40:  # (A() B(...)): a complex entity, members follow each other without commas
            root = a.add(K_COMPLEX)
            p += 1
            var last = -1
            while True:
                _ws(b, p)
                if p >= len(b):
                    raise Error(String("mojocv: step: complex entity #", id, " ends early"))
                if b[p] == 41:
                    p += 1
                    break
                if not _is_name(b[p]) or _in(b[p], 48, 57):
                    raise Error(String("mojocv: step: complex entity #", id, " holds a non-entity at byte ", p))
                var m = _val(b, p, a, 1)
                if last < 0:
                    a.first[root] = m
                else:
                    a.next[last] = m
                last = m
        else:
            root = _val(b, p, a, 0)
        _ws(b, p)
        if p >= len(b) or b[p] != 59:
            raise Error(String("mojocv: step: ';' expected after #", id))
        p += 1
        if id in a.ents:
            raise Error(String("mojocv: step: duplicate entity id #", id))
        a.ents[id] = root
    return a^


def _typed(a: P21, root: Int, name: String) -> Int:
    """The K_TYPED node called `name` in an entity (itself, or a member of a complex entity), or -1."""
    if a.kind[root] == K_TYPED:
        return root if a.text[root] == name else -1
    if a.kind[root] == K_COMPLEX:
        var c = a.first[root]
        while c >= 0:
            if a.text[c] == name:
                return c
            c = a.next[c]
    return -1


def _check_known(a: P21) raises:
    var known = Set[String]()
    for n in String(KNOWN).split(" "):
        if n.byte_length() > 0:
            known.add(String(n))
    for item in a.ents.items():
        var root = item.value
        var members = List[Int]()
        if a.kind[root] == K_COMPLEX:
            var c = a.first[root]
            while c >= 0:
                members.append(c)
                c = a.next[c]
        else:
            members.append(root)
        for m in members:
            if a.text[m] not in known:
                raise Error(String("mojocv: step: unsupported entity ", a.text[m], " (#", item.key, ")"))
            _check_refs(a, m)


def _check_refs(a: P21, node: Int) raises:
    var c = a.first[node]
    while c >= 0:
        if a.kind[c] == K_REF and Int(a.num[c]) not in a.ents:
            raise Error(String("mojocv: step: dangling reference #", Int(a.num[c])))
        if a.kind[c] == K_LIST or a.kind[c] == K_TYPED:
            _check_refs(a, c)
        c = a.next[c]


def _point_xyz(a: P21, ref_node: Int, mut out: List[Float64]) raises:
    var e = a.ents[Int(a.num[ref_node])]
    if a.kind[e] != K_TYPED or a.text[e] != "CARTESIAN_POINT":
        raise Error(String("mojocv: step: #", Int(a.num[ref_node]), " is not a CARTESIAN_POINT"))
    var co = a.child(e, 1)
    if a.count(co) != 3:
        raise Error("mojocv: step: CARTESIAN_POINT must have 3 coordinates")
    for k in range(3):
        out.append(a.num[a.child(co, k)])


struct StepData(Movable):
    var verts: Mat[DType.float64]
    var faces: Mat[DType.int32]
    var net: Mat[DType.float64]  # nu x nv x 4, 0 rows when the file has no surface
    var p: Int
    var q: Int

    def __init__(out self):
        self.verts = Mat[DType.float64](0, 3)
        self.faces = Mat[DType.int32](0, 3)
        self.net = Mat[DType.float64](0, 0, 4)
        self.p = 0
        self.q = 0


def read_step(path: String, strict: Bool = True) raises -> StepData:
    var a = parse_p21(open(path, "r").read_bytes())
    if strict:
        _check_known(a)
    var d = StepData()
    var vid = Dict[Int, Int]()  # point entity id -> vertex row, in order of first use
    for item in a.ents.items():
        var root = item.value
        if a.kind[root] == K_TYPED and a.text[root] == "FACETED_BREP":
            var shell = a.ents[Int(a.num[a.child(root, 1)])]
            if a.text[shell] != "CLOSED_SHELL":
                raise Error("mojocv: step: FACETED_BREP must reference a CLOSED_SHELL")
            var fl = a.child(shell, 1)
            var fc = a.first[fl]
            while fc >= 0:
                var face = a.ents[Int(a.num[fc])]
                if a.text[face] != "FACE" or a.count(a.child(face, 1)) != 1:
                    raise Error("mojocv: step: only single-bound FACE entities are supported")
                var bound = a.ents[Int(a.num[a.first[a.child(face, 1)]])]
                var loop = a.ents[Int(a.num[a.child(bound, 1)])]
                if a.text[bound] != "FACE_OUTER_BOUND" or a.text[loop] != "POLY_LOOP":
                    raise Error("mojocv: step: faces must be a FACE_OUTER_BOUND of a POLY_LOOP")
                var pl = a.child(loop, 1)
                if a.count(pl) != 3:
                    raise Error("mojocv: step: only triangular POLY_LOOPs are supported")
                var pc = a.first[pl]
                while pc >= 0:
                    var pid = Int(a.num[pc])
                    if pid not in vid:
                        vid[pid] = d.verts.rows
                        _point_xyz(a, pc, d.verts.data)
                        d.verts.rows += 1
                    d.faces.data.append(Int32(vid[pid]))
                    pc = a.next[pc]
                d.faces.rows += 1
                fc = a.next[fc]
        var s = _typed(a, root, "B_SPLINE_SURFACE_WITH_KNOTS")
        if s >= 0:
            var base = _typed(a, root, "B_SPLINE_SURFACE") if a.kind[root] == K_COMPLEX else s
            var off = 0 if a.kind[root] == K_COMPLEX else 1  # plain entity: ('', p, q, grid, ...); complex: (p, q, grid, ...)
            d.p = Int(a.num[a.child(base, off)])
            d.q = Int(a.num[a.child(base, off + 1)])
            var grid = a.child(base, off + 2)
            var nu = a.count(grid)
            var nv = a.count(a.first[grid])
            d.net = Mat[DType.float64](nu, nv, 4)
            var r = a.first[grid]
            var i = 0
            while r >= 0:
                var cc = a.first[r]
                var j = 0
                while cc >= 0:
                    var xyz = List[Float64]()
                    _point_xyz(a, cc, xyz)
                    for k in range(3):
                        d.net.data[(i * nv + j) * 4 + k] = xyz[k]
                    d.net.data[(i * nv + j) * 4 + 3] = 1.0
                    cc = a.next[cc]
                    j += 1
                r = a.next[r]
                i += 1
            var rat = _typed(a, root, "RATIONAL_B_SPLINE_SURFACE")
            if rat >= 0:
                var wr = a.first[a.child(rat, 0)]
                var i2 = 0
                while wr >= 0:
                    var wc = a.first[wr]
                    var j2 = 0
                    while wc >= 0:
                        d.net.data[(i2 * nv + j2) * 4 + 3] = a.num[wc]
                        wc = a.next[wc]
                        j2 += 1
                    wr = a.next[wr]
                    i2 += 1
            # knots must be the clamped uniform vectors this library evaluates
            var ks = s
            var moff = 0 if a.kind[root] == K_COMPLEX else 8
            _check_knots(a, a.child(ks, moff), a.child(ks, moff + 2), nu, d.p)
            _check_knots(a, a.child(ks, moff + 1), a.child(ks, moff + 3), nv, d.q)
    if d.verts.rows == 0 and d.net.rows == 0:
        raise Error("mojocv: step: no FACETED_BREP or B_SPLINE_SURFACE_WITH_KNOTS found")
    return d^


def _check_knots(a: P21, mult: Int, vals: Int, n: Int, deg: Int) raises:
    var expect = knots(n, deg)
    var k = 0
    var m = a.first[mult]
    var v = a.first[vals]
    while m >= 0 and v >= 0:
        for _ in range(Int(a.num[m])):
            if k >= len(expect) or abs(expect[k] - a.num[v]) > 1e-12:
                raise Error("mojocv: step: knot vector is not clamped uniform, unsupported")
            k += 1
        m = a.next[m]
        v = a.next[v]
    if k != len(expect) or m >= 0 or v >= 0:
        raise Error("mojocv: step: knot vector is not clamped uniform, unsupported")
