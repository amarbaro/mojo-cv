# 3D / time / CAD / CAE ops for the mojo-cv CLI: `mojo-cv3d OP OUT IN... [k=v]`. src/main.mojo calls
# dispatch3d for any op it does not know. Every op reads and writes .mcv files; scalars are printed too.
from std.sys import argv
from std.time import perf_counter_ns
from std.collections import Dict
from mojocv.mat import Mat, load, save
from mojocv.voxel import voxelize, unvoxelize, occupied
from mojocv.align import umeyama
from mojocv.wavelet import haar3, compress
from mojocv.desc3d import describe, match_desc
from mojocv.tapvid import tapvid
from mojocv.phasecorr import phase_corr
from mojocv.burst import burst, register
from mojocv.nurbs import eval_surface, fit_surface
from mojocv.mesh import voxel_mesh, edges_euler
from mojocv.glb import write_glb, read_glb
from mojocv.step import write_step_mesh, write_step_surface, read_step
from mojocv.cae import stress_measures, colormap
from mojocv.octree import Octree, walk, walk_cells


def _f(kv: Dict[String, String], name: String, dflt: Float64) raises -> Float64:
    return Float64(kv[name]) if name in kv else dflt


def _i(kv: Dict[String, String], name: String, dflt: Int) raises -> Int:
    return Int(kv[name]) if name in kv else dflt


def _trees(ins: List[String], depth: Int) raises -> List[Octree]:
    var ts = List[Octree]()
    for p in ins:
        ts.append(Octree(load[DType.int32](p), depth))
    return ts^


def _spec(op: String) -> String:
    """`min_inputs max_inputs allowed_keys...` per op; "" for an op this module does not own."""
    if op == "voxelize" or op == "unvoxelize" or op == "occupied":
        return "1 1 size"
    if op == "ohash":
        return "1 1 depth reps"
    if op == "odiff":
        return "2 2 depth reps"
    if op == "static":
        return "1 64 depth reps"
    if op == "umeyama":
        return "2 2 scale reps"
    if op == "haar3f" or op == "haar3i":
        return "1 1 levels"
    if op == "haar3c":
        return "1 1 thr levels reps"
    if op == "describe":
        return "1 1 k reps"
    if op == "match":
        return "2 2 reps"
    if op == "tapvid":
        return "4 4 focal metric"
    if op == "phasecorr":
        return "2 2 window reps"
    if op == "burst":
        return "1 65 scale shifts window reps"
    if op == "nurbs_fit":
        return "1 1 nu nv p q reps"
    if op == "nurbs_eval":
        return "1 1 p q m n reps"
    if op == "mesh":
        return "1 1 faces reps"
    if op == "glb":
        return "2 3"
    if op == "glb_read":
        return "1 1 faces normals colors"
    if op == "step":
        return "2 2"
    if op == "step_surface":
        return "1 1 p q"
    if op == "step_read":
        return "1 1 faces net lenient"
    if op == "stress":
        return "1 1"
    if op == "colormap":
        return "1 1 lo hi"
    return ""


def _validate(op: String, ins: List[String], kv: Dict[String, String]) raises:
    var parts = _spec(op).split(" ")
    var lo = Int(parts[0])
    var hi = Int(parts[1])
    if len(ins) < lo or len(ins) > hi:
        raise Error(String("mojocv: ", op, " takes ", lo, " to ", hi, " inputs, got ", len(ins)))
    for item in kv.items():
        var ok = False
        for i in range(2, len(parts)):
            ok = ok or parts[i] == item.key
        if not ok:
            raise Error(String("mojocv: ", op, " has no option '", item.key, "'"))


def _req(kv: Dict[String, String], name: String) raises -> String:
    if name not in kv:
        raise Error(String("mojocv: missing option ", name, "=PATH"))
    return kv[name]


def _lap(t0: Int, mut best: Int):
    best = min(best, Int(perf_counter_ns()) - t0)


def dispatch3d(args: List[String]) raises -> Bool:
    """Args = OP OUT IN... [k=v...]; True when the op belongs to this module set."""
    if len(args) < 2:
        return False
    var op = args[0]
    var out = args[1]
    var ins = List[String]()
    var kv = Dict[String, String]()
    for i in range(2, len(args)):
        var eq = args[i].find("=")
        if eq > 0:
            kv[String(args[i][byte=:eq])] = String(args[i][byte=eq + 1 :])
        else:
            ins.append(args[i])
    if _spec(op).byte_length() == 0:
        return False
    _validate(op, ins, kv)
    var reps = _i(kv, "reps", 0)  # reps=N: repeat the compute step in-process, print the best time (loading excluded)
    var best = Int.MAX
    if op == "voxelize":
        save(voxelize(load[DType.float64](ins[0]), _f(kv, "size", 1.0)), out)
    elif op == "unvoxelize":
        save(unvoxelize(load[DType.float64](ins[0]), _f(kv, "size", 1.0)), out)
    elif op == "occupied":
        save(occupied(load[DType.float64](ins[0]), _f(kv, "size", 1.0)), out)
    elif op == "ohash":
        var cells = load[DType.int32](ins[0])
        var t = Octree(cells, _i(kv, "depth", 6))
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            t = Octree(cells, _i(kv, "depth", 6))
            _lap(t0, best)
        var h = Mat[DType.uint8](1, 8)
        for i in range(8):
            h.data[i] = UInt8((t.root_hash() >> UInt64(8 * i)) & 0xFF)
        print("ohash cells=", t.cells())
        save(h, out)
    elif op == "odiff" or op == "static":
        var depth = _i(kv, "depth", 6)
        var ts = _trees(ins, depth)
        var w = walk(ts, op == "static")
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            w = walk(ts, op == "static")
            _lap(t0, best)
        print(op, "visited=", w.visited, "dense=", 1 << (3 * depth), "out=", len(w.keys))
        save(walk_cells(w, op == "static"), out)
    elif op == "umeyama":
        var a = load[DType.float64](ins[0])
        var b = load[DType.float64](ins[1])
        var r = umeyama(a, b, _i(kv, "scale", 1) != 0)
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            r = umeyama(a, b, _i(kv, "scale", 1) != 0)
            _lap(t0, best)
        save(r, out)
    elif op == "haar3f" or op == "haar3i":
        var v = load[DType.float64](ins[0])
        save(haar3[False](v, _i(kv, "levels", 0)) if op == "haar3f" else haar3[True](v, _i(kv, "levels", 0)), out)
    elif op == "haar3c":
        var v = load[DType.float64](ins[0])
        var r = compress(v, _f(kv, "thr", 0.0), _i(kv, "levels", 0))
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            r = compress(v, _f(kv, "thr", 0.0), _i(kv, "levels", 0))
            _lap(t0, best)
        print("haar3c kept=", r[1], "of=", v.size())
        save(r[0], out)
    elif op == "describe":
        var p = load[DType.float64](ins[0])
        var r = describe(p, _i(kv, "k", 16))
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            r = describe(p, _i(kv, "k", 16))
            _lap(t0, best)
        save(r, out)
    elif op == "match":
        var a = load[DType.uint8](ins[0])
        var b = load[DType.uint8](ins[1])
        var r = match_desc(a, b)
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            r = match_desc(a, b)
            _lap(t0, best)
        save(r, out)
    elif op == "tapvid":
        var r = tapvid(
            load[DType.float64](ins[0]), load[DType.uint8](ins[1]), load[DType.float64](ins[2]),
            load[DType.uint8](ins[3]), _f(kv, "focal", 500.0), _i(kv, "metric", 0) != 0,
        )
        print("tapvid apd=", r.data[0], "oa=", r.data[1], "aj=", r.data[2])
        save(r, out)
    elif op == "phasecorr":
        var a = load[DType.float64](ins[0])
        var b = load[DType.float64](ins[1])
        var r = phase_corr(a, b, _i(kv, "window", 0) != 0)
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            r = phase_corr(a, b, _i(kv, "window", 0) != 0)
            _lap(t0, best)
        save(r, out)
    elif op == "burst":
        # IN... = frames, then (shifts=1) a T x 2 float64 file of known [dx, dy]; otherwise phase correlation finds them
        var frames = List[Mat[DType.float64]]()
        var nf = len(ins) - (1 if _i(kv, "shifts", 0) != 0 else 0)
        for i in range(nf):
            frames.append(load[DType.float64](ins[i]))
        var sh = load[DType.float64](ins[nf]) if nf < len(ins) else register(frames, _i(kv, "window", 0) != 0)
        for t in range(sh.rows):
            print("burst shift", t, sh.data[t * 2], sh.data[t * 2 + 1])
        var r = burst(frames, sh, _i(kv, "scale", 2))
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            r = burst(frames, sh, _i(kv, "scale", 2))
            _lap(t0, best)
        save(r, out)
    elif op == "nurbs_fit":
        var g = load[DType.float64](ins[0])
        var r = fit_surface(g, _i(kv, "nu", 6), _i(kv, "nv", 6), _i(kv, "p", 3), _i(kv, "q", 3))
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            r = fit_surface(g, _i(kv, "nu", 6), _i(kv, "nv", 6), _i(kv, "p", 3), _i(kv, "q", 3))
            _lap(t0, best)
        save(r, out)
    elif op == "nurbs_eval":
        var net = load[DType.float64](ins[0])
        var r = eval_surface(net, _i(kv, "p", 3), _i(kv, "q", 3), _i(kv, "m", 16), _i(kv, "n", 16))
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            r = eval_surface(net, _i(kv, "p", 3), _i(kv, "q", 3), _i(kv, "m", 16), _i(kv, "n", 16))
            _lap(t0, best)
        save(r, out)
    elif op == "mesh":
        # OUT = vertices (V x 3 float64), faces=PATH = triangles (F x 3 int32)
        var cells = load[DType.int32](ins[0])
        var m = voxel_mesh(cells)
        for _ in range(reps):
            var t0 = Int(perf_counter_ns())
            m = voxel_mesh(cells)
            _lap(t0, best)
        var ec = edges_euler(m)
        print("mesh V=", m.verts.rows, "E=", ec[0], "F=", m.faces.rows, "chi=", ec[1])
        save(m.verts, out)
        save(m.faces, _req(kv, "faces"))
    elif op == "glb":
        # OUT.glb = IN vertices (V x 3 float64), IN faces (F x 3 int32), optional IN colors (V x 3 float64 in [0, 1])
        var colors = load[DType.float64](ins[2]) if len(ins) > 2 else Mat[DType.float64](0, 3)
        write_glb(load[DType.float64](ins[0]), load[DType.int32](ins[1]), colors, out)
    elif op == "glb_read":
        # IN.glb -> OUT = vertices; faces=PATH, and normals=PATH / colors=PATH when the file has them
        var g = read_glb(ins[0])
        print("glb_read V=", g.verts.rows, "F=", g.faces.rows, "normals=", g.normals.rows, "colors=", g.colors.rows)
        save(g.verts, out)
        save(g.faces, _req(kv, "faces"))
        if "normals" in kv and g.normals.rows > 0:
            save(g.normals, kv["normals"])
        if "colors" in kv and g.colors.rows > 0:
            save(g.colors, kv["colors"])
    elif op == "step":
        write_step_mesh(load[DType.float64](ins[0]), load[DType.int32](ins[1]), out)
    elif op == "step_surface":
        write_step_surface(load[DType.float64](ins[0]), _i(kv, "p", 3), _i(kv, "q", 3), out)
    elif op == "step_read":
        # IN.stp -> OUT = vertices, faces=PATH; net=PATH for a surface (p and q are printed)
        var d = read_step(ins[0], _i(kv, "lenient", 0) == 0)
        print("step_read V=", d.verts.rows, "F=", d.faces.rows, "net=", d.net.rows, "x", d.net.cols, "p=", d.p, "q=", d.q)
        save(d.verts, out)
        save(d.faces, _req(kv, "faces"))
        if "net" in kv and d.net.rows > 0:
            save(d.net, kv["net"])
    elif op == "stress":
        save(stress_measures(load[DType.float64](ins[0])), out)
    elif op == "colormap":
        var v = load[DType.float64](ins[0])
        var lo = _f(kv, "lo", 1e300)  # a missing end is the field's own min or max
        var hi = _f(kv, "hi", -1e300)
        for x in v.data:
            if "lo" not in kv:
                lo = min(lo, x)
            if "hi" not in kv:
                hi = max(hi, x)
        save(colormap(v, lo, hi), out)
    else:
        return False
    if reps > 0:
        print("time best_ms=", Float64(best) / 1e6, "reps=", reps)
    return True


def main() raises:
    var a = List[String]()
    var av = argv()
    for i in range(1, len(av)):
        a.append(String(av[i]))
    if not dispatch3d(a):
        print("FAIL cli3d: unknown op or too few arguments")
        raise Error("cli3d")
