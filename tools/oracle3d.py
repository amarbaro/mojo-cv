"""Independent numpy oracles for the L3D ops (brute force or closed form), plus the fixture generator.
OPS[op](inputs, kv) -> ndarray, same shape/dtype as the Mojo op writes."""
import numpy as np

U64 = np.uint64
GOLD = U64(0x9E3779B97F4A7C15)
rng = np.random.default_rng(20261022)


def splitmix(x):
    with np.errstate(over="ignore"):
        z = x.astype(U64) + GOLD
        z = (z ^ (z >> U64(30))) * U64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
        return z ^ (z >> U64(31))


def zkey(axis, v):
    return splitmix((U64(axis + 1) << U64(40)) + v.astype(U64))


def cell_hash(c):
    c = c.astype(np.int64)
    return splitmix(zkey(0, c[:, 0]) ^ zkey(1, c[:, 1]) ^ zkey(2, c[:, 2]))


def f(kv, k, d):
    return float(kv.get(k, d))


def voxelize(ins, kv):
    s = ins[0] / f(kv, "size", 1.0)
    c = np.floor(s)
    return np.hstack([c, s - c])


def unvoxelize(ins, kv):
    v = ins[0]
    return (v[:, :3] + v[:, 3:]) * f(kv, "size", 1.0)


def occupied(ins, kv):
    return np.unique(np.floor(ins[0] / f(kv, "size", 1.0)).astype(np.int32), axis=0)


def ohash(ins, kv):
    h = np.bitwise_xor.reduce(cell_hash(ins[0])) if len(ins[0]) else U64(0)
    return np.frombuffer(np.array([h], dtype="<u8").tobytes(), np.uint8).reshape(1, 8)


def _sets(ins):
    return [set(map(tuple, a.tolist())) for a in ins]


def odiff(ins, kv):
    a, b = _sets(ins)
    return np.array(sorted(a ^ b), dtype=np.int32).reshape(-1, 3)


def static(ins, kv):
    sets = _sets(ins)
    allc = sorted(set().union(*sets))
    return np.array([(*c, int(all(c in s for s in sets))) for c in allc], dtype=np.int32).reshape(-1, 4)


def _haar(a, inverse, levels):
    n = a.shape[0]
    full = int(np.log2(n))
    nl = int(levels) or full
    a = a.astype(np.float64).copy()
    r = np.sqrt(0.5)
    for k in range(nl):
        m = n >> (nl - 1 - k if inverse else k)
        for ax in range(3):
            v = np.moveaxis(a[:m, :m, :m], ax, 0)  # a view: assignments write through
            if inverse:
                lo, hi = v[: m // 2].copy(), v[m // 2 :].copy()
                v[0::2], v[1::2] = (lo + hi) * r, (lo - hi) * r
            else:
                e, o = v[0::2].copy(), v[1::2].copy()
                v[: m // 2], v[m // 2 :] = (e + o) * r, (e - o) * r
    return a


def haar3f(ins, kv):
    return _haar(ins[0], False, kv.get("levels", 0))


def haar3i(ins, kv):
    return _haar(ins[0], True, kv.get("levels", 0))


def haar3c(ins, kv):
    c = _haar(ins[0], False, kv.get("levels", 0))
    c[np.abs(c) < f(kv, "thr", 0.0)] = 0.0
    return _haar(c, True, kv.get("levels", 0))


def describe(ins, kv):
    p, k = ins[0], int(kv.get("k", 16))
    n = len(p)
    tri = []
    for j in range(64):
        a, b, c = (int(splitmix(np.array([3 * j + t]))[0] % U64(k)) for t in range(3))
        while b == a:
            b = (b + 1) % k
        while c in (a, b):
            c = (c + 1) % k
        tri.append((a, b, c))
    out = np.zeros((n, 8), np.uint8)
    for i in range(n):
        d = p - p[i]
        d2 = (d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]) + d[:, 2] * d[:, 2]
        d2[i] = np.inf
        nb = np.lexsort((np.arange(n), d2))[:k]
        bits = 0
        for j, (a, b, c) in enumerate(tri):
            u, v, w = p[nb[a]] - p[i], p[nb[b]] - p[i], p[nb[c]] - p[i]
            cx, cy, cz = u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]
            if (cx * w[0] + cy * w[1]) + cz * w[2] > 0:
                bits |= 1 << j
        out[i] = np.frombuffer(bits.to_bytes(8, "little"), np.uint8)
    return out


def match(ins, kv):
    a, b = ins
    ham = np.unpackbits(a[:, None, :] ^ b[None, :, :], axis=2).sum(2)
    idx = ham.argmin(1)  # first minimum = lowest index on ties
    return np.stack([idx, ham[np.arange(len(a)), idx]], 1).astype(np.int32)


def tapvid(ins, kv):
    gt, gv, pr, pv = ins[0], ins[1].reshape(-1) != 0, ins[2], ins[3].reshape(-1) != 0
    err = np.linalg.norm(pr - gt, axis=1)
    apd = aj = 0.0
    for k in range(5):
        delta = 0.01 * 4.0 ** k if int(kv.get("metric", 0)) else 2.0 ** k * gt[:, 2] / f(kv, "focal", 500.0)
        close = err < delta
        tp = (gv & pv & close).sum()
        fp = (pv & ~(gv & close)).sum()
        fn = (gv & ~(pv & close)).sum()
        apd += (gv & close).sum() / gv.sum() / 5
        aj += tp / (tp + fp + fn) / 5
    return np.array([[apd, (gv == pv).mean(), aj]])


def _frac(rm, r0, rp, n):
    nb = max(rm, rp)
    if r0 <= 0 or nb <= 0:
        return 0.0
    rho = nb / r0
    fr = n / np.pi * np.arctan(rho * np.sin(np.pi / n) / (1 + rho * np.cos(np.pi / n)))
    return fr if rp >= rm else -fr


def phasecorr(ins, kv):
    a, b = ins
    rows, cols = a.shape
    if int(kv.get("window", 0)):
        w = np.outer(0.5 * (1 - np.cos(2 * np.pi * np.arange(rows) / (rows - 1))),
                     0.5 * (1 - np.cos(2 * np.pi * np.arange(cols) / (cols - 1))))
        a, b = a * w, b * w
    c = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    r = np.real(np.fft.ifft2(c / (np.abs(c) + 1e-12)))
    py, px = np.unravel_index(np.argmax(r), r.shape)
    sg = lambda i, n: i - n if i > (n - 1) // 2 else i
    dx = sg(px, cols) + _frac(r[py, (px - 1) % cols], r[py, px], r[py, (px + 1) % cols], cols)
    dy = sg(py, rows) + _frac(r[(py - 1) % rows, px], r[py, px], r[(py + 1) % rows, px], rows)
    return np.array([[dx, dy, r[py, px]]])


def burst(ins, kv):
    nf = len(ins) - (1 if int(kv.get("shifts", 0)) else 0)
    frames = ins[:nf]
    sc = int(kv.get("scale", 2))
    sh = ins[nf] if nf < len(ins) else np.array(
        [[0.0, 0.0]] + [phasecorr([frames[0], fr], kv)[0, :2] for fr in frames[1:]])
    h, w = frames[0].shape
    num, den = np.zeros((h * sc, w * sc)), np.zeros((h * sc, w * sc))
    yy, xx = np.mgrid[0:h, 0:w]
    for fr, (dx, dy) in zip(frames, sh):
        px, py = (xx - dx) * sc, (yy - dy) * sc
        x0, y0 = np.floor(px).astype(int), np.floor(py).astype(int)
        fx, fy = px - x0, py - y0
        for j in (0, 1):
            for i in (0, 1):
                wgt = (fx if i else 1 - fx) * (fy if j else 1 - fy)
                gx, gy = x0 + i, y0 + j
                ok = (gx >= 0) & (gx < w * sc) & (gy >= 0) & (gy < h * sc)
                np.add.at(num, (gy[ok], gx[ok]), (wgt * fr)[ok])
                np.add.at(den, (gy[ok], gx[ok]), wgt[ok])
    gy, gx = np.mgrid[0:h * sc, 0:w * sc]
    sx, sy = np.minimum(gx / sc, w - 1), np.minimum(gy / sc, h - 1)
    ix, iy = np.minimum(sx.astype(int), w - 2), np.minimum(sy.astype(int), h - 2)
    ax, ay = sx - ix, sy - iy
    r = frames[0]
    fill = (1 - ay) * ((1 - ax) * r[iy, ix] + ax * r[iy, ix + 1]) + ay * ((1 - ax) * r[iy + 1, ix] + ax * r[iy + 1, ix + 1])
    return np.where(den > 1e-9, num / np.maximum(den, 1e-300), fill)


def _knots(n, p):
    return np.concatenate([np.zeros(p), np.linspace(0, 1, n - p + 1), np.ones(p)])


def _bsp(n, p, u):
    """Cox-de Boor: basis (len(u), n) and derivative (len(u), n) on the clamped uniform knots."""
    kn = _knots(n, p)
    u = np.asarray(u, float)

    def level(deg):
        m = len(kn) - deg - 1
        if deg == 0:
            b = np.zeros((len(u), len(kn) - 1))
            for i in range(len(kn) - 1):
                if kn[i] < kn[i + 1]:
                    last = kn[i + 1] == 1.0
                    b[:, i] = (u >= kn[i]) & ((u < kn[i + 1]) | (last & (u == 1.0)))
            return b
        lo = level(deg - 1)
        b = np.zeros((len(u), m))
        for i in range(m):
            d1, d2 = kn[i + deg] - kn[i], kn[i + deg + 1] - kn[i + 1]
            if d1 > 0:
                b[:, i] += (u - kn[i]) / d1 * lo[:, i]
            if d2 > 0:
                b[:, i] += (kn[i + deg + 1] - u) / d2 * lo[:, i + 1]
        return b

    b = level(p)
    lo = level(p - 1)
    d = np.zeros_like(b)
    for i in range(n):
        d1, d2 = kn[i + p] - kn[i], kn[i + p + 1] - kn[i + 1]
        if d1 > 0:
            d[:, i] += p * lo[:, i] / d1
        if d2 > 0:
            d[:, i] -= p * lo[:, i + 1] / d2
    return b, d


def nurbs_eval(ins, kv):
    net = ins[0]
    nu, nv = net.shape[:2]
    p, q, m, n = (int(kv.get(k, d)) for k, d in (("p", 3), ("q", 3), ("m", 16), ("n", 16)))
    bu, du = _bsp(nu, p, np.linspace(0, 1, m))
    bv, dv = _bsp(nv, q, np.linspace(0, 1, n))
    pw = np.concatenate([net[..., :3] * net[..., 3:], net[..., 3:]], axis=2)
    h = np.einsum("ai,bj,ijc->abc", bu, bv, pw)
    hu = np.einsum("ai,bj,ijc->abc", du, bv, pw)
    hv = np.einsum("ai,bj,ijc->abc", bu, dv, pw)
    s = h[..., :3] / h[..., 3:]
    su = (hu[..., :3] - hu[..., 3:] * s) / h[..., 3:]
    sv = (hv[..., :3] - hv[..., 3:] * s) / h[..., 3:]
    nrm = np.cross(su, sv)
    nrm = nrm / np.linalg.norm(nrm, axis=2, keepdims=True)
    return np.concatenate([s, nrm], axis=2)


def nurbs_fit(ins, kv):
    g = ins[0]
    nu, nv, p, q = (int(kv.get(k, d)) for k, d in (("nu", 6), ("nv", 6), ("p", 3), ("q", 3)))
    m, n = g.shape[:2]
    bu, _ = _bsp(nu, p, np.linspace(0, 1, m))
    bv, _ = _bsp(nv, q, np.linspace(0, 1, n))
    net = np.ones((nu, nv, 4))
    for c in range(3):
        qm = np.linalg.lstsq(bu, g[..., c], rcond=None)[0]
        net[..., c] = np.linalg.lstsq(bv, qm.T, rcond=None)[0].T
    return net


def umeyama(ins, kv):
    x, y = ins
    mx, my = x.mean(0), y.mean(0)
    xc, yc = x - mx, y - my
    u, d, vt = np.linalg.svd(yc.T @ xc / len(x))
    sgn = np.ones(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sgn[2] = -1
    r = u @ np.diag(sgn) @ vt
    c = (d * sgn).sum() / (xc ** 2).sum(1).mean() if int(kv.get("scale", 1)) else 1.0
    m = np.eye(4)
    m[:3, :3] = c * r
    m[:3, 3] = my - c * r @ mx
    return m


def stress(ins, kv):
    s = ins[0]
    out = np.zeros((len(s), 6))
    for k, (xx, yy, zz, xy, yz, xz) in enumerate(s):
        m = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        w = np.linalg.eigvalsh(m)[::-1]
        vm = np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy ** 2 + yz ** 2 + xz ** 2))
        out[k] = [(xx + yy + zz) / 3, vm, *w, (w[0] - w[2]) / 2]
    return out


_STOPS = np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]], float)


def colormap(ins, kv):
    v = ins[0].reshape(-1)
    lo = float(kv["lo"]) if "lo" in kv else v.min()
    hi = float(kv["hi"]) if "hi" in kv else v.max()
    t = np.clip((v - lo) / (hi - lo), 0, 1)
    return np.stack([np.interp(t, np.linspace(0, 1, 5), _STOPS[:, c]) for c in range(3)], axis=1)


OPS = {"voxelize": voxelize, "unvoxelize": unvoxelize, "occupied": occupied, "ohash": ohash, "odiff": odiff,
       "static": static, "umeyama": umeyama, "haar3f": haar3f, "haar3i": haar3i,
       "haar3c": haar3c, "tapvid": tapvid, "phasecorr": phasecorr, "nurbs_eval": nurbs_eval, "nurbs_fit": nurbs_fit, "stress": stress, "colormap": colormap, "burst": burst, "describe": describe, "match": match}


def fixtures():
    """name -> ndarray; written as .mcv by the driver."""
    global rng
    rng = np.random.default_rng(20261022)  # same arrays on every call
    fx = {}
    pts = rng.uniform(-8, 8, (500, 3))
    fx["pts500"] = pts
    fx["pts_dup"] = np.vstack([pts[:50], pts[:50] + 1e-3, pts[:10]])
    fx["pts_empty"] = np.zeros((0, 3))
    scene = np.unique(rng.integers(0, 64, (3000, 3)), axis=0).astype(np.int32)
    fx["scene64"] = scene
    extra = np.array([[63, 63, 63]], np.int32) if not (scene == 63).all(1).any() else np.array([[0, 0, 0]], np.int32)
    fx["scene64_add1"] = np.vstack([scene, extra])
    fx["scene64_rm1"] = scene[1:]
    # four cells on a rectangle: raw per-axis Zobrist XOR cancels them, the mixed hash must not
    rect = np.array([[5, 7, 9], [50, 7, 9], [5, 40, 9], [50, 40, 9]], np.int32)
    rect = np.array([r for r in rect if not (scene == r).all(1).any()], np.int32)
    fx["scene64_rect"] = np.vstack([scene, rect])
    fx["rect_len"] = np.array([[len(rect)]], np.int32)
    fx["cells_empty"] = np.zeros((0, 3), np.int32)
    # static/moving window: a fixed room plus a block that walks along x
    room = np.array([(x, y, 0) for x in range(16) for y in range(16)], np.int32)
    fx["frame0"] = np.vstack([room, [[3, 3, 5], [3, 3, 6]]]).astype(np.int32)
    fx["frame1"] = np.vstack([room, [[4, 3, 5], [4, 3, 6]]]).astype(np.int32)
    fx["frame2"] = np.vstack([room[:-1], [[5, 3, 5], [5, 3, 6]]]).astype(np.int32)
    fx.update(align_fixtures())
    fx.update(wavelet_fixtures())
    fx.update(desc_fixtures())
    fx.update(tap_fixtures())
    fx.update(pc_fixtures())
    fx.update(burst_fixtures())
    fx.update(nurbs_fixtures())
    fx.update(mesh_fixtures())
    fx.update(cae_fixtures())
    return fx


def _rot(rs):
    q, r = np.linalg.qr(rs.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else q * np.array([1, 1, -1])


def _sim(s, r, t):
    m = np.eye(4)
    m[:3, :3] = s * r
    m[:3, 3] = t
    return m


def align_fixtures():
    rs = np.random.default_rng(7)
    fx = {}
    src = rs.normal(size=(200, 3)) * [3, 2, 1]
    truth = _sim(1.7, _rot(rs), [4.0, -2.5, 0.75])
    fx["al_src"] = src
    fx["al_truth"] = truth
    fx["al_dst"] = src @ truth[:3, :3].T + truth[:3, 3]
    fx["al_dst_noisy"] = fx["al_dst"] + rs.normal(size=(200, 3)) * 0.05
    plane = src.copy()
    plane[:, 2] = 0
    truth2 = _sim(0.6, _rot(rs), [-1.0, 0.5, 2.0])
    fx["al_plane"] = plane
    fx["al_plane_truth"] = truth2
    fx["al_plane_dst"] = plane @ truth2[:3, :3].T + truth2[:3, 3]
    fx["al_mirror_dst"] = src * [-1, 1, 1]
    fx["al_line"] = np.outer(np.arange(10.0), [1, 2, 3])
    fx["al_few"] = src[:3]
    fx["al_few_dst"] = fx["al_dst"][:3]
    fx["al_one"] = src[:1]
    return fx


def wavelet_fixtures():
    rs = np.random.default_rng(11)
    g = np.linspace(-1, 1, 32)
    z, y, x = np.meshgrid(g, g, g, indexing="ij")
    return {
        "wv_rand8": rs.normal(size=(8, 8, 8)),
        "wv_blob32": np.exp(-4 * (x ** 2 + 2 * y ** 2 + z ** 2)) + 0.01 * rs.normal(size=(32, 32, 32)),
        "wv_const16": np.full((16, 16, 16), 3.0),
        "wv_const_coef": np.pad(np.full((1, 1, 1), 3.0 * 16 ** 1.5), ((0, 15),) * 3),
        "wv_n2": rs.normal(size=(2, 2, 2)),
        "wv_bad6": rs.normal(size=(6, 6, 6)),
    }


def desc_fixtures():
    """A cloud whose descriptors are pairwise >= 7 bits apart, so 3 planted bit flips cannot change the match."""
    for seed in range(13, 200):
        rs = np.random.default_rng(seed)
        n = 100
        p = rs.normal(size=(n, 3))
        da = describe([p], {})
        ham = np.unpackbits(da[:, None, :] ^ da[None, :, :], axis=2).sum(2)
        np.fill_diagonal(ham, 99)
        if ham.min() >= 7:
            break
    t = _sim(2.5, _rot(rs), [1.0, -3.0, 0.5])
    perm = rs.permutation(n)
    db = da[perm].copy()
    for i in range(n):  # three flipped bits per descriptor
        for bit in rs.choice(64, 3, replace=False):
            db[i, bit // 8] ^= 1 << (bit % 8)
    return {
        "ds_pts": p,
        "ds_pts_moved": p @ t[:3, :3].T + t[:3, 3],
        "ds_few": p[:10],
        "ds_pts_mirror": p * [-1, 1, 1],
        "ds_desc_not": np.bitwise_not(da),
        "ds_ident": np.stack([np.arange(n), np.zeros(n)], 1).astype(np.int32),
        "ds_noisy": db,
        "ds_noisy_truth": np.stack([np.argsort(perm), np.full(n, 3)], 1).astype(np.int32),
        "ds_one": da[:1],
        "ds_empty": np.zeros((0, 8), np.uint8),
    }


def tap_fixtures():
    rs = np.random.default_rng(17)
    m = 400
    gt = np.column_stack([rs.normal(size=(m, 2)), rs.uniform(5, 15, m)])
    gv = (rs.random(m) < 0.7).astype(np.uint8)
    flip = rs.random(m) < 0.1
    one = np.ones((4, 1), np.uint8)
    z10 = np.column_stack([rs.normal(size=(4, 2)), np.full(4, 10.0)])
    return {
        "tp_gt": gt, "tp_gv": gv.reshape(-1, 1),
        "tp_pred": gt + rs.normal(size=(m, 3)) * 0.1,
        "tp_pv": (gv ^ flip).astype(np.uint8).reshape(-1, 1),
        "tp_perfect": np.array([[1.0, 1.0, 1.0]]),
        "tp_z10": z10, "tp_z10_off": z10 + [0.05, 0, 0], "tp_one": one,
        "tp_z10_truth": np.array([[0.6, 1.0, 0.6]]),
        "tp_c_gv": np.array([[1], [1], [0], [0]], np.uint8), "tp_c_pv": np.array([[1], [0], [1], [0]], np.uint8),
        "tp_c_truth": np.array([[1.0, 0.5, 1 / 3]]),
        "tp_zero": np.zeros((4, 1), np.uint8),
    }


def fshift(img, dy, dx):
    """Periodic (Fourier) translation by a fractional amount: b(x) = a(x - d)."""
    ky = np.fft.fftfreq(img.shape[0])[:, None]
    kx = np.fft.fftfreq(img.shape[1])[None, :]
    return np.real(np.fft.ifft2(np.fft.fft2(img) * np.exp(-2j * np.pi * (ky * dy + kx * dx))))


def pc_fixtures():
    rs = np.random.default_rng(19)
    fx = {}
    for tag, shp in (("", (64, 96)), ("odd", (45, 60)), ("prime", (31, 37))):
        a = rs.random(shp)
        dy, dx = 5 % shp[0], -7
        fx["pc_a" + tag] = a
        fx["pc_b" + tag] = np.roll(a, (dy, dx), (0, 1))
        fx["pc_t" + tag] = np.array([[dx, dy if dy <= (shp[0] - 1) // 2 else dy - shp[0], 1.0]])
    a = rs.random((65, 97))
    fx["pc_af"] = a
    fx["pc_bf"] = fshift(a, 2.5, -3.25)
    fx["pc_tf"] = np.array([[-3.25, 2.5]])
    fx["pc_bf2"] = fshift(a, 0.77, -0.3)
    fx["pc_tf2"] = np.array([[-0.3, 0.77]])
    fx["pc_bnoise"] = np.roll(a, (3, 4), (0, 1)) + 0.2 * rs.random((65, 97))
    fx["pc_zero"] = np.zeros((16, 16))
    fx["pc_zt"] = np.zeros((1, 3))
    fx["pc_small"] = rs.random((8, 8))
    return fx


def burst_fixtures():
    """A smooth periodic fine image H and four LR frames sampled half an LR pixel apart (each fine pixel is hit once)."""
    rs = np.random.default_rng(23)
    h, w = 24, 32
    spec = np.zeros((h, w), complex)
    for ky in range(-3, 4):
        for kx in range(-3, 4):
            if (ky, kx) != (0, 0):
                spec[ky % h, kx % w] = rs.normal() + 1j * rs.normal()
    hr = np.real(np.fft.ifft2(spec)) * h * w / 20 + 0.5
    frames = {}
    for t, (sx, sy) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):  # shifts in fine pixels
        frames[f"bs_f{t}"] = np.roll(hr, (sy, sx), (0, 1))[::2, ::2].copy()
    # np.roll content by +s means frame(x) = hr(2x - s): the shift of frame t in LR pixels is s / 2
    frames["bs_shifts"] = np.array([[0, 0], [0.5, 0], [0, 0.5], [0.5, 0.5]])
    frames["bs_nan"] = np.array([[0, 0], [np.nan, 0], [0, 0.5], [0.5, 0.5]])
    frames["bs_hr"] = hr
    # broadband frames (all spectrum bins informative, odd sizes: exact Fourier shifts) for the registration rows
    r0 = rs.random((13, 17))
    for t, (sx, sy) in enumerate([(0, 0), (0.5, 0), (0, 0.5), (0.5, 0.5)]):
        frames[f"bs_r{t}"] = fshift(r0, sy, sx)
    frames["bs_rn"] = frames["bs_r3"] + 0.05 * rs.random((13, 17))
    return frames


def nurbs_fixtures():
    rs = np.random.default_rng(29)
    fx = {}
    net = np.ones((6, 5, 4))
    net[..., :3] = rs.normal(size=(6, 5, 3))
    fx["ns_net"] = net
    netw = net.copy()
    netw[..., 3] = rs.uniform(0.5, 2.0, (6, 5))
    fx["ns_netw"] = netw
    grid = nurbs_eval([net], {"p": 3, "q": 2, "m": 30, "n": 25})[..., :3]
    fx["ns_grid"] = grid
    fx["ns_grid_noisy"] = grid + 0.01 * rs.normal(size=grid.shape)
    fx["ns_grid_small"] = grid[:4, :4]
    # plane: control points at the Greville abscissae reproduce (u, v, 0) exactly
    ku, kv_ = _knots(6, 3), _knots(5, 2)
    gu = np.array([ku[i + 1:i + 4].mean() for i in range(6)])
    gv = np.array([kv_[j + 1:j + 3].mean() for j in range(5)])
    plane = np.ones((6, 5, 4))
    plane[..., 0], plane[..., 1], plane[..., 2] = gu[:, None], gv[None, :], 0.0
    fx["ns_plane"] = plane
    u, v = np.meshgrid(np.linspace(0, 1, 9), np.linspace(0, 1, 7), indexing="ij")
    fx["ns_plane_truth"] = np.concatenate(
        [np.stack([u, v, 0 * u], 2), np.broadcast_to([0.0, 0.0, 1.0], (9, 7, 3))], axis=2)
    cyl = np.ones((3, 2, 4))
    for i, (x, y, w) in enumerate([(1, 0, 1), (1, 1, np.sqrt(0.5)), (0, 1, 1)]):
        for j in range(2):
            cyl[i, j] = [x, y, j, w]
    fx["ns_cyl"] = cyl
    return fx


# --- ops whose result is not one array: custom checks, ctx = dict(name, ins, args, expect, work, run, mcv) ---
def mesh_invariants(v, f):
    """Closed-surface facts of a triangle mesh, computed independently of the Mojo code."""
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol = float(np.einsum("ij,ij->i", p0, np.cross(p1, p2)).sum() / 6)
    area = float(0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1).sum())
    manifold = True
    link = {}
    for a, b, c in f.tolist():
        link.setdefault(a, []).append((b, c))
        link.setdefault(b, []).append((c, a))
        link.setdefault(c, []).append((a, b))
    for vtx, segs in link.items():  # the triangles around a vertex must close into ONE cycle
        nxt = dict(segs)
        if len(nxt) != len(segs):
            manifold = False
            break
        start = segs[0][0]
        cur, steps = start, 0
        while True:
            cur = nxt.get(cur)
            steps += 1
            if cur is None or cur == start or steps > len(segs):
                break
        manifold &= cur == start and steps == len(segs)
    return dict(V=len(v), E=len(uniq), F=len(f), edges_twice=bool((cnt == 2).all()), vol=vol, area=area,
                chi=len(v) - len(uniq) + len(f), manifold=manifold, unused=len(set(range(len(v))) - set(f.ravel().tolist())))


def check_mesh(ctx):
    cells = ctx["mcv"].load(ctx["ins"][0])
    out, faces = ctx["work"] / f"{ctx['name']}.v.mcv", ctx["work"] / f"{ctx['name']}.f.mcv"
    if "raises" in ctx["expect"]:  # option faces=PATH left out
        code, stdout, stderr = ctx["run"](out, ctx["ins"], ctx["args"])
        return code != 0, ("" if code != 0 else "accepted a missing faces= option")
    code, stdout, stderr = ctx["run"](out, ctx["ins"], ctx["args"] + [f"faces={faces}"])
    if code != 0:
        return False, f"exit {code}: {(stdout + stderr).strip()[:200]}"
    v, f = ctx["mcv"].load(out), ctx["mcv"].load(faces).astype(int)
    inv = mesh_invariants(v, f)
    cs = set(map(tuple, cells.tolist()))
    quads = sum((c[0] + d[0], c[1] + d[1], c[2] + d[2]) not in cs for c in cs
                for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)))
    bad = [k for k, ok in (("edges not on exactly two faces", inv["edges_twice"]), ("vertex fan not one cycle", inv["manifold"]),
                           ("faces != 2 x boundary quads", inv["F"] == 2 * quads),
                           ("area != boundary quads", abs(inv["area"] - quads) < 1e-9),
                           ("signed volume != cell count (orientation or closure)", abs(inv["vol"] - len(cs)) < 1e-9),
                           ("unused vertex", inv["unused"] == 0)) if not ok]
    for k, val in ctx["expect"].items():
        if k in ("V", "chi") and inv[k] != int(val):
            bad.append(f"{k} {inv[k]} != {val}")
    if f"V={inv['V']} E={inv['E']} F={inv['F']} chi={inv['chi']}".replace("=", "=") and f"chi= {inv['chi']}" not in stdout:
        bad.append("printed chi differs from independent chi")
    print(f"     {ctx['name']}: V={inv['V']} E={inv['E']} F={inv['F']} chi={inv['chi']} volume={inv['vol']:g} area={inv['area']:g}")
    return not bad, "; ".join(bad)


def make_glb(v, f, idx="u32", interleave=False, normals=None, colors=None, no_index=False, mode=4, mutate=None):
    """An independent numpy GLB writer with the layout variants the Mojo reader must accept (and, mutated, reject)."""
    import json, struct
    v32 = v.astype("<f4")
    views, accs, blob = [], [], bytearray()

    def add(data, target, stride=None):
        while len(blob) % 4:
            blob.append(0)
        views.append(dict(buffer=0, byteOffset=len(blob), byteLength=len(data), **({"byteStride": stride} if stride else {}),
                          target=target))
        blob.extend(data)
        return len(views) - 1

    attrs = {}
    if interleave:  # positions and normals share one strided bufferView
        nrm = normals if normals is not None else np.zeros_like(v)
        inter = np.concatenate([v32, nrm.astype("<f4")], axis=1)
        bv = add(inter.tobytes(), 34962, 24)
        accs.append(dict(bufferView=bv, byteOffset=0, componentType=5126, count=len(v), type="VEC3",
                         min=v32.min(0).tolist(), max=v32.max(0).tolist()))
        accs.append(dict(bufferView=bv, byteOffset=12, componentType=5126, count=len(v), type="VEC3"))
        attrs = {"POSITION": 0, "NORMAL": 1}
    else:
        bv = add(v32.tobytes(), 34962)
        accs.append(dict(bufferView=bv, componentType=5126, count=len(v), type="VEC3", min=v32.min(0).tolist(),
                         max=v32.max(0).tolist()))
        attrs = {"POSITION": 0}
        if normals is not None:
            accs.append(dict(bufferView=add(normals.astype("<f4").tobytes(), 34962), componentType=5126, count=len(v), type="VEC3"))
            attrs["NORMAL"] = len(accs) - 1
    if colors is not None:
        accs.append(dict(bufferView=add(colors.astype("<f4").tobytes(), 34962), componentType=5126, count=len(v), type="VEC3"))
        attrs["COLOR_0"] = len(accs) - 1
    prim = dict(attributes=attrs, mode=mode)
    if not no_index:
        dt, ct = {"u32": ("<u4", 5125), "u16": ("<u2", 5123), "u8": ("u1", 5121)}[idx]
        accs.append(dict(bufferView=add(f.astype(dt).ravel().tobytes(), 34963), componentType=ct, count=f.size, type="SCALAR"))
        prim["indices"] = len(accs) - 1
    doc = dict(asset=dict(version="2.0"), scene=0, scenes=[dict(nodes=[0])], nodes=[dict(mesh=0)],
               meshes=[dict(primitives=[prim])], buffers=[dict(byteLength=len(blob))], bufferViews=views, accessors=accs)
    if mutate:
        mutate(doc, blob)
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    while len(blob) % 4:
        blob.append(0)
    return (struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def parse_glb(raw):
    """Independent strict reader/validator for GLBs this library writes; returns (arrays dict, problems list)."""
    import json, struct
    bad = []
    magic, ver, length = struct.unpack_from("<III", raw, 0)
    if magic != 0x46546C67 or ver != 2 or length != len(raw):
        bad.append("header")
    jl, jt = struct.unpack_from("<II", raw, 12)
    if jt != 0x4E4F534A or jl % 4:
        bad.append("json chunk")
    doc = json.loads(raw[20:20 + jl])
    bl, bt = struct.unpack_from("<II", raw, 20 + jl)
    if bt != 0x004E4942 or bl % 4 or 28 + jl + bl != len(raw):
        bad.append("bin chunk")
    blob = raw[28 + jl:28 + jl + bl]
    if doc["asset"]["version"] != "2.0" or doc["buffers"][0]["byteLength"] != len(blob):
        bad.append("asset/buffer")
    out = {}
    cts = {5126: "<f4", 5125: "<u4", 5123: "<u2", 5121: "u1"}
    for k, a in enumerate(doc["accessors"]):
        bv = doc["bufferViews"][a["bufferView"]]
        n = {"SCALAR": 1, "VEC3": 3}[a["type"]]
        dt = np.dtype(cts[a["componentType"]])
        if bv["byteOffset"] % 4 or a.get("byteOffset", 0) % dt.itemsize or bv["byteOffset"] + bv["byteLength"] > len(blob):
            bad.append(f"accessor {k} alignment/range")
        out[k] = np.frombuffer(blob, dt, a["count"] * n, bv["byteOffset"] + a.get("byteOffset", 0)).reshape(a["count"], n)
        if "min" in a and (not np.array_equal(out[k].min(0), a["min"]) or not np.array_equal(out[k].max(0), a["max"])):
            bad.append(f"accessor {k} min/max")
    prim = doc["meshes"][0]["primitives"][0]
    res = dict(pos=out[prim["attributes"]["POSITION"]].astype(np.float64), idx=out[prim["indices"]].reshape(-1, 3).astype(np.int64),
               nrm=out[prim["attributes"]["NORMAL"]].astype(np.float64) if "NORMAL" in prim["attributes"] else None,
               col=out[prim["attributes"]["COLOR_0"]].astype(np.float64) if "COLOR_0" in prim["attributes"] else None)
    if res["idx"].max() >= len(res["pos"]) or prim.get("mode") != 4:
        bad.append("indices/mode")
    return res, bad


def vertex_normals(v, f):
    c = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], c)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(ln > 0, n / np.maximum(ln, 1e-300), 0.0)


def check_glb(ctx):
    mcv, work, name = ctx["mcv"], ctx["work"], ctx["name"]
    v, f = mcv.load(ctx["ins"][0]), mcv.load(ctx["ins"][1]).astype(np.int64)
    ins = list(ctx["ins"])
    col = None
    if "colors" in ctx["expect"]:
        col = np.random.default_rng(5).random((len(v), 3))
        mcv.save(col, work / f"{name}.col.mcv")
        ins.append(work / f"{name}.col.mcv")
    out = work / f"{name}.glb"
    code, stdout, stderr = ctx["run"](out, ins, ctx["args"])
    if code != 0:
        return False, f"exit {code}: {(stdout + stderr).strip()[:200]}"
    res, bad = parse_glb(out.read_bytes())
    v32 = v.astype("<f4").astype(np.float64)
    if not np.array_equal(res["pos"], v32):
        bad.append("positions differ from float32 input")
    if not np.array_equal(res["idx"], f):
        bad.append("indices differ")
    if np.abs(res["nrm"] - vertex_normals(v32, f)).max() > 1e-6:
        bad.append("normals differ from the area-weighted numpy normals")
    if col is not None and not np.array_equal(res["col"], col.astype("<f4").astype(np.float64)):
        bad.append("colors differ")
    # and back through the Mojo reader
    rv, rf, rn, rc = (work / f"{name}.r{k}.mcv" for k in "vfnc")
    code, stdout, stderr = ctx["run3"]("glb_read", rv, [out], [f"faces={rf}", f"normals={rn}", f"colors={rc}"])
    if code != 0:
        bad.append(f"own reader failed: {(stdout + stderr).strip()[:150]}")
    else:
        if not np.array_equal(mcv.load(rv), v32) or not np.array_equal(mcv.load(rf), f):
            bad.append("round trip through the Mojo reader changed vertices or faces")
        if np.abs(mcv.load(rn) - res["nrm"]).max() > 0:
            bad.append("round trip normals differ")
        if col is not None and not np.array_equal(mcv.load(rc), res["col"]):
            bad.append("round trip colors differ")
    print(f"     {name}: {out.stat().st_size} bytes, V={len(v)} F={len(f)}, validator problems: {bad if bad else 'none'}")
    return not bad, "; ".join(bad)


def check_glb_read(ctx):
    """A GLB made by the numpy writer (a layout variant, or a corrupted one) goes through the Mojo reader."""
    mcv, work, name, variant = ctx["mcv"], ctx["work"], ctx["name"], ctx["expect"]["variant"]
    rs = np.random.default_rng(3)
    v = rs.normal(size=(40, 3))
    f = np.stack([rs.permutation(40)[:3] for _ in range(30)])
    nrm, col = rs.normal(size=(40, 3)), rs.random((40, 3))
    if variant == "noindex":
        v, nrm, col = v[:39], nrm[:39], col[:39]  # sequential triangles need a multiple of 3
    kw, mut = {}, None
    if variant in ("u16", "u8"):
        kw["idx"] = variant
    elif variant == "stride":
        kw.update(interleave=True, normals=nrm)
    elif variant == "noindex":
        kw["no_index"] = True
    elif variant == "all":
        kw.update(normals=nrm, colors=col)
    elif variant == "idx_oob":
        mut = lambda d, b: b.__setitem__(slice(d["bufferViews"][-1]["byteOffset"], d["bufferViews"][-1]["byteOffset"] + 4), (999).to_bytes(4, "little"))
    elif variant == "acc_oob":
        mut = lambda d, b: d["accessors"][0].__setitem__("count", 10 ** 6)
    elif variant == "norm_count":
        kw["normals"] = nrm[:39]
    elif variant == "lines":
        kw["mode"] = 1
    elif variant == "no_pos":
        mut = lambda d, b: d["meshes"][0]["primitives"][0]["attributes"].pop("POSITION")
    raw = make_glb(v, f, mutate=mut, **kw)
    if variant == "bad_magic":
        raw = b"glTX" + raw[4:]
    elif variant == "truncated":
        raw = raw[:len(raw) // 2]
    elif variant == "badlen":
        raw = raw[:8] + (len(raw) + 4).to_bytes(4, "little") + raw[12:]
    elif variant == "json_bad":
        jl = int.from_bytes(raw[12:16], "little")
        raw = raw[:20] + b"{" * jl + raw[20 + jl:]
    elif variant == "empty":
        raw = b""
    path = work / f"{name}.glb"
    path.write_bytes(raw)
    rv, rf, rn, rc = (work / f"{name}.r{k}.mcv" for k in "vfnc")
    code, stdout, stderr = ctx["run"](rv, [path], [f"faces={rf}", f"normals={rn}", f"colors={rc}"])
    if "raises" in ctx["expect"]:
        return code != 0, ("" if code != 0 else "accepted a corrupt GLB")
    if code != 0:
        return False, f"exit {code}: {(stdout + stderr).strip()[:200]}"
    bad = []
    if not np.array_equal(mcv.load(rv), v.astype("<f4").astype(np.float64)):
        bad.append("vertices")
    if variant == "noindex":
        if not np.array_equal(mcv.load(rf), np.arange(39).reshape(-1, 3)):
            bad.append("sequential faces")
    elif not np.array_equal(mcv.load(rf), f):
        bad.append("faces")
    if variant in ("stride",) and not np.array_equal(mcv.load(rn), nrm.astype("<f4").astype(np.float64)):
        bad.append("strided normals")
    if variant == "all" and (not np.array_equal(mcv.load(rn), nrm.astype("<f4").astype(np.float64))
                              or not np.array_equal(mcv.load(rc), col.astype("<f4").astype(np.float64))):
        bad.append("normals/colors")
    return not bad, "; ".join(bad)


def cae_fixtures():
    rs = np.random.default_rng(37)
    special = np.array([[120, 0, 0, 0, 0, 0], [0, 0, 0, 70, 0, 0], [-30, -30, -30, 0, 0, 0], [100, 50, 0, 0, 0, 0],
                        [80, 20, 0, 30, 0, 0]], float)
    r2 = np.sqrt(30 ** 2 + 30 ** 2)
    truth = np.array([[40, 120, 120, 0, 0, 60],
                      [0, 70 * np.sqrt(3), 70, 0, -70, 70],
                      [-30, 0, -30, -30, -30, 0],
                      [50, np.sqrt(7500), 100, 50, 0, 50],
                      [100 / 3, np.sqrt(7900), 50 + r2, 0.0 + 50 - r2, 0, (50 + r2) / 2]], float)
    truth[4, 3], truth[4, 4] = 50 - r2, 0.0  # sigma_z = 0 is the smallest of (92.43, 7.57, 0)
    truth[4] = [100 / 3, np.sqrt(7900), 50 + r2, 50 - r2, 0.0, (50 + r2) / 2]
    near = np.array([[10, 10.0001, 10, 0.0001, 0, 0.00005], [5, 5, 5, 1e-9, 0, 0]], float)
    stops = np.array([0, .25, .5, .75, 1.0]).reshape(-1, 1)
    return {"cae_rand": rs.normal(size=(500, 6)) * 50, "cae_special": special, "cae_special_truth": truth,
            "cae_near": near, "cae_empty": np.zeros((0, 6)), "cae_bad": np.zeros((3, 5)),
            "cm_vals": rs.normal(size=(200, 1)) * 30, "cm_stops": stops, "cm_stops_truth": _STOPS,
            "cm_flat": np.full((4, 1), 7.0)}


# ---- STEP: an independent Part 21 parser and the checks built on it ----
def _p21_value(t, i):
    import re
    while t[i].isspace():
        i += 1
    c = t[i]
    if c == "$":
        return None, i + 1
    if c == "*":
        return "*", i + 1
    if c == "#":
        m = re.compile(r"#(\d+)").match(t, i)
        return ("ref", int(m[1])), m.end()
    if c == "'":
        j = i + 1
        out = ""
        while True:
            if t[j] == "'" and t[j + 1:j + 2] == "'":
                out += "'"
                j += 2
            elif t[j] == "'":
                return out, j + 1
            else:
                out += t[j]
                j += 1
    if c == ".":
        j = t.index(".", i + 1)
        return ("enum", t[i + 1:j]), j + 1
    if c == "(":
        items, i = [], i + 1
        while True:
            while t[i].isspace():
                i += 1
            if t[i] == ")":
                return items, i + 1
            if t[i] == ",":
                i += 1
                continue
            v, i = _p21_value(t, i)
            items.append(v)
    m = re.compile(r"[A-Z_][A-Z0-9_]*").match(t, i)
    if m:
        j = m.end()
        while t[j].isspace():
            j += 1
        args, j = _p21_value(t, j)
        return ("typed", m[0], args), j
    m = re.compile(r"[+-]?\d+(\.\d*)?([eE][+-]?\d+)?").match(t, i)
    return float(m[0]), m.end()


def p21_parse(text):
    import re
    assert text.startswith("ISO-10303-21;") and text.rstrip().endswith("END-ISO-10303-21;"), "envelope"
    assert "FILE_SCHEMA" in text and "AP242" in text, "schema"
    data = re.search(r"DATA;(.*)ENDSEC;\s*END-ISO-10303-21;", text, re.S)[1]
    data = re.sub(r"/\*.*?\*/", "", data, flags=re.S)
    ents, i = {}, 0
    while True:
        while i < len(data) and data[i].isspace():
            i += 1
        if i >= len(data):
            return ents
        m = re.compile(r"#(\d+)\s*=\s*").match(data, i)
        v, i = _p21_value(data, m.end())
        if isinstance(v, list):
            v = ("complex", v)
        assert data[i] == ";", "statement end"
        assert int(m[1]) not in ents, "duplicate id"
        ents[int(m[1])] = v
        i += 1


def _refs(v, acc):
    if isinstance(v, tuple) and v[0] == "ref":
        acc.add(v[1])
    elif isinstance(v, tuple) and v[0] == "typed":
        _refs(v[2], acc)
    elif isinstance(v, tuple) and v[0] == "complex":
        _refs(v[1], acc)
    elif isinstance(v, list):
        for x in v:
            _refs(x, acc)
    return acc


def _pt(ents, ref):
    return tuple(ents[ref[1]][2][1])


def step_triangles(ents):
    """[(p0, p1, p2) of coordinates, ids] of every FACE under the FACETED_BREP, plus directed edges by point id."""
    tris, edges = [], []
    brep = [v for v in ents.values() if isinstance(v, tuple) and v[0] == "typed" and v[1] == "FACETED_BREP"]
    for b in brep:
        for fref in ents[b[2][1][1]][2][1]:
            bound = ents[ents[fref[1]][2][1][0][1]]
            loop = ents[bound[2][1][1]][2][1]
            tris.append(tuple(_pt(ents, r) for r in loop))
            ids = [r[1] for r in loop]
            edges += [(ids[k], ids[(k + 1) % len(ids)]) for k in range(len(ids))]
    return tris, edges


def check_step(ctx):
    import re
    mcv, work, name = ctx["mcv"], ctx["work"], ctx["name"]
    v, f = mcv.load(ctx["ins"][0]), mcv.load(ctx["ins"][1]).astype(int)
    out = work / f"{name}.stp"
    code, stdout, stderr = ctx["run"](out, ctx["ins"], ctx["args"])
    if code != 0:
        return False, f"exit {code}: {(stdout + stderr).strip()[:200]}"
    text = out.read_text()
    bad = []
    if re.search(r"(?<![.\d])\d+[eE]", text) or re.search(r"\d\.?\d*e", text):
        bad.append("a real is not in Part 21 form (digit, '.', optional E exponent)")
    ents = p21_parse(text)
    allrefs = set()
    for val in ents.values():
        _refs(val, allrefs)
    if not allrefs <= set(ents):
        bad.append("dangling references")
    kinds = {(v_[1] if v_[0] == "typed" else "complex") for v_ in ents.values() if isinstance(v_, tuple)}
    for need in ("PRODUCT", "PRODUCT_DEFINITION", "SHAPE_DEFINITION_REPRESENTATION", "ADVANCED_BREP_SHAPE_REPRESENTATION",
                 "FACETED_BREP", "CLOSED_SHELL", "FACE", "FACE_OUTER_BOUND", "POLY_LOOP", "CARTESIAN_POINT"):
        if need not in kinds:
            bad.append(f"missing {need}")
    tris, edges = step_triangles(ents)
    want = sorted(tuple(map(tuple, v[t].tolist())) for t in f)
    if sorted(tris) != want:
        bad.append("triangles differ from the mesh")
    es = set(edges)
    if len(es) != len(edges) or any((b, a) not in es for a, b in es):
        bad.append("shell not closed and consistently oriented (every directed edge needs its reverse once)")
    # back through the Mojo reader
    rv, rf = work / f"{name}.rv.mcv", work / f"{name}.rf.mcv"
    code, so, se = ctx["run3"]("step_read", rv, [out], [f"faces={rf}"])
    if code != 0:
        bad.append(f"own reader failed: {(so + se).strip()[:150]}")
    else:
        got = sorted(tuple(map(tuple, mcv.load(rv)[t].tolist())) for t in mcv.load(rf).astype(int))
        if got != want or len(mcv.load(rv)) != len(v):
            bad.append("round trip through the Mojo reader changed the mesh")
    print(f"     {name}: {len(ents)} entities, {len(tris)} faces, {out.stat().st_size} bytes, problems: {bad if bad else 'none'}")
    return not bad, "; ".join(bad)


def check_step_surface(ctx):
    mcv, work, name = ctx["mcv"], ctx["work"], ctx["name"]
    net = mcv.load(ctx["ins"][0])
    p, q = int(dict(a.split("=") for a in ctx["args"])["p"]), int(dict(a.split("=") for a in ctx["args"])["q"])
    out = work / f"{name}.stp"
    code, stdout, stderr = ctx["run"](out, ctx["ins"], ctx["args"])
    if code != 0:
        return False, f"exit {code}: {(stdout + stderr).strip()[:200]}"
    ents = p21_parse(out.read_text())
    bad = []
    surf = [v_ for v_ in ents.values() if isinstance(v_, tuple) and (
        (v_[0] == "complex" and any(m[1] == "B_SPLINE_SURFACE" for m in v_[1])) or
        (v_[0] == "typed" and v_[1] == "B_SPLINE_SURFACE_WITH_KNOTS"))]
    if len(surf) != 1:
        return False, "expected exactly one surface entity"
    s = surf[0]
    parts = {m[1]: m[2] for m in s[1]} if s[0] == "complex" else None
    if parts:
        pp, qq, grid = parts["B_SPLINE_SURFACE"][0], parts["B_SPLINE_SURFACE"][1], parts["B_SPLINE_SURFACE"][2]
        mu, mv, ku, kv = parts["B_SPLINE_SURFACE_WITH_KNOTS"][:4]
        w = np.array(parts["RATIONAL_B_SPLINE_SURFACE"][0])
    else:
        a = s[2]
        pp, qq, grid, mu, mv, ku, kv = a[1], a[2], a[3], a[8], a[9], a[10], a[11]
        w = np.ones((len(grid), len(grid[0])))
    cp = np.array([[_pt(ents, r) for r in row] for row in grid])
    got = np.concatenate([cp, w[..., None]], axis=2)
    if (int(pp), int(qq)) != (p, q) or got.shape != net.shape or np.abs(got - net).max() > 1e-14 * max(1, np.abs(net).max()):
        bad.append("control net, weights or degrees differ")
    kn_u = np.repeat(ku, [int(m) for m in mu])
    kn_v = np.repeat(kv, [int(m) for m in mv])
    if np.abs(kn_u - _knots(net.shape[0], p)).max() > 1e-14 or np.abs(kn_v - _knots(net.shape[1], q)).max() > 1e-14:
        bad.append("knot vectors expanded from multiplicities are not the clamped uniform ones")
    rn = work / f"{name}.rn.mcv"
    code, so, se = ctx["run3"]("step_read", work / f"{name}.rv.mcv", [out], [f"faces={work / (name + '.rf.mcv')}", f"net={rn}"])
    if code != 0:
        bad.append(f"own reader failed: {(so + se).strip()[:150]}")
    else:
        back = mcv.load(rn)
        if back.shape != net.shape or np.abs(back - net).max() > 1e-14 * max(1, np.abs(net).max()):
            bad.append("round trip through the Mojo reader changed the net")
        if f"p= {p} q= {q}" not in so:
            bad.append("reader degrees")
    kind = "rational" if parts else "plain"
    print(f"     {name}: {kind} surface {net.shape[0]}x{net.shape[1]} p={p} q={q}, problems: {bad if bad else 'none'}")
    return not bad, "; ".join(bad)


def check_step_read(ctx):
    """Our reader on a numpy-side edited copy of a good file: accepted layouts and rejected corruptions."""
    import re
    mcv, work, name, variant = ctx["mcv"], ctx["work"], ctx["name"], ctx["expect"]["variant"]
    if not all((work / f"mesh_one.{suffix}.mcv").exists() for suffix in ("v", "f")):
        code, so, se = ctx["run3"]("mesh", work / "mesh_one.v.mcv", [work / "ms_one.mcv"],
                                    [f"faces={work / 'mesh_one.f.mcv'}"])
        if code != 0:
            return False, f"mesh fixture failed: {(so + se).strip()[:200]}"
    base = (work / "step_base.stp")
    if not base.exists():
        code, so, se = ctx["run3"]("step", base, [work / "mesh_one.v.mcv", work / "mesh_one.f.mcv"], [])
        assert code == 0, so + se
    t = base.read_text()
    lenient = []
    if variant == "pretty":  # whitespace and comments the format allows
        t = re.sub(r",(?=[^']*(?:'[^']*'[^']*)*$)", ",\n   ", t.split("DATA;")[0]) + "DATA;" + re.sub(
            r",", " ,\n  /* c */ ", t.split("DATA;")[1].replace("\n", " \n\n "))
    elif variant == "unknown_entity":
        t = t.replace("ENDSEC;\nEND-", "#9000=CIRCLE('',#16,1.);\nENDSEC;\nEND-")
    elif variant == "unknown_lenient":
        t = t.replace("ENDSEC;\nEND-", "#9000=CIRCLE('',#16,1.);\nENDSEC;\nEND-")
        lenient = ["lenient=1"]
    elif variant == "dangling":
        t = t.replace("POLY_LOOP('',(#20,", "POLY_LOOP('',(#99999,", 1)
    elif variant == "no_end":
        t = t.replace("END-ISO-10303-21;", "")
    elif variant == "truncated":
        t = t[: len(t) // 2]
    elif variant == "dup_id":
        t = t.replace("#21=CARTESIAN_POINT", "#20=CARTESIAN_POINT", 1)
    elif variant == "open_string":
        t = t.replace("'part','part'", "'part,'part", 1)
    elif variant == "bad_char":
        t = t.replace("#20=CARTESIAN_POINT", "#20=CARTESIAN_POINT@", 1)
    elif variant == "quad":
        t = t.replace("POLY_LOOP('',(#20,#21,#22", "POLY_LOOP('',(#20,#21,#22,#23", 1)
    elif variant == "empty":
        t = ""
    elif variant == "no_geometry":
        t = re.sub(r"#\d+=(FACETED_BREP|CLOSED_SHELL)[^\n]*\n", "", t)
    path = work / f"{name}.stp"
    path.write_text(t)
    rv, rf = work / f"{name}.rv.mcv", work / f"{name}.rf.mcv"
    code, so, se = ctx["run"](rv, [path], [f"faces={rf}", *lenient])
    if "raises" in ctx["expect"]:
        return code != 0, ("" if code != 0 else "accepted a corrupt or unsupported STEP file")
    if code != 0:
        return False, f"exit {code}: {(so + se).strip()[:200]}"
    v = mcv.load(work / "mesh_one.v.mcv")
    f = mcv.load(work / "mesh_one.f.mcv").astype(int)
    got = sorted(tuple(map(tuple, mcv.load(rv)[tt].tolist())) for tt in mcv.load(rf).astype(int))
    want = sorted(tuple(map(tuple, v[tt].tolist())) for tt in f)
    return got == want, "" if got == want else "mesh differs"


CUSTOM = {"mesh": check_mesh, "glb": check_glb, "glb_read": check_glb_read, "step": check_step,
          "step_surface": check_step_surface, "step_read": check_step_read}


def mesh_fixtures():
    def box(a, b, c, o=(0, 0, 0)):
        return [(x + o[0], y + o[1], z + o[2]) for x in range(a) for y in range(b) for z in range(c)]
    arr = lambda cs: np.array(sorted(set(cs)), np.int32).reshape(-1, 3)
    ring = [c for c in box(3, 3, 1) if c != (1, 1, 0)]
    shell = [c for c in box(3, 3, 3) if c != (1, 1, 1)]
    rs = np.random.default_rng(31)
    blob = np.unique(rs.integers(0, 6, (120, 3)), axis=0).astype(np.int32)
    return {
        "ms_one": arr([(0, 0, 0)]), "ms_cube3": arr(box(3, 3, 3)), "ms_bar": arr(box(4, 1, 2)),
        "ms_ring": arr(ring), "ms_shell": arr(shell), "ms_two": arr(box(1, 1, 1) + box(1, 1, 1, (3, 0, 0))),
        "ms_edge": arr([(0, 0, 0), (1, 1, 0)]), "ms_corner": arr([(0, 0, 0), (1, 1, 1)]),
        "ms_L": arr(box(2, 1, 1) + box(1, 2, 1) + box(1, 1, 2)), "ms_blob": blob,
        "ms_neg": arr([(-1, -1, -1), (0, -1, -1)]),
        # two cells touching along one edge whose end points are each joined through a chain of cells below and above
        "ms_pinch": arr([(0, 0, 1), (1, 1, 1), (0, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 2), (1, 1, 2), (0, 1, 2)]),
    }
