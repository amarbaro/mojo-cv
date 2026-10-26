"""cv2 5.0 oracle and test driver for mojo-cv.

  oracle_cv2.py OP OUT IN... [k=v]     run one op exactly as `.bin/mojo-cv OP OUT IN... [k=v]` would
  oracle_cv2.py --fixtures DIR         write the .mcv inputs the case rows name
  oracle_cv2.py --suite [--bench] CASES...   run every row through ours and cv2, compare, print PASS/FAIL

A case row is `id op IN... [k=v...] => cmp[, cmp...]`; one cmp per output file (OUT, OUT.1, ...):
  exact | max:T | mean:T | frac:T:F (share of values within T) | iou:F (nonzero masks)
  pts:T:F (share of ref rows with one of ours within distance T) | cnt:R (|n_ours - n_ref| <= R * n_ref)
"""
import os, re, subprocess, sys, time, struct
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import mcv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX = os.path.join(ROOT, "tests", "fixtures")
DT = {0: np.uint8, 1: np.float32, 2: np.int32, 3: np.float64, 4: np.int16}
OPS = {}


def op(*names):
    def deco(f):
        for n in names or [f.__name__]:
            OPS[n] = f
        return f
    return deco


def kv_f(kv, k, d):
    return float(kv[k]) if k in kv else d


def kv_i(kv, k, d):
    return int(float(kv[k])) if k in kv else d


# ---- core -------------------------------------------------------------------------------------------------
@op()
def flip(i, kv): return cv2.flip(i[0], kv_i(kv, "code", 1))
@op()
def rotate(i, kv): return cv2.rotate(i[0], [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][kv_i(kv, "code", 0)])
@op()
def transpose(i, kv): return cv2.transpose(i[0])
@op()
def absdiff(i, kv): return cv2.absdiff(i[0], i[1])
@op()
def bitwise_and(i, kv): return cv2.bitwise_and(i[0], i[1])
@op()
def bitwise_or(i, kv): return cv2.bitwise_or(i[0], i[1])
@op()
def bitwise_xor(i, kv): return cv2.bitwise_xor(i[0], i[1])
@op()
def bitwise_not(i, kv): return cv2.bitwise_not(i[0])
@op()
def convert(i, kv):
    """convertTo: saturate_cast(round_half_even(a * alpha + beta)), computed in float64."""
    dt = DT[kv_i(kv, "to", 0)]
    r = i[0].astype(np.float64) * kv_f(kv, "alpha", 1.0) + kv_f(kv, "beta", 0.0)
    if np.issubdtype(dt, np.integer):
        r = np.clip(np.rint(r), np.iinfo(dt).min, np.iinfo(dt).max)
    return r.astype(dt)


# ---- imgproc ----------------------------------------------------------------------------------------------
@op()
def GaussianBlur(i, kv):
    k = kv_i(kv, "ksize", 5)
    return cv2.GaussianBlur(i[0], (k, k), kv_f(kv, "sigma", 0.0))
@op()
def pyrDown(i, kv): return cv2.pyrDown(i[0])
@op()
def Sobel(i, kv):
    return cv2.Sobel(i[0], cv2.CV_16S if i[0].dtype == np.uint8 else -1, kv_i(kv, "dx", 1), kv_i(kv, "dy", 0),
                     ksize=kv_i(kv, "ksize", 3), scale=kv_f(kv, "scale", 1.0))


@op()
def threshold(i, kv):
    t, mv = kv_f(kv, "thresh", 127.0), kv_f(kv, "maxval", 255.0)
    ty = kv_i(kv, "type", 0) | (cv2.THRESH_OTSU if kv_i(kv, "otsu", 0) else 0) | (cv2.THRESH_TRIANGLE if kv_i(kv, "tri", 0) else 0)
    r, o = cv2.threshold(i[0], t, mv, ty)
    return [o, np.array([[r]], np.float64)]
@op()
def integral(i, kv): return cv2.integral(i[0], sdepth=cv2.CV_32S if i[0].dtype == np.uint8 else cv2.CV_64F)
SHAPES = [cv2.MORPH_RECT, cv2.MORPH_CROSS, cv2.MORPH_ELLIPSE]
def _se(kv):
    k = kv_i(kv, "ksize", 3)
    return cv2.getStructuringElement(SHAPES[kv_i(kv, "shape", 0)], (k, k))
@op()
def erode(i, kv): return cv2.erode(i[0], _se(kv), iterations=kv_i(kv, "iters", 1))
@op()
def dilate(i, kv): return cv2.dilate(i[0], _se(kv), iterations=kv_i(kv, "iters", 1))
@op()
def cvtColor(i, kv):
    c = {"bgr2gray": cv2.COLOR_BGR2GRAY, "rgb2gray": cv2.COLOR_RGB2GRAY, "gray2bgr": cv2.COLOR_GRAY2BGR, "bgr2rgb": cv2.COLOR_BGR2RGB,
         "bgr2hsv": cv2.COLOR_BGR2HSV, "hsv2bgr": cv2.COLOR_HSV2BGR, "bgr2ycrcb": cv2.COLOR_BGR2YCrCb, "ycrcb2bgr": cv2.COLOR_YCrCb2BGR,
         "bgr2lab": cv2.COLOR_BGR2Lab, "lab2bgr": cv2.COLOR_Lab2BGR, "bgr2luv": cv2.COLOR_BGR2Luv, "luv2bgr": cv2.COLOR_Luv2BGR}
    return cv2.cvtColor(i[0], c[kv["code"]])


INTERP = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4, cv2.INTER_AREA]
BORDER = [cv2.BORDER_REFLECT_101, cv2.BORDER_REPLICATE, cv2.BORDER_CONSTANT]  # border=0 reflect101, 1 replicate, 2 constant


def kv_l(kv, k, d=None):
    return [float(x) for x in kv[k].split(",")] if k in kv else d
@op()
def resize(i, kv):
    h, w = i[0].shape[:2]
    return cv2.resize(i[0], (kv_i(kv, "w", w), kv_i(kv, "h", h)), interpolation=INTERP[kv_i(kv, "interp", 1)])
@op()
def warpAffine(i, kv):
    h, w = i[0].shape[:2]
    fl = INTERP[kv_i(kv, "interp", 1)] | (cv2.WARP_INVERSE_MAP if kv_i(kv, "inv", 0) else 0)
    return cv2.warpAffine(i[0], np.array(kv_l(kv, "M", [1, 0, 0, 0, 1, 0])).reshape(2, 3), (kv_i(kv, "w", w), kv_i(kv, "h", h)), flags=fl,
                          borderMode=BORDER[kv_i(kv, "border", 2)], borderValue=(kv_f(kv, "bval", 0.0),) * 4)
@op()
def remap(i, kv):
    return cv2.remap(i[0], i[1], i[2], INTERP[kv_i(kv, "interp", 1)], borderMode=BORDER[kv_i(kv, "border", 2)], borderValue=(kv_f(kv, "bval", 0.0),) * 4)
@op()
def blobFromImage(i, kv):
    h, w = i[0].shape[:2]
    b = cv2.dnn.blobFromImage(i[0], kv_f(kv, "scale", 1.0), (kv_i(kv, "w", w), kv_i(kv, "h", h)), tuple(kv_l(kv, "mean", [0, 0, 0])), bool(kv_i(kv, "swap", 0)), False)
    return b.reshape(-1, b.shape[3])


@op()
def connectedComponentsWithStats(i, kv):
    n, lab, st, cen = cv2.connectedComponentsWithStats(i[0], connectivity=kv_i(kv, "connectivity", 8), ltype=cv2.CV_32S)
    return [lab, st, cen]
@op()
def Canny(i, kv):
    return cv2.Canny(i[0], kv_f(kv, "t1", 100.0), kv_f(kv, "t2", 200.0), apertureSize=kv_i(kv, "aperture", 3), L2gradient=bool(kv_i(kv, "l2", 0)))
@op()
def findContours(i, kv):
    cs, hi = cv2.findContours(i[0], [cv2.RETR_EXTERNAL, cv2.RETR_LIST, cv2.RETR_CCOMP, cv2.RETR_TREE][kv_i(kv, "mode", 0)], [cv2.CHAIN_APPROX_NONE, cv2.CHAIN_APPROX_SIMPLE][kv_i(kv, "method", 0)])
    pts = np.concatenate([c.reshape(-1, 2) for c in cs]) if cs else np.zeros((0, 2), np.int32)
    hier = np.zeros((0, 4), np.int32) if hi is None else hi.reshape(-1, 4).astype(np.int32)
    return [pts.astype(np.int32), np.array([[len(c)] for c in cs], np.int32).reshape(-1, 1), hier]
@op()
def goodFeaturesToTrack(i, kv):
    r = cv2.goodFeaturesToTrack(i[0], kv_i(kv, "maxCorners", 100), kv_f(kv, "quality", 0.01), kv_f(kv, "minDist", 10.0), blockSize=kv_i(kv, "block", 3), useHarrisDetector=bool(kv_i(kv, "harris", 0)), k=kv_f(kv, "k", 0.04))
    return np.zeros((0, 2), np.float32) if r is None else r.reshape(-1, 2).astype(np.float32)
@op()
def line(i, kv):
    im = i[0].copy()
    c = tuple(kv_l(kv, "color", [255, 255, 255]))
    return cv2.line(im, (kv_i(kv, "x1", 0), kv_i(kv, "y1", 0)), (kv_i(kv, "x2", 0), kv_i(kv, "y2", 0)), c, kv_i(kv, "thick", 1))
@op()
def rectangle(i, kv):
    im = i[0].copy()
    c = tuple(kv_l(kv, "color", [255, 255, 255]))
    return cv2.rectangle(im, (kv_i(kv, "x1", 0), kv_i(kv, "y1", 0)), (kv_i(kv, "x2", 0), kv_i(kv, "y2", 0)), c, kv_i(kv, "thick", 1))


@op()
def bfmatch(i, kv):
    m = cv2.BFMatcher(cv2.NORM_HAMMING).match(i[0], i[1])
    return np.array([[x.trainIdx, int(x.distance)] for x in m], np.int32).reshape(-1, 2)
@op()
def knnmatch(i, kv):
    k = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(i[0], i[1], k=2)
    tab = np.array([[m[0].trainIdx, int(m[0].distance), m[1].trainIdx, int(m[1].distance)] for m in k], np.int32).reshape(-1, 4)
    r = kv_f(kv, "ratio", 0.75)
    good = np.array([[q, m[0].trainIdx, int(m[0].distance)] for q, m in enumerate(k) if m[0].distance < r * m[1].distance], np.int32).reshape(-1, 3)
    return [tab, good]


@op()
def cornerSubPix(i, kv):
    w = kv_i(kv, "win", 5)
    z = kv_i(kv, "zero", -1)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, kv_i(kv, "maxIter", 30), kv_f(kv, "eps", 0.01))
    return cv2.cornerSubPix(i[0], i[1].reshape(-1, 1, 2).copy(), (w, w), (z, z), crit).reshape(-1, 2)
@op()
def calcOpticalFlowPyrLK(i, kv):
    w = kv_i(kv, "win", 21)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, kv_i(kv, "maxCount", 30), kv_f(kv, "eps", 0.01))
    nxt, st, err = cv2.calcOpticalFlowPyrLK(i[0], i[1], i[2].reshape(-1, 1, 2).copy(), None, winSize=(w, w), maxLevel=kv_i(kv, "maxLevel", 3), criteria=crit)
    return [nxt.reshape(-1, 2), st.reshape(-1, 1), err.reshape(-1, 1)]


@op()
def calcOpticalFlowFarneback(i, kv):
    return cv2.calcOpticalFlowFarneback(i[0], i[1], None, kv_f(kv, "pyrScale", 0.5), kv_i(kv, "levels", 3), kv_i(kv, "winsize", 15),
                                        kv_i(kv, "iters", 3), kv_i(kv, "polyN", 5), kv_f(kv, "polySigma", 1.2), 0)


# ---- calib3d (all arrays float64; points are (n, 2) or (n, 3)) --------------------------------------------
@op()
def Rodrigues(i, kv): return cv2.Rodrigues(i[0])[0]
@op()
def projectPoints(i, kv): return cv2.projectPoints(i[0], i[1], i[2], i[3], i[4])[0].reshape(-1, 2)
@op()
def fisheye_projectPoints(i, kv): return cv2.fisheye.projectPoints(i[0].reshape(-1, 1, 3), i[1], i[2], i[3], i[4])[0].reshape(-1, 2)
@op()
def undistortPoints(i, kv):
    R = i[3] if len(i) > 3 else None
    P = i[4] if len(i) > 4 else None
    return cv2.undistortPoints(i[0].reshape(-1, 1, 2), i[1], i[2], R=R, P=P).reshape(-1, 2)
@op()
def fisheye_undistortPoints(i, kv):
    R = i[3] if len(i) > 3 else None
    P = i[4] if len(i) > 4 else None
    return cv2.fisheye.undistortPoints(i[0].reshape(-1, 1, 2), i[1], i[2], R=R, P=P).reshape(-1, 2)
@op()
def initUndistortRectifyMap(i, kv):
    return list(cv2.initUndistortRectifyMap(i[0], i[1], i[2], i[3], (kv_i(kv, "w", 640), kv_i(kv, "h", 480)), cv2.CV_32FC1))
@op()
def fisheye_initUndistortRectifyMap(i, kv):
    return list(cv2.fisheye.initUndistortRectifyMap(i[0], i[1], i[2], i[3], (kv_i(kv, "w", 640), kv_i(kv, "h", 480)), cv2.CV_32FC1))


@op()
def estimateAffine2D(i, kv):
    m, inl = cv2.estimateAffine2D(i[0], i[1], method=cv2.RANSAC, ransacReprojThreshold=kv_f(kv, "thresh", 3.0), maxIters=kv_i(kv, "maxIters", 2000), confidence=kv_f(kv, "conf", 0.99))
    return [m, inl.reshape(-1, 1).astype(np.uint8)]
@op()
def solvePnP(i, kv):
    flag = cv2.SOLVEPNP_EPNP if kv_i(kv, "epnp", 0) else cv2.SOLVEPNP_ITERATIVE
    ok, r, t = cv2.solvePnP(i[0], i[1], i[2], i[3] if i[3].size else None, flags=flag)
    if kv_i(kv, "epnp", 0):
        r, t = cv2.solvePnPRefineLM(i[0], i[1], i[2], i[3], r, t)
    return [r.reshape(3, 1), t.reshape(3, 1)]
@op()
def solvePnPRansac(i, kv):
    ok, r, t, inl = cv2.solvePnPRansac(i[0], i[1], i[2], i[3] if i[3].size else None, iterationsCount=kv_i(kv, "iters", 100),
                                      reprojectionError=kv_f(kv, "thresh", 8.0), confidence=kv_f(kv, "conf", 0.99), flags=cv2.SOLVEPNP_ITERATIVE)
    return [r.reshape(3, 1), t.reshape(3, 1), np.sort(inl.reshape(-1, 1).astype(np.int32), axis=0)]
@op()
def findEssentialMat(i, kv):
    E, m = cv2.findEssentialMat(i[0], i[1], i[2], method=cv2.RANSAC, prob=kv_f(kv, "prob", 0.999), threshold=kv_f(kv, "thresh", 1.0))
    if len(i[0]) == 5:
        models = E.reshape(-1, 3, 3)
        models = models / np.linalg.norm(models, axis=(1, 2))[:, None, None]
        for e in models:
            if e.flat[np.argmax(np.abs(e))] < 0:
                e *= -1
        return [models[np.argsort(models[:, 0, 0])].reshape(-1, 3), m.reshape(-1, 1).astype(np.uint8)]
    E = E[:3] / np.linalg.norm(E[:3])
    if E.flat[np.argmax(np.abs(E))] < 0:
        E = -E
    return [E, m.reshape(-1, 1).astype(np.uint8)]
@op()
def recoverPose(i, kv):
    mask = i[4].copy() if len(i) > 4 else None
    n, R, t, _ = cv2.recoverPose(i[0], i[1], i[2], i[3], mask=mask)
    return [R, t.reshape(3, 1), np.array([[n]], np.int32)]
@op()
def stereoRectify(i, kv):
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(i[0], i[1], i[2], i[3], (kv_i(kv, "w", 640), kv_i(kv, "h", 480)), i[4], i[5],
                                                 flags=cv2.CALIB_ZERO_DISPARITY if kv_i(kv, "zero", 1) else 0, alpha=kv_f(kv, "alpha", -1.0))
    return [R1, R2, P1, P2, Q]


@op()
def StereoSGBM(i, kv):
    b = kv_i(kv, "block", 5)
    sg = cv2.StereoSGBM_create(minDisparity=kv_i(kv, "minDisp", 0), numDisparities=kv_i(kv, "numDisp", 32), blockSize=b, P1=kv_i(kv, "P1", 8 * b * b),
                               P2=kv_i(kv, "P2", 32 * b * b), disp12MaxDiff=kv_i(kv, "lrMax", 0), uniquenessRatio=kv_i(kv, "uniq", 0))
    return sg.compute(i[0], i[1])


@op()
def findChessboardCorners(i, kv):
    ok, c = cv2.findChessboardCorners(i[0], (kv_i(kv, "cols", 9), kv_i(kv, "rows", 6)))
    return c.reshape(-1, 2).astype(np.float32) if ok else np.zeros((0, 2), np.float32)


@op()
def aruco_detect(i, kv):
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(i[0])
    if ids is None:
        return [np.zeros((0, 1), np.int32), np.zeros((0, 2), np.float32)]
    o = np.argsort(ids.ravel())
    return [ids.reshape(-1, 1)[o].astype(np.int32), np.concatenate([corners[k].reshape(4, 2) for k in o]).astype(np.float32)]


@op()
def ORB(i, kv):
    orb = cv2.ORB_create(nfeatures=kv_i(kv, "nfeatures", 500), fastThreshold=kv_i(kv, "fastThreshold", 20))
    kps, d = orb.detectAndCompute(i[0], None)
    if d is None:
        return np.zeros((0, 36), np.float32)
    rows = [[k.pt[0], k.pt[1], k.angle, k.octave] + list(r) for k, r in zip(kps, d)]
    return np.array(rows, np.float32)


PY4 = os.environ.get("MCV_PY4", os.path.join(ROOT, ".venv4", "bin", "python"))  # cv2 4.8.1 for the HOG rows, see scripts/ci.sh
HOG_OPS = ("HOGScores", "detectMultiScale", "groupRectangles")  # cv2 5.0 dropped HOGDescriptor: these run under cv2 4.8.1 (PY4)


def _hog(svm):
    h = cv2.HOGDescriptor()
    h.setSVMDetector(svm.reshape(-1, 1).astype(np.float32))
    return h
@op()
def HOGScores(i, kv):
    d = _hog(i[1]).compute(i[0], (8, 8), (0, 0)).reshape(-1, 3780)
    nx = (i[0].shape[1] - 64) // 8 + 1
    return (d @ i[1].ravel()[:3780] + i[1].ravel()[3780]).reshape(-1, nx).astype(np.float32)
@op()
def detectMultiScale(i, kv):
    r, _ = _hog(i[1]).detectMultiScale(i[0], hitThreshold=kv_f(kv, "hit", 0.0), winStride=(8, 8), padding=(0, 0), scale=kv_f(kv, "scale", 1.05), groupThreshold=kv_i(kv, "group", 2))
    return np.array(r, np.int32).reshape(-1, 4)


@op()
def groupRectangles(i, kv):
    r, _ = cv2.groupRectangles(i[0].reshape(-1, 4).tolist(), kv_i(kv, "thresh", 2), 0.2)
    return np.array(r, np.int32).reshape(-1, 4)


@op()
def grabCut(i, kv):
    r = [int(x) for x in kv_l(kv, "rect", [0, 0, i[0].shape[1], i[0].shape[0]])]
    mask = np.zeros(i[0].shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    cv2.grabCut(i[0], mask, tuple(r), bgd, fgd, kv_i(kv, "iters", 5), cv2.GC_INIT_WITH_RECT)
    return ((mask & 1) * 255).astype(np.uint8)


# ---- compare ----------------------------------------------------------------------------------------------
def compare(ours, ref, spec):
    if ours.shape != ref.shape and not spec.startswith(("orb", "pts", "cnt")):
        return False, f"shape {ours.shape} vs {ref.shape}"
    if ours.dtype != ref.dtype:
        return False, f"dtype {ours.dtype} vs {ref.dtype}"
    k, *p = spec.split(":")
    p = [float(x) for x in p]
    o, r = ours.astype(np.float64), ref.astype(np.float64)
    d = np.abs(o - r) if o.shape == r.shape else None
    if k == "exact":
        return bool((d == 0).all()), f"maxdiff {d.max() if d.size else 0}"
    if k == "max":
        return bool((d <= p[0]).all()), f"maxdiff {d.max() if d.size else 0} (tol {p[0]})"
    if k == "mean":
        return bool(d.mean() <= p[0]), f"meandiff {d.mean():.4g} (tol {p[0]})"
    if k == "frac":
        f = (d <= p[0]).mean() if d.size else 1.0
        return bool(f >= p[1]), f"within {p[0]}: {f:.4f} (want {p[1]})"
    if k == "iou":
        a, b = ours != 0, ref != 0
        u = (a | b).sum()
        v = (a & b).sum() / u if u else 1.0
        return bool(v >= p[0]), f"iou {v:.4f} (want {p[0]})"
    if k == "cnt":
        n, m = len(ours), len(ref)
        return bool(abs(n - m) <= p[0] * max(m, 1)), f"count {n} vs {m}"
    if k == "orb":  # orb:T:F:H rows are x, y, angle, octave, 32 descriptor bytes
        if len(ref) == 0:
            return True, "no ref keypoints"
        if len(ours) == 0:
            return False, "no keypoints"
        hits, ham = 0, []
        for rr in ref:
            same = ours[ours[:, 3] == rr[3]]
            if len(same) == 0:
                continue
            dist = np.hypot(same[:, 0] - rr[0], same[:, 1] - rr[1])
            j = dist.argmin()
            if dist[j] <= p[0]:
                hits += 1
                ham.append(int(np.unpackbits((same[j, 4:].astype(np.uint8) ^ rr[4:].astype(np.uint8))).sum()))
        f = hits / len(ref)
        mh = float(np.mean(ham)) if ham else 1e9
        return bool(f >= p[1] and mh <= p[2]), f"kp matched {f:.3f} (want {p[1]}), mean hamming {mh:.1f} (want <= {p[2]}), ours {len(ours)} ref {len(ref)}"
    if k == "pts":
        a, b = o.reshape(len(o), -1), r.reshape(len(r), -1)
        if len(b) == 0:
            return True, "no ref points"
        if len(a) == 0:
            return False, "no points"
        dist = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(1)
        f = (dist <= p[0]).mean()
        return bool(f >= p[1]), f"ref points matched within {p[0]}: {f:.4f} (want {p[1]})"
    raise ValueError("unknown cmp " + spec)


# ---- fixtures ---------------------------------------------------------------------------------------------
def scene(seed, h=1080, w=1920):
    """A deterministic 1080p test picture with the statistics the ops care about: flat panels with sharp edges, a
    smooth colour field, text at several sizes, diagonal lines, circles, and noisy texture patches."""
    r = np.random.default_rng(seed)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([120 + 80 * np.sin(gx / 310 + seed) * np.cos(gy / 270), 110 + 70 * np.cos(gx / 190 - gy / 410),
                    100 + 90 * np.sin((gx + gy) / 520 + seed)], axis=2)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for _ in range(90):
        x0, y0 = int(r.integers(0, w - 60)), int(r.integers(0, h - 40))
        x1, y1 = x0 + int(r.integers(30, 420)), y0 + int(r.integers(20, 260))
        cv2.rectangle(img, (x0, y0), (min(x1, w - 1), min(y1, h - 1)), tuple(int(c) for c in r.integers(0, 256, 3)), -1)
    for _ in range(60):
        org = (int(r.integers(0, w - 200)), int(r.integers(30, h - 10)))
        cv2.putText(img, "mojo-cv %d" % int(r.integers(0, 99999)), org, cv2.FONT_HERSHEY_SIMPLEX, float(r.uniform(0.4, 2.2)),
                    tuple(int(c) for c in r.integers(0, 256, 3)), int(r.integers(1, 4)), cv2.LINE_AA)
    for _ in range(50):
        cv2.line(img, (int(r.integers(0, w)), int(r.integers(0, h))), (int(r.integers(0, w)), int(r.integers(0, h))),
                 tuple(int(c) for c in r.integers(0, 256, 3)), int(r.integers(1, 4)), cv2.LINE_AA)
    for _ in range(40):
        cv2.circle(img, (int(r.integers(0, w)), int(r.integers(0, h))), int(r.integers(8, 90)),
                   tuple(int(c) for c in r.integers(0, 256, 3)), int(r.choice([-1, 1, 2, 3])), cv2.LINE_AA)
    for _ in range(25):
        x0, y0 = int(r.integers(0, w - 200)), int(r.integers(0, h - 150))
        ph, pw = int(r.integers(40, 150)), int(r.integers(60, 200))
        patch = img[y0:y0 + ph, x0:x0 + pw].astype(np.int16) + r.integers(-40, 41, (ph, pw, 3))
        img[y0:y0 + ph, x0:x0 + pw] = np.clip(patch, 0, 255).astype(np.uint8)
    return img


def make_fixtures(d):
    """Inputs named by the case rows. hd* are 1080p synthetic scenes (`scene`); the rest are synthetic too."""
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(7)
    A = scene(11)
    # the second frame: the same scene seen 0.6 degrees and (6.3, -4.6) px later, with sensor noise and a few new panels
    R = np.random.default_rng(12)
    Mb = cv2.getRotationMatrix2D((960, 540), 0.6, 1.0)
    Mb[:, 2] += (6.3, -4.6)
    B = cv2.warpAffine(A, Mb, (1920, 1080), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    B = np.clip(B.astype(np.int16) + R.integers(-3, 4, B.shape), 0, 255).astype(np.uint8)
    for _ in range(12):
        x0, y0 = int(R.integers(0, 1700)), int(R.integers(0, 900))
        cv2.rectangle(B, (x0, y0), (x0 + int(R.integers(30, 200)), y0 + int(R.integers(20, 120))), tuple(int(c) for c in R.integers(0, 256, 3)), -1)
    fx = {"hd": A[:, :1920], "hd2": B[:, :1920]}
    fx["hdg"] = cv2.cvtColor(fx["hd"], cv2.COLOR_BGR2GRAY)
    fx["hdg2"] = cv2.cvtColor(fx["hd2"], cv2.COLOR_BGR2GRAY)
    fx["sm"], fx["sm2"] = fx["hd"][300:570, 400:880].copy(), fx["hd2"][300:570, 400:880].copy()
    fx["smg"], fx["smg2"] = fx["hdg"][300:570, 400:880].copy(), fx["hdg2"][300:570, 400:880].copy()
    fx["sm_lab"], fx["sm_luv"] = cv2.cvtColor(fx["sm"], cv2.COLOR_BGR2Lab), cv2.cvtColor(fx["sm"], cv2.COLOR_BGR2Luv)
    smf = fx["sm"].astype(np.float32) / 255
    fx["sm_lab_f"], fx["sm_luv_f"] = cv2.cvtColor(smf, cv2.COLOR_BGR2Lab), cv2.cvtColor(smf, cv2.COLOR_BGR2Luv)
    fx["bin"] = (fx["smg"] < 128).astype(np.uint8) * 255
    fx["binhd"] = (fx["hdg"] < 128).astype(np.uint8) * 255
    shapes = np.zeros((60, 80), np.uint8)
    cv2.rectangle(shapes, (5, 5), (30, 30), 255, -1)
    cv2.rectangle(shapes, (12, 12), (22, 22), 0, -1)
    cv2.circle(shapes, (55, 30), 15, 255, -1)
    cv2.circle(shapes, (55, 30), 6, 0, -1)
    cv2.line(shapes, (3, 50), (70, 57), 255, 1)
    shapes[40, 40] = 255
    shapes[0, 0:5] = 255
    fx["shapes"] = shapes
    fx["blank"] = np.zeros((9, 11), np.uint8)
    fx["full"] = np.full((9, 11), 255, np.uint8)
    M = np.array([[1, 0, 2.6], [0, 1, 1.7]], np.float64)
    hm = cv2.warpAffine(fx["hdg"], M, (1920, 1080), flags=cv2.INTER_LINEAR)
    fx["lk_prev"], fx["lk_next"] = fx["smg"].copy(), hm[300:570, 400:880].copy()
    gp = cv2.goodFeaturesToTrack(fx["smg"], 120, 0.05, 10).reshape(-1, 2).astype(np.float32)
    fx["lk_pts"] = np.vstack([gp, np.array([[3, 3], [476, 266]], np.float32)])
    # cornerSubPix start points: jittered corners where cv2's iteration has converged (10, 50 and 100 iterations agree);
    # on a slowly drifting point the result after 50 iterations is decided by float rounding and cv2's restore rule
    jit = gp + rng.uniform(-1.2, 1.2, gp.shape).astype(np.float32)
    fixed_ = []
    for pt in jit:
        c = [cv2.cornerSubPix(fx["smg"], pt.reshape(1, 1, 2).copy(), (3, 3), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, n, 0.001)) for n in (10, 50, 100)]
        if max(np.abs(c[0] - c[1]).max(), np.abs(c[1] - c[2]).max()) < 1e-3:
            fixed_.append(pt)
    fx["sp_pts"] = np.array(fixed_[:80], np.float32)
    assert len(fx["sp_pts"]) >= 60, len(fixed_)
    da = rng.integers(0, 256, (200, 32), dtype=np.uint8)
    db = rng.integers(0, 256, (300, 32), dtype=np.uint8)
    db[:150] = da[:150] ^ (rng.random((150, 32)) < 0.02).astype(np.uint8)  # 150 near copies of the first 150 queries
    fx["desc_a"], fx["desc_b"] = da, db
    fx["desc_one"] = da[:1].copy()
    # calib3d inputs
    fx["K"] = np.array([[800, 0, 320], [0, 790, 240], [0, 0, 1.0]])
    fx["Kb"] = np.array([[1000, 0, 500], [0, 1000, 400], [0, 0, 1.0]])
    fx["d4"] = np.array([[-0.2], [0.05], [0.001], [-0.002]])
    fx["d5"] = np.array([[-0.25], [0.08], [0.001], [-0.002], [-0.02]])
    fx["d8"] = np.array([[-0.2], [0.05], [0.001], [-0.002], [0.01], [0.1], [-0.05], [0.02]])
    fx["d12"] = np.array([[-0.2], [0.05], [0.001], [-0.002], [0.01], [0.1], [-0.05], [0.02], [0.001], [-0.0005], [0.0008], [0.0002]])
    fx["dz"] = np.zeros((5, 1))
    fx["dfe"] = np.array([[-0.05], [0.01], [-0.002], [0.0003]])
    fx["obj3"] = np.column_stack([rng.uniform(-1, 1, 30), rng.uniform(-1, 1, 30), rng.uniform(3, 6, 30)])
    fx["rv"] = np.array([[0.1], [-0.2], [0.05]])
    fx["tv"] = np.array([[0.1], [-0.05], [0.3]])
    fx["rv0"] = np.zeros((3, 1))
    fx["pix"] = np.vstack([rng.uniform(0, 640, 40), rng.uniform(0, 480, 40)]).T
    fx["Rr"] = cv2.Rodrigues(np.array([0.02, -0.03, 0.01]))[0]
    fx["Pn"] = np.array([[700, 0, 330, 0], [0, 700, 250, 0], [0, 0, 1, 0.0]])
    fx["rod_v"] = np.array([[0.3], [-0.7], [0.5]])
    fx["rod_v0"] = np.array([[0.0], [0.0], [0.0]])
    fx["rod_m"] = cv2.Rodrigues(np.array([0.3, -0.7, 0.5]))[0]
    fx["rod_mpi"] = cv2.Rodrigues(np.array([2.2214414691, 2.2214414691, 0.0]))[0]
    fx["rod_mid"] = np.eye(3)
    # geometry fixtures: two views of a point cloud, 3D-2D correspondences, affine pairs
    Kc = fx["K"]
    cloud = np.column_stack([rng.uniform(-1.5, 1.5, 60), rng.uniform(-1, 1, 60), rng.uniform(4, 9, 60)])
    rv2, tv2 = np.array([0.05, -0.12, 0.03]), np.array([[-0.8], [0.05], [0.1]])
    tv2 = tv2 / np.linalg.norm(tv2)
    q1 = cv2.projectPoints(cloud, np.zeros(3), np.zeros(3), Kc, None)[0].reshape(-1, 2)
    q2 = cv2.projectPoints(cloud, rv2, tv2, Kc, None)[0].reshape(-1, 2)
    q2n = q2 + rng.normal(0, 0.3, q2.shape)
    for q in (q2, q2n):
        q[:12] = rng.uniform(0, 640, (12, 2))  # gross outliers
    fx["ess_p1"], fx["ess_p2"], fx["ess_p2n"] = q1, q2, q2n
    fx["ess_six1"], fx["ess_six2"] = q1[12:18].copy(), q2[12:18].copy()
    fx["ess_five1"], fx["ess_five2"] = q1[12:17].copy(), q2[12:17].copy()
    fx["ess_clean1"], fx["ess_clean2"] = q1[12:].copy(), q2[12:].copy()
    tn = tv2.reshape(3)
    fx["pose_E"] = np.array([[0, -tn[2], tn[1]], [tn[2], 0, -tn[0]], [-tn[1], tn[0], 0]]) @ cv2.Rodrigues(rv2)[0]
    fx["pose_mask"] = np.concatenate([np.zeros(12), np.ones(48)]).reshape(-1, 1).astype(np.uint8)
    fx["pnp_obj"] = np.column_stack([rng.uniform(-1, 1, 40), rng.uniform(-0.8, 0.8, 40), rng.uniform(-0.5, 0.5, 40)])
    fx["pnp_obj_pl"] = np.column_stack([rng.uniform(-1, 1, 12), rng.uniform(-0.8, 0.8, 12), np.zeros(12)])
    rvp, tvp = np.array([0.2, -0.3, 0.1]), np.array([0.1, -0.2, 5.0])
    fx["pnp_img"] = cv2.projectPoints(fx["pnp_obj"], rvp, tvp, Kc, fx["d4"])[0].reshape(-1, 2)
    fx["pnp_img_pl"] = cv2.projectPoints(fx["pnp_obj_pl"], rvp, tvp, Kc, fx["d4"])[0].reshape(-1, 2)
    out = fx["pnp_img"].copy()
    out[:12] = rng.uniform(0, 640, (12, 2))
    fx["pnp_img_out"] = out
    outp = fx["pnp_img_pl"].copy()
    outp[:3] = rng.uniform(0, 640, (3, 2))
    fx["pnp_img_pl_out"] = np.vstack([outp, cv2.projectPoints(fx["pnp_obj_pl"][:6], rvp, tvp, Kc, fx["d4"])[0].reshape(-1, 2)])
    fx["pnp_obj_pl2"] = np.vstack([fx["pnp_obj_pl"], fx["pnp_obj_pl"][:6]])
    fx["pnp_four"] = np.array([[-1., -1., 0.], [1., -1., .3], [0., 1., -.2], [.2, .1, 1.]])
    fx["pnp_four_img"] = cv2.projectPoints(fx["pnp_four"], rvp, tvp, Kc, fx["d4"])[0].reshape(-1, 2)
    fx["pnp_img_noise"] = fx["pnp_img"] + np.random.default_rng(103).normal(0, .2, (40, 2))
    Af = np.array([[0.9, -0.3, 12.0], [0.25, 1.1, -7.0]])
    pa = rng.uniform(0, 300, (80, 2))
    pb = pa @ Af[:, :2].T + Af[:, 2]
    pb[:16] = rng.uniform(0, 300, (16, 2))
    fx["aff_from"], fx["aff_to"] = pa, pb
    fx["Rst"] = cv2.Rodrigues(np.array([0.01, -0.02, 0.005]))[0]
    fx["Tst"] = np.array([[-0.12], [0.003], [0.002]])
    fx["Tstv"] = np.array([[0.003], [-0.12], [0.002]])
    fx["K2"] = np.array([[820, 0, 315], [0, 830, 245], [0, 0, 1.0]])
    fx["dst1"] = np.array([[-0.2], [0.05], [0], [0], [0]])
    fx["dst2"] = np.array([[-0.1], [0.02], [0], [0], [0]])
    H, W = 120, 200
    base = cv2.GaussianBlur(rng.integers(0, 256, (H, W + 60), dtype=np.uint8), (0, 0), 1.2)
    dm = np.full((H, W), 6, np.int32)
    dm[30:90, 70:150] = 14
    SL = base[:, 30:30 + W].copy()
    SR = np.zeros((H, W), np.uint8)
    for yy in range(H):
        for xx in range(W):
            xr_ = xx - dm[yy, xx]
            if 0 <= xr_ < W:
                SR[yy, xr_] = SL[yy, xx]
    fx["st_l"], fx["st_r"] = SL, SR
    def board(cols_, rows_, sq=40, margin=60):
        b = np.zeros(((rows_ + 1) * sq, (cols_ + 1) * sq), np.uint8)
        for r_ in range(rows_ + 1):
            for c_ in range(cols_ + 1):
                if (r_ + c_) % 2 == 0:
                    b[r_ * sq:(r_ + 1) * sq, c_ * sq:(c_ + 1) * sq] = 255
        big = np.full((b.shape[0] + 2 * margin, b.shape[1] + 2 * margin), 200, np.uint8)
        big[margin:-margin, margin:-margin] = b
        return big
    for nm, (cc_, rr_, Mh) in {"cb_a": (9, 6, [[0.9, 0.15, 40], [-0.1, 0.95, 30], [0.0002, 0.0001, 1]]),
                              "cb_b": (7, 5, [[0.6, -0.5, 250], [0.5, 0.6, 20], [0.0003, -0.0002, 1]]),
                              "cb_c": (6, 4, [[1.2, 0.0, 30], [0.0, 1.2, 40], [0.0, 0.0, 1]])}.items():
        fx[nm] = cv2.GaussianBlur(cv2.warpPerspective(board(cc_, rr_), np.array(Mh), (640, 480), borderValue=200), (3, 3), 0)
    dic = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sc = np.full((480, 640), 190, np.uint8)
    for mid, (ang, scale, ox, oy) in {7: (0.0, 1.0, 40, 40), 23: (0.5, 0.8, 330, 60), 42: (-0.9, 1.1, 120, 270), 3: (2.4, 0.7, 450, 280)}.items():
        mk = cv2.aruco.generateImageMarker(dic, mid, 90)
        mk = cv2.copyMakeBorder(mk, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        Mrot = cv2.getRotationMatrix2D((60, 60), np.degrees(ang), scale)
        Mrot[:, 2] += [ox, oy]
        warped = cv2.warpAffine(mk, Mrot, (640, 480), flags=cv2.INTER_LINEAR, borderValue=190)
        msk = cv2.warpAffine(np.ones_like(mk) * 255, Mrot, (640, 480), flags=cv2.INTER_NEAREST, borderValue=0)
        sc[msk > 0] = warped[msk > 0]
    fx["aruco_a"] = cv2.GaussianBlur(sc, (3, 3), 0)
    sc2 = np.full((240, 320), 120, np.uint8)
    mk = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(dic, 11, 100), 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    P = cv2.getPerspectiveTransform(np.float32([[0, 0], [140, 0], [140, 140], [0, 140]]), np.float32([[60, 40], [200, 55], [215, 190], [50, 170]]))
    wp = cv2.warpPerspective(mk, P, (320, 240), borderValue=120)
    mp = cv2.warpPerspective(np.ones_like(mk) * 255, P, (320, 240), flags=cv2.INTER_NEAREST)
    sc2[mp > 0] = wp[mp > 0]
    fx["aruco_b"] = sc2
    # grabCut inputs: a coloured scene (blobs on a textured background) and a scene crop
    gy, gx = np.mgrid[0:110, 0:150]
    sc3 = np.zeros((110, 150, 3), np.uint8)
    sc3[:] = (40, 90, 30)
    sc3 = np.clip(sc3.astype(np.int16) + rng.integers(-12, 13, sc3.shape), 0, 255).astype(np.uint8)
    cv2.ellipse(sc3, (75, 55), (32, 24), 20, 0, 360, (200, 60, 50), -1)
    cv2.circle(sc3, (60, 50), 10, (230, 200, 40), -1)
    sc3 = np.clip(sc3.astype(np.int16) + rng.integers(-10, 11, sc3.shape), 0, 255).astype(np.uint8)
    fx["gc_scene"] = sc3
    best_c, best_xy = -1, (0, 0)  # the scene crop with the most distinct colours
    for y0 in range(0, 1080 - 110, 20):
        for x0 in range(0, 1920 - 150, 30):
            c_ = len(np.unique(fx["hd"][y0:y0 + 110:3, x0:x0 + 150:3].reshape(-1, 3), axis=0))
            if c_ > best_c:
                best_c, best_xy = c_, (y0, x0)
    gs = fx["hd"][best_xy[0]:best_xy[0] + 110, best_xy[1]:best_xy[1] + 150].copy()
    obj = np.zeros_like(gs)
    obj[:] = (30, 160, 230)
    obj = np.clip(obj.astype(np.int16) + rng.integers(-25, 26, obj.shape), 0, 255).astype(np.uint8)
    om = np.zeros(gs.shape[:2], np.uint8)
    cv2.ellipse(om, (80, 55), (38, 28), -15, 0, 360, 255, -1)
    gs[om > 0] = obj[om > 0]  # a coloured object on a scene crop
    fx["gc_shot"] = gs
    fx["odd"] = rng.integers(0, 256, (7, 13, 3), dtype=np.uint8)
    fx["odd2"] = rng.integers(0, 256, (7, 13, 3), dtype=np.uint8)
    fx["one"] = np.array([[[200, 0, 77]]], np.uint8)
    fx["one2"] = np.array([[[10, 255, 77]]], np.uint8)
    fx["g1"] = rng.integers(0, 256, (5, 9), dtype=np.uint8)
    fx["g2"] = rng.integers(0, 256, (5, 9), dtype=np.uint8)
    fx["i16"] = rng.integers(-32768, 32768, (6, 11, 2), dtype=np.int16)
    fx["i16b"] = rng.integers(-32768, 32768, (6, 11, 2), dtype=np.int16)
    fx["i32"] = rng.integers(-2**31, 2**31, (6, 11), dtype=np.int32)
    fx["i32b"] = rng.integers(-2**31, 2**31, (6, 11), dtype=np.int32)
    fx["f32"] = (rng.standard_normal((9, 17, 3)) * 300).astype(np.float32)
    fx["f32b"] = (rng.standard_normal((9, 17, 3)) * 300).astype(np.float32)
    fx["f64"] = rng.standard_normal((8, 5)) * 1e3
    fx["half"] = np.array([[-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 254.5, 255.5, 300.0, -7.0]], np.float32)
    for nm, (hh, ww) in {"": (1080, 1920), "s": (270, 480)}.items():
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        fx["mapx" + nm] = (xx + 12 * np.sin(yy / 17) + 0.37).astype(np.float32)
        fx["mapy" + nm] = (yy + 9 * np.cos(xx / 23) - 0.21).astype(np.float32)
    fx["mapxo"] = rng.uniform(-3, 16, (7, 13)).astype(np.float32)
    fx["mapyo"] = rng.uniform(-3, 10, (7, 13)).astype(np.float32)
    # raw HOG boxes of cv2 4.x on hdg: the input of the groupRectangles row (hdg.mcv must exist before the call)
    mcv.save(fx["hdg"], os.path.join(d, "hdg.mcv"))
    code = ("import cv2,sys,numpy as np;sys.path.insert(0,%r);import mcv;h=cv2.HOGDescriptor();"
            "h.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector());"
            "r,_=h.detectMultiScale(mcv.load(%r),hitThreshold=-1.0,winStride=(8,8),padding=(0,0),scale=1.05,groupThreshold=0);"
            "mcv.save(np.array(r,np.int32).reshape(-1,4),%r)" % (os.path.join(ROOT, "tools"), os.path.join(d, "hdg.mcv"), os.path.join(d, "hog_raw.mcv")))
    raw = subprocess.run([PY4, "-c", code], capture_output=True, text=True)
    if raw.returncode:
        raise SystemExit("FAIL fixtures: HOG oracle (cv2 4.x at %s): %s" % (PY4, raw.stderr.strip()[-300:]))
    # the default people detector (3780 weights + bias) only exists in cv2 4.x
    sv = subprocess.run([PY4, "-c", "import cv2,sys;sys.stdout.buffer.write(cv2.HOGDescriptor_getDefaultPeopleDetector().astype('<f4').tobytes())"], capture_output=True)
    if sv.returncode or len(sv.stdout) != 3781 * 4:
        raise SystemExit("FAIL fixtures: the default people detector from cv2 4.x (%s) did not return 3781 weights" % PY4)
    fx["hog_svm"] = np.frombuffer(sv.stdout, "<f4").reshape(-1, 1).copy()
    for n, a in fx.items():
        mcv.save(a, os.path.join(d, n + ".mcv"))


# ---- driver -----------------------------------------------------------------------------------------------
def outputs(res):
    return res if isinstance(res, (list, tuple)) else [res]


def run_one(opname, out, ins, kv):
    arrs = [mcv.load(p) for p in ins]
    reps = kv_i(kv, "reps", 1)
    best = 1e30
    for _ in range(reps):
        t = time.perf_counter()
        res = OPS[opname](arrs, kv)
        best = min(best, (time.perf_counter() - t) * 1e3)
    for n, a in enumerate(outputs(res)):
        mcv.save(np.asarray(a), out if n == 0 else f"{out}.{n}")
    if reps > 1:
        print("TIME", best)


def parse_row(line):
    left, cmp_ = line.split("=>")
    tok = left.split()
    cid, opn, rest = tok[0], tok[1], tok[2:]
    ins = [t for t in rest if "=" not in t]
    kv = dict(t.split("=", 1) for t in rest if "=" in t)
    return cid, opn, ins, kv, [c.strip() for c in cmp_.split(",")]


def rows(files):
    for f in files:
        for ln in open(f):
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                yield f, parse_row(ln)


def fpath(n):
    return n if "/" in n else os.path.join(FIX, n + ".mcv")


def time_of(text):
    m = re.search(r"TIME ([0-9.e+-]+)", text)
    return float(m.group(1)) if m else None


BENCH_NOTES = """- **Parallel dispatch**: `max.algorithm.parallelize` with the default 16 workers costs about 215 us per region (best of
  50 trivial regions); with 8 workers (the physical cores) it costs about 8 us, so `core.par` uses 8 and takes 16 only for
  regions that run for milliseconds. Allocating and zero-filling a 2 MB `Mat` is 18 us.
- **Read the ratios with care.** One run of this table is one sample on a shared desktop: a row near 1.0 can land on either
  side, and a loaded run of the same binary moves a row by up to a factor of two. `tools/abbench.py OLD_BIN` runs an old
  build, the current build and cv2 back to back per row, which is the fair comparison when the machine is not quiet.
- Point-set ops (projectPoints, estimateAffine2D, stereoRectify) take microseconds on both sides; 0.00 ms entries are
  below the timer's resolution, not a result."""


def write_md(path, table):
    hd = [t for t in table if t[4]]
    other = [t for t in table if not t[4]]

    def fmt(rs):
        out = ["| op | row | ours ms | cv2 ms | ours/cv2 |", "|---|---|---:|---:|---:|"]
        for opn, cid, a, b, _ in rs:
            if a is None or b is None:
                out.append(f"| {opn} | {cid} | n/a | n/a | |")
                continue
            ratio = f"{a / b:.2f}" if b > 0.005 else "n/a"
            out.append(f"| {opn} | {cid} | {a:.2f} | {b:.2f} | {ratio}{' **faster**' if b > 0.005 and a < b else ''} |")
        return "\n".join(out)

    ok = [t for t in hd if t[2] and t[3] and t[3] > 0.005]
    faster = sum(1 for t in ok if t[2] < t[3])
    close = sum(1 for t in ok if t[3] <= t[2] <= 1.5 * t[3])
    slow = ', '.join(sorted({t[1] for t in ok if t[2] > 1.5 * t[3]}))
    with open(path, "w") as f:
        f.write(f"""# 2D speed rows: mojo-cv vs cv2 5.0

Method: `tests/run.sh --bench-all --md docs/bench.md`. Each row runs the same op in one resident process with `reps=5`
and keeps the best wall time of the compute only (file load and save excluded), ours via `.bin/mojo-cv`, cv2 via
`tools/oracle_cv2.py` (numpy in, cv2 call, best of 5). Both sides use all threads of the 7800X3D; CPU only, no GPU.
Inputs are the same `.mcv` fixtures (`hd*` = 1920x1080 synthetic scenes from `scene()` in `tools/oracle_cv2.py`, the rest
smaller). Target: ours <= cv2. **Misses are listed as measured; nothing is hidden.** The HOG rows time cv2 4.8.1
(cv2 5.0 has no HOG).

1080p rows: {faster} of {len(ok)} faster than cv2, {close} more within 1.5x, the rest slower (ratio column).

Slower than cv2 in this run: {slow}.

{BENCH_NOTES}

## 1920x1080 rows

{fmt(hd)}

## Other sizes (270x480 and smaller, point sets, stereo pairs)

{fmt(other)}
""")


def suite(files, bench, every=False, md=None):
    work = os.path.join(ROOT, ".work", "t")
    os.makedirs(work, exist_ok=True)
    exe = os.path.join(ROOT, ".bin", "mojo-cv")
    npass = nfail = 0
    table = []
    for f, (cid, opn, ins, kv, cmps) in rows(files):
        paths = [fpath(n) for n in ins]
        kvs = [f"{k}={v}" for k, v in kv.items()]
        hd = any(n.startswith("hd") for n in ins)
        if bench and not hd and not every:
            continue
        kvs_b = kvs + (["reps=5"] if bench else [])
        o, r = os.path.join(work, cid + ".ours"), os.path.join(work, cid + ".ref")
        for p in (o, r):
            for g in [p] + [f"{p}.{n}" for n in range(1, 6)]:
                if os.path.exists(g):
                    os.remove(g)
        try:
            tr = subprocess.run([PY4 if opn in HOG_OPS else sys.executable, "-I", __file__, opn, r, *paths, *kvs_b], capture_output=True, text=True)
            to = subprocess.run([exe, opn, o, *paths, *kvs_b], capture_output=True, text=True)
            msg, ok = "", True
            if tr.returncode:
                ok, msg = False, "oracle failed: " + tr.stderr.strip().splitlines()[-1]
            elif to.returncode:
                ok, msg = False, "ours failed: " + (to.stdout + to.stderr).strip()[-300:]
            else:
                for n, spec in enumerate(cmps):
                    sfx = "" if n == 0 else f".{n}"
                    if not os.path.exists(r + sfx) or not os.path.exists(o + sfx):
                        ok, msg = False, f"missing output {n}"
                        break
                    good, why = compare(mcv.load(o + sfx), mcv.load(r + sfx), spec)
                    msg += f"[{n}] {spec}: {why}  "
                    ok &= good
        except Exception as e:  # a harness bug must not read as a pass
            ok, msg = False, f"harness error: {e!r}"
        print(("PASS" if ok else "FAIL"), cid, "-", msg.strip(), flush=True)
        npass += ok
        nfail += not ok
        if bench and ok:
            table.append((opn, cid, time_of(to.stdout), time_of(tr.stdout), hd))
    print(f"rows: {npass + nfail}  pass: {npass}  fail: {nfail}")
    if bench:
        print("| op | row | ours ms | cv2 ms | ratio |\n|---|---|---|---|---|")
        for opn, cid, a, b, _ in table:
            print(f"| {opn} | {cid} | {a:.2f} | {b:.2f} | {a / b:.2f} |" if a and b else f"| {opn} | {cid} | {a} | {b} | |")
        if md:
            write_md(md, table)
    return nfail


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--suite":
        bench = "--bench" in a or "--bench-all" in a
        md = a[a.index("--md") + 1] if "--md" in a else None
        files = [x for x in a[1:] if not x.startswith("--") and x != md]
        sys.exit(1 if suite(files, bench, "--bench-all" in a, md) else 0)
    if a and a[0] == "--fixtures":
        make_fixtures(a[1])
        sys.exit(0)
    run_one(a[0], a[1], [x for x in a[2:] if "=" not in x], dict(x.split("=", 1) for x in a[2:] if "=" in x))
