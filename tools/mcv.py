"""MCV1 read/write for the cv2 oracle (numpy). Format: b"MCV1", u32 rows, u32 cols, u16 ch, u16 dtype code, data."""
import struct
import numpy as np

CODES = {0: np.uint8, 1: np.float32, 2: np.int32, 3: np.float64, 4: np.int16}


def load(path):
    b = open(path, "rb").read()
    if b[:4] != b"MCV1":
        raise ValueError(f"not an MCV1 file: {path}")
    rows, cols, ch, code = struct.unpack_from("<IIHH", b, 4)
    a = np.frombuffer(b, CODES[code], offset=16).reshape(rows, cols, ch)
    return a[:, :, 0] if ch == 1 else a


def save(a, path):
    a = np.ascontiguousarray(a)
    code = {v: k for k, v in CODES.items()}[a.dtype.type]
    ch = 1 if a.ndim == 2 else a.shape[2]
    with open(path, "wb") as f:
        f.write(b"MCV1" + struct.pack("<IIHH", a.shape[0], a.shape[1], ch, code) + a.tobytes())
