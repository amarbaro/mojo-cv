#!/usr/bin/env bash
# Downloads the two Kaggle datasets behind tests/cases/real.txt (needs the kaggle CLI and its credentials) and converts
# them to .mcv under tests/real/. Not part of scripts/ci.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REAL="$ROOT/tests/real"
mkdir -p "$REAL"
PY="${MCV_PY:-$ROOT/.venv/bin/python}"
# any video file with a few seconds of ordinary footage; its first frame feeds the orb_real row
: "${MCV_VIDEO:?set MCV_VIDEO to a video file}"

# chessboard photos
if [ ! -d "$REAL/data/imgs/leftcamera" ]; then
  (cd "$REAL" && kaggle datasets download -d danielwe14/stereocamera-chessboard-pictures -p . --unzip)
fi
# real ArUco photos
if [ ! -d "$REAL/aruco_yolo" ]; then
  mkdir -p "$REAL/aruco_yolo"
  (cd "$REAL/aruco_yolo" && kaggle datasets download -d rahul14112003/aruco-yolo -p . --unzip)
fi

$PY - "$REAL" <<'PY'
import cv2, sys, os, glob, shutil
real=sys.argv[1]
sys.path.insert(0, os.path.join(real,'..','..','tools'))
import mcv, numpy as np

# 5 chessboard photos
names=[1,10,11,12,13]
for i,n in enumerate(names):
    p=os.path.join(real,'data','imgs','leftcamera',f'Im_L_{n}.png')
    im=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    assert im is not None, p
    mcv.save(im, os.path.join(real, f'chess_{i}.mcv'))
    objp=np.zeros((11*7,3),dtype=np.float64)
    objp[:,:2]=np.mgrid[0:11,0:7].T.reshape(-1,2)
    K=np.array([[800,0,im.shape[1]//2],[0,800,im.shape[0]//2],[0,0,1]],dtype=np.float64)
    dist=np.zeros((5,1),dtype=np.float64)
    ok,corners=cv2.findChessboardCorners(im,(11,7))
    assert ok
    mcv.save(objp, os.path.join(real, f'pnp_obj_{i}.mcv'))
    mcv.save(corners.reshape(-1,2).astype(np.float64), os.path.join(real, f'pnp_img_{i}.mcv'))
    mcv.save(K, os.path.join(real, f'pnp_K_{i}.mcv'))
    mcv.save(dist, os.path.join(real, f'pnp_d_{i}.mcv'))

# 5 real ArUco photos
imgs=sorted(glob.glob(os.path.join(real,'aruco_yolo','valid','images','*.jpg')))
chosen=[]
for p in imgs:
    im=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    det=cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), cv2.aruco.DetectorParameters())
    corners,ids,_=det.detectMarkers(im)
    if ids is not None and len(ids)>=1:
        chosen.append(p)
    if len(chosen)>=5: break
assert len(chosen)==5, len(chosen)
for i,p in enumerate(chosen):
    im=cv2.imread(p,cv2.IMREAD_GRAYSCALE)
    mcv.save(im, os.path.join(real, f'aruco_{i}.mcv'))

# one video frame
frame=os.path.join(real,'frame0.png')
if not os.path.exists(frame):
    import subprocess
    subprocess.run(['ffmpeg','-y','-i',os.environ['MCV_VIDEO'],'-vf','scale=640:360,format=gray','-frames:v','1',frame],check=True,stderr=subprocess.DEVNULL)
im=cv2.imread(frame,cv2.IMREAD_GRAYSCALE)
mcv.save(im, os.path.join(real,'frame0.mcv'))
print('fetch_fixtures: converted real inputs')
PY