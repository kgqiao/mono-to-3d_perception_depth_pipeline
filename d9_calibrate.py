# d9_calibrate.py (D9, optional): calibrate your phone camera from 15-20 checkerboard photos.
#   python d9_calibrate.py --images data/checker/*.jpg --cols 9 --rows 6 --square 0.025
# cols/rows = INNER corners. Lock zoom, focus and stabilization while filming and while taking these photos.
import argparse
import glob
import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True); ap.add_argument("--cols", type=int, default=9); ap.add_argument("--rows", type=int, default=6)
ap.add_argument("--square", type=float, default=0.025, help="square size in meters"); ap.add_argument("--out", default="out/phone_calib.npz")
a = ap.parse_args()
obj = np.zeros((a.rows * a.cols, 3), np.float32)
obj[:, :2] = np.mgrid[0:a.cols, 0:a.rows].T.reshape(-1, 2) * a.square
objs, imgs, size = [], [], None
for f in sorted(glob.glob(a.images)):
    g = cv2.imread(f, 0); size = g.shape[::-1]
    ok, c = cv2.findChessboardCorners(g, (a.cols, a.rows))
    if ok:
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        objs.append(obj); imgs.append(c)
print(f"corners found in {len(imgs)} images")
rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
print(f"reprojection RMS {rms:.3f} px (under about 0.5 is good)\nK =\n{K}\ndistortion = {dist.ravel()}")
np.savez(a.out, K=K, dist=dist, size=size)
print(f"saved {a.out}")
