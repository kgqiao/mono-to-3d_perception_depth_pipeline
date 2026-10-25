# d10_lidar_depth.py (D10, optional): turn Velodyne scans into sparse depth maps aligned with the images.
#   python d10_lidar_depth.py --root data/kitti --seq 00 --cam 0 --out out/lidar/00 --n 500
# Then: python d6_scale.py ... --lidar-dir out/lidar/00
import argparse
import glob
import os
import cv2
import numpy as np
from kitti_io import read_calib, read_velo, image_paths
from sensors import project_velo

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True); ap.add_argument("--seq", default="00"); ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=0)
a = ap.parse_args()
sdir = os.path.join(a.root, "sequences", a.seq)
cal = read_calib(os.path.join(sdir, "calib.txt"))
imgs = image_paths(sdir, a.cam); imgs = imgs[:a.n] if a.n else imgs
os.makedirs(a.out, exist_ok=True)
for f in imgs:
    name = os.path.splitext(os.path.basename(f))[0]
    h, w = cv2.imread(f, 0).shape
    uv, z, keep = project_velo(read_velo(os.path.join(sdir, "velodyne", name + ".bin")), cal["Tr"], cal[f"P{a.cam}"], (h, w))
    d = np.zeros((h, w), np.float32)
    ui, vi, zz = uv[keep, 0].astype(int), uv[keep, 1].astype(int), z[keep]
    order = np.argsort(-zz); d[vi[order], ui[order]] = zz[order]           # nearest point wins
    np.save(os.path.join(a.out, name + ".npy"), d)
print(f"wrote {len(imgs)} sparse LiDAR depth maps to {a.out}")
