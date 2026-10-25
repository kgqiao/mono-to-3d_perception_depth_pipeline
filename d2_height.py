# d2_height.py (D2): ground plane and camera height over many frames.
#   python d2_height.py --depth-dir out/depth/00 --calib data/kitti/sequences/00/calib.txt --cam 2 --n 30
#   python d2_height.py --synthetic
import argparse
import glob
import os
import numpy as np
from geom import backproject, ransac_plane, K_from_P
from plots import hist

ap = argparse.ArgumentParser()
ap.add_argument("--depth-dir"); ap.add_argument("--calib"); ap.add_argument("--cam", type=int, default=2)
ap.add_argument("--n", type=int, default=30); ap.add_argument("--thresh", type=float, default=0.10)
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

if a.synthetic:
    import synth
    T = synth.make_trajectory(a.n + 1); W = synth.make_world(T); K = synth.K
    maps = [synth.depth_image(W, T[i]) * np.random.default_rng(i).uniform(0.95, 1.05) for i in range(a.n)]
else:
    from kitti_io import read_calib
    K = K_from_P(read_calib(a.calib)[f"P{a.cam}"])
    maps = [np.load(f) for f in sorted(glob.glob(os.path.join(a.depth_dir, "*.npy")))[:a.n]]

heights, tilts = [], []
for d in maps:
    h = d.shape[0]
    d = d.copy(); d[: h // 2] = 0                       # lower half of the image first (TODO: widen later)
    pts, _, _ = backproject(d, K)
    plane, inl = ransac_plane(pts, thresh=a.thresh)
    heights.append(plane[3])
    tilts.append(np.degrees(np.arccos(np.clip(-plane[1], -1, 1))))   # angle between normal and straight up
heights = np.array(heights)
print(f"camera height: median {np.median(heights):.3f} m, spread (p10-p90) {np.percentile(heights, 10):.3f}-{np.percentile(heights, 90):.3f}; truth about 1.65")
print(f"scale factor 1.65 / median height = {1.65 / np.median(heights):.3f}; normal tilt median {np.median(tilts):.1f} deg")
os.makedirs("out", exist_ok=True)
hist(heights, "out/d2_heights.png", "estimated camera height (m)")
print("wrote out/d2_heights.png")
