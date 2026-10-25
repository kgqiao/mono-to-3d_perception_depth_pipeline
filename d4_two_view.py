# d4_two_view.py (D4): relative pose from consecutive frames vs ground truth.
#   python d4_two_view.py --root data/kitti --seq 00 --n 200 [--min-motion 0.2]
#   python d4_two_view.py --synthetic
import argparse
import numpy as np
from vo import relative_pose, gt_relative, rotation_error_deg, tdir_error_deg
from pipeline import SyntheticSource, KittiSource
from plots import hist

ap = argparse.ArgumentParser()
ap.add_argument("--root"); ap.add_argument("--seq", default="00"); ap.add_argument("--n", type=int, default=200)
ap.add_argument("--min-motion", type=float, default=0.0, help="skip pairs whose true step is shorter (m)")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()
src = SyntheticSource(n=60) if a.synthetic else KittiSource(a.root, a.seq, n=a.n)

rot, tdir, inls, skipped = [], [], [], 0
for i in range(len(src) - 1):
    g = gt_relative(src.T[i], src.T[i + 1])
    if np.linalg.norm(g[:3, 3]) < a.min_motion:
        skipped += 1
        continue
    p1, p2 = src.matches(i)
    if len(p1) < 8:
        continue
    T_est, inl = relative_pose(p1, p2, src.K)
    if T_est is None:
        continue
    rot.append(rotation_error_deg(T_est[:3, :3], g[:3, :3]))
    tdir.append(tdir_error_deg(T_est[:3, 3], g[:3, 3]))
    inls.append(int(inl.sum()))
print(f"{len(rot)} pairs ({skipped} skipped as nearly stopped)")
print(f"rotation error: median {np.median(rot):.3f} deg, p90 {np.percentile(rot, 90):.3f}")
print(f"translation direction error: median {np.median(tdir):.2f} deg, p90 {np.percentile(tdir, 90):.2f}")
print(f"RANSAC inliers: median {int(np.median(inls))}, min {min(inls)}")
hist(rot, "out/d4_rot_err.png", "rotation error (deg)"); hist(tdir, "out/d4_tdir_err.png", "translation direction error (deg)")
print("wrote out/d4_rot_err.png, out/d4_tdir_err.png")
