# d5_vo.py (D5): chain unit-length relative poses into a trajectory and score it.
#   python d5_vo.py --root data/kitti --seq 00 --n 500
#   python d5_vo.py --synthetic
# Writes out/d5_<seq>_est.txt (KITTI format, for evo), out/d5_<seq>_frames.csv (for D7), and a top-down plot.
import argparse
import csv
import numpy as np
from pipeline import SyntheticSource, KittiSource, run_vo
from traj import ate, rpe, kitti_drift
from kitti_io import save_poses
from vo import gt_relative, rotation_error_deg
from plots import topdown

ap = argparse.ArgumentParser()
ap.add_argument("--root"); ap.add_argument("--seq", default="00"); ap.add_argument("--n", type=int, default=None)
ap.add_argument("--images", help="use this image folder instead of image_0 (e.g. data/corrupt/04/fog_2)")
ap.add_argument("--tag", help="name for the output files (default: the sequence)")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()
src = SyntheticSource(n=80) if a.synthetic else KittiSource(a.root, a.seq, n=a.n, images=a.images)
tag = a.tag or ("synthetic" if a.synthetic else a.seq)

log = []
est, _ = run_vo(src, "unit", log=log)
gt = src.T
pe, pg = est[:, :3, 3], gt[:, :3, 3]

print(f"{len(est)} poses. Unit-scale baseline, so ATE needs scale alignment (sim3). This is NOT a metric result.")
print(f"ATE sim3 = {ate(pe, pg, 'sim3'):.3f} m   ATE se3 (no scale) = {ate(pe, pg, 'se3'):.3f} m")
print("KITTI-style drift (unit scale, so only meaningful after you fix scale in D6):", kitti_drift(est, gt))
print("RPE (1 frame):", {k: round(v, 4) for k, v in rpe(est, gt).items()})
save_poses(f"out/d5_{tag}_est.txt", est); save_poses(f"out/d5_{tag}_gt.txt", gt)

with open(f"out/d5_{tag}_frames.csv", "w", newline="") as f:              # per-frame facts for D7
    w = csv.writer(f); w.writerow(["frame", "inliers", "true_step_m", "turn_deg", "rot_err_deg"])
    for r in log:
        i = r["frame"]; g = gt_relative(gt[i], gt[i + 1]); e = np.linalg.inv(est[i]) @ est[i + 1]
        w.writerow([i, r["inliers"], np.linalg.norm(g[:3, 3]), rotation_error_deg(np.eye(3), g[:3, :3]),
                    rotation_error_deg(e[:3, :3], g[:3, :3])])
from traj import umeyama

s, R, t = umeyama(pe, pg, with_scale=True)
aligned = est.copy(); aligned[:, :3, 3] = (s * (R @ pe.T)).T + t

topdown({"ground truth": gt, "VO, unit scale (sim3-aligned)": aligned}, f"out/d5_{tag}_topdown.png", f"sequence {tag}")
print(f"wrote out/d5_{tag}_est.txt, _gt.txt, _frames.csv, _topdown.png")
print(f"evo:  evo_ape kitti out/d5_{tag}_gt.txt out/d5_{tag}_est.txt -as --plot     (-as = align with scale)")
