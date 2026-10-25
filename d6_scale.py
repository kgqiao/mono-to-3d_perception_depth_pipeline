# d6_scale.py (D6, D10, D12): compare scale sources on the same sequence.
#   python d6_scale.py --root data/kitti --seq 00 --n 500 --depth-dir out/depth/00 [--lidar-dir out/lidar/00] [--oxts data/raw/2011_10_03_drive_0027_sync]
#   python d6_scale.py --synthetic
# Variants: A unit (baseline), B oracle (true step length: an upper bound, not a method), C plane, D pnp (depth model),
#           E lidar (D10), F imu clean and F' imu noisy (D12). Metric variants are scored with RIGID alignment only (se3).
import argparse
import numpy as np
from pipeline import SyntheticSource, KittiSource, run_vo
from traj import ate, rpe, kitti_drift
from plots import topdown, scale_over_time

ap = argparse.ArgumentParser()
ap.add_argument("--root"); ap.add_argument("--seq", default="00"); ap.add_argument("--n", type=int, default=None)
ap.add_argument("--depth-dir"); ap.add_argument("--lidar-dir"); ap.add_argument("--oxts")
ap.add_argument("--oxts-start", type=int, default=0, help="first raw frame of this sequence (08 starts at 1100; see the devkit readme)")
ap.add_argument("--images", help="use this image folder instead of image_0 (corrupted copies)")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

if a.synthetic:
    src = SyntheticSource(n=80)
else:
    oxts = None
    if a.oxts:                                       # D12: match OXTS rows to images (sync drives: one row per image)
        from kitti_io import read_oxts
        vf, t = read_oxts(a.oxts); k = a.oxts_start
        oxts = (vf[k:], np.diff(t)[k:])
    src = KittiSource(a.root, a.seq, n=a.n, depth_dir=a.depth_dir, lidar_dir=a.lidar_dir, oxts=oxts, images=a.images)

variants = [("A unit", "unit", None), ("B oracle", "oracle", None), ("C plane", "plane", None)]
if a.synthetic or a.depth_dir: variants.append(("D pnp-depth", "pnp", None))
if a.synthetic or a.lidar_dir: variants.append(("E lidar", "lidar", None))
if a.synthetic or a.oxts:
    variants += [("F imu clean", "imu", None), ("F imu noisy", "imu", {"sigma": 0.1, "bias_rate": 0.01, "seed": 0})]

gt = src.T; trajs = {"ground truth": gt}; scales = {}
true_steps = np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1)
print(f"{'variant':<14} {'ATE se3 m':>10} {'ATE sim3 m':>11} {'RPE m':>8} {'drift %':>8} {'scale err %':>12}")
for name, v, noise in variants:
    est, raw = run_vo(src, v, imu_noise=noise)
    d = kitti_drift(est, gt)
    serr = 100 * np.nanmedian(np.abs(raw - true_steps) / true_steps)
    print(f"{name:<14} {ate(est[:, :3, 3], gt[:, :3, 3], 'se3'):10.2f} {ate(est[:, :3, 3], gt[:, :3, 3], 'sim3'):11.2f} "
          f"{rpe(est, gt)['trans']:8.3f} {d['trans_pct']:8.2f} {serr:12.1f}")
    trajs[name] = est; scales[name] = raw
scales["true"] = true_steps
tag = "synthetic" if a.synthetic else a.seq
topdown(trajs, f"out/d6_{tag}_topdown.png", f"scale variants, {tag}")
scale_over_time(scales, f"out/d6_{tag}_scale.png")
print(f"wrote out/d6_{tag}_topdown.png and out/d6_{tag}_scale.png")
print("Note: 'drift %' needs paths of 100 m or more; on short runs it is nan. Use evo or the KITTI devkit for final numbers.")
