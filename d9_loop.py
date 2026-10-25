# d9_loop.py (D9, optional): measured loop vs estimated path.
#   python d9_loop.py --traj out/phone_est.txt --measured 120.0
import argparse
from kitti_io import read_poses
from sensors import path_length, closure_gap

ap = argparse.ArgumentParser()
ap.add_argument("--traj", required=True, help="poses in KITTI format (12 numbers per line)")
ap.add_argument("--measured", type=float, required=True, help="loop length you measured, meters")
a = ap.parse_args()
p = read_poses(a.traj)[:, :3, 3]
L, gap = path_length(p), closure_gap(p)
print(f"estimated length {L:.1f} m vs measured {a.measured:.1f} m -> scale error {100 * (L / a.measured - 1):+.1f}%")
print(f"start-to-end gap {gap:.2f} m = {100 * gap / a.measured:.1f}% of the loop (truth: 0)")
