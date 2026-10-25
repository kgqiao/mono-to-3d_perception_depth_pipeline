# d9_run.py (D9, optional): run VO on your own footage with ground-plane scale from the height you held the phone at.
#   ffmpeg -i loop.mp4 -vf fps=10 data/phone/%06d.png
#   python d9_run.py --images data/phone --calib out/phone_calib.npz --cam-height 1.40 --out out/phone_est.txt
#   python d9_loop.py --traj out/phone_est.txt --measured 120
import argparse
from pipeline import FolderSource, run_vo
from kitti_io import save_poses
from plots import topdown

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True); ap.add_argument("--calib", required=True)
ap.add_argument("--cam-height", type=float, required=True, help="phone height above the ground, meters (measure it)")
ap.add_argument("--step", type=int, default=1, help="use every Nth frame (walking is slow; try 3)")
ap.add_argument("--variant", default="plane", choices=["unit", "plane"]); ap.add_argument("--out", default="out/phone_est.txt")
a = ap.parse_args()
src = FolderSource(a.images, a.calib, a.step)
est, raw = run_vo(src, a.variant, cam_height=a.cam_height)
save_poses(a.out, est)
topdown({"your loop": est}, a.out.replace(".txt", ".png"), "own footage")
print(f"{len(est)} poses -> {a.out}; plane scale found on {int((raw == raw).sum())} of {len(raw)} steps")
