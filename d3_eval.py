# d3_eval.py (D3): depth metrics with the Eigen protocol (Garg crop, 80 m cap) and by distance bucket.
#   python d3_eval.py --pred-dir out/depth/eigen --gt-dir data/eigen_gt --scale none|median
#   python d3_eval.py --synthetic
# pred-dir: <name>.npy predictions; gt-dir: <name>.png KITTI depth PNGs (or <name>.npy in meters), same names.
import argparse
import glob
import os
import numpy as np
from geom import metrics, garg_crop, load_depth_png

ap = argparse.ArgumentParser()
ap.add_argument("--pred-dir"); ap.add_argument("--gt-dir"); ap.add_argument("--scale", default="none")
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()
BUCKETS = [(0, 20), (20, 40), (40, 80)]


def pairs():
    if a.synthetic:
        import synth
        T = synth.make_trajectory(6); W = synth.make_world(T)
        for i in range(5):
            gt = synth.depth_image(W, T[i])
            pred = np.where(gt > 0, gt * (1 + 0.004 * gt), 0)           # error grows with distance, like real models
            yield f"synthetic_{i}", pred, gt
        return
    for f in sorted(glob.glob(os.path.join(a.pred_dir, "*.npy"))):
        name = os.path.splitext(os.path.basename(f))[0]
        g = os.path.join(a.gt_dir, name + ".png")
        gt = load_depth_png(g) if os.path.exists(g) else np.load(os.path.join(a.gt_dir, name + ".npy"))
        yield name, np.load(f), gt


rows, by = [], {b: [] for b in BUCKETS}
for name, pred, gt in pairs():
    crop = garg_crop(*gt.shape)
    gt = np.where(crop, gt, 0)
    if a.scale == "median":                                             # only for relative-depth models; say so in results
        m = (gt > 0) & (pred > 0)
        pred = pred * np.median(gt[m]) / np.median(pred[m])
    rows.append(metrics(pred, gt))
    for lo, hi in BUCKETS:
        by[(lo, hi)].append(metrics(pred, np.where((gt >= lo) & (gt < hi), gt, 0)))


def mean(rs, k):
    rs = [r for r in rs if r["n_pixels"] > 0]
    return np.mean([r[k] for r in rs]) if rs else float("nan")


print(f"{len(rows)} images, scale alignment: {a.scale}")
print(f"{'':>10} {'AbsRel':>8} {'RMSE':>8} {'d<1.25':>8}")
print(f"{'all':>10} {mean(rows, 'abs_rel'):8.4f} {mean(rows, 'rmse'):8.3f} {mean(rows, 'delta1'):8.4f}")
for (lo, hi), rs in by.items():
    print(f"{f'{lo}-{hi} m':>10} {mean(rs, 'abs_rel'):8.4f} {mean(rs, 'rmse'):8.3f} {mean(rs, 'delta1'):8.4f}")
