# d8_boxes.py (D8, optional): remove the ground, cluster, fit oriented boxes.
#   python d8_boxes.py --depth out/depth/00/000000.npy --calib data/kitti/sequences/00/calib.txt --cam 2 --image data/kitti/sequences/00/image_2/000000.png
#   python d8_boxes.py --synthetic
import argparse
import numpy as np
from geom import backproject, ransac_plane, height_above, K_from_P
from boxes import cluster_objects, fit_box_bev, car_sized

ap = argparse.ArgumentParser()
ap.add_argument("--depth"); ap.add_argument("--calib"); ap.add_argument("--cam", type=int, default=2); ap.add_argument("--image")
ap.add_argument("--eps", type=float, default=0.6); ap.add_argument("--min-samples", type=int, default=30)
ap.add_argument("--synthetic", action="store_true")
a = ap.parse_args()

if a.synthetic:                                     # a road plus one 4.2 x 1.8 x 1.5 m car, 30 deg heading, 12 m ahead
    rng = np.random.default_rng(0)
    road = np.c_[rng.uniform(-8, 8, 20000), np.full(20000, 1.65), rng.uniform(3, 40, 20000)]
    th = np.radians(30); c, s = np.cos(th), np.sin(th)
    loc = np.c_[rng.uniform(-2.1, 2.1, 4000), rng.uniform(-0.9, 0.9, 4000)]
    xz = loc @ np.array([[c, s], [-s, c]]) + [2.0, 12.0]
    car = np.c_[xz[:, 0], 1.65 - rng.uniform(0, 1.5, 4000), xz[:, 1]]
    pts = np.vstack([road, car])
else:
    from kitti_io import read_calib
    pts, _, _ = backproject(np.load(a.depth), K_from_P(read_calib(a.calib)[f"P{a.cam}"]))
    pts = pts[pts[:, 2] < 40]

plane, ground = ransac_plane(pts, thresh=0.15)
h = height_above(pts, plane)
obj = pts[(~ground) & (h > 0.2) & (h < 3.0)]
hobj = height_above(obj, plane)
labels = cluster_objects(obj, a.eps, a.min_samples)
print(f"ground: {ground.sum()} points, camera height {plane[3]:.2f} m; objects: {len(obj)} points, {labels.max() + 1} clusters")
for k in range(labels.max() + 1):
    m = labels == k
    b = fit_box_bev(obj[m][:, [0, 2]], hobj[m])     # assumes a level camera: x and z are ground coordinates
    tag = "car-sized" if car_sized(b) else "other"
    print(f"cluster {k}: {m.sum():5d} pts  center ({b['cx']:.1f}, {b['cz']:.1f}) m  L {b['length']:.2f} W {b['width']:.2f} "
          f"H {b['height']:.2f}  heading {np.degrees(b['heading']):.0f} deg  [{tag}]")
