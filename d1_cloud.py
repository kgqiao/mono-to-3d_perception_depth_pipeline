# d1_cloud.py (D1): depth + intrinsics -> point cloud, compared with LiDAR.
#   python d1_cloud.py --depth out/depth/00/000000.npy --calib data/kitti/sequences/00/calib.txt --cam 2 [--lidar lidar.npy] [--show]
#   python d1_cloud.py --synthetic
import argparse
import numpy as np
import open3d as o3d

from geom import backproject, K_from_P #import written functions from geom.py


ap = argparse.ArgumentParser()
ap.add_argument("--depth"); ap.add_argument("--calib"); ap.add_argument("--cam", type=int, default=2)
ap.add_argument("--lidar", help="sparse LiDAR depth .npy for the same image (meters, 0 = none)")
ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--show", action="store_true")
a = ap.parse_args()

if a.synthetic:
    import synth
    T = synth.make_trajectory(5); W = synth.make_world(T)
    lidar = synth.depth_image(W, T[0]); depth = lidar * 1.07       # a depth model that is 7% too far
    K = synth.K
else:
    from kitti_io import read_calib
    depth = np.load(a.depth); K = K_from_P(read_calib(a.calib)[f"P{a.cam}"])
    lidar = np.load(a.lidar) if a.lidar else None

pts, u, v = backproject(depth, K)

print(f"{len(pts)} points, z from {pts[:, 2].min():.1f} to {pts[:, 2].max():.1f} m")

if lidar is not None:
    m = (lidar > 0) & (depth > 0)
    ratio = np.median(depth[m] / lidar[m])
    print(f"median depth / LiDAR on {m.sum()} pixels = {ratio:.3f}  -> scale error {100 * (ratio - 1):+.1f}%")


# Create Point Cloud from Open3D
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
o3d.io.write_point_cloud("out/d1_cloud.ply", pc)
print("wrote out/d1_cloud.ply")
if a.show:
    o3d.visualization.draw_geometries([pc])
