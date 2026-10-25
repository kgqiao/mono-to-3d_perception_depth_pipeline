# geom.py: depth -> 3D points, the ground plane, and depth metrics (steps D1 to D3).
# Same functions as depthkit.py in the Wing depth-to-3D starter. If you already wrote them there, paste yours in.
import numpy as np


def load_depth_png(path):
    """KITTI depth PNG: uint16, meters = value / 256, 0 = no LiDAR return."""
    from PIL import Image
    return np.asarray(Image.open(path), dtype=np.float32) / 256.0


def K_from_P(P):
    """KITTI odometry calib gives 3x4 projection matrices. The left 3x3 block is K."""
    return np.asarray(P, dtype=float)[:3, :3].copy()


def backproject(depth, K):
    """depth: H x W meters (0 = skip). Return (N x 3 points in the camera frame: x right, y down, z forward, u, v)."""
    raise NotImplementedError("You write this (D1): the pinhole model run backwards: X = (u - cx) Z / fx, Y = (v - cy) Z / fy")


def ransac_plane(pts, thresh=0.10, iters=500, seed=0):
    """Fit a*x + b*y + c*z + d = 0 with |(a, b, c)| = 1 and the normal pointing toward the camera (d > 0).
    Return (np.array([a, b, c, d]), inlier mask)."""
    raise NotImplementedError("You write this (D2): sample 3 points, plane through them, count points within thresh, keep the best, refit with SVD, flip so d > 0")


def height_above(pts, plane):
    """Signed distance of each point to the plane (+ = on the camera's side, i.e. above the road)."""
    raise NotImplementedError("You write this (D2): a dot product with the normal, plus d")


def metrics(pred, gt, max_depth=80.0):
    """Only where 0 < gt < max_depth and pred > 0. Return dict abs_rel, rmse, delta1, n_pixels."""
    raise NotImplementedError("You write this (D3): mask first, then AbsRel = mean |p - g| / g, RMSE, and the share with max(p/g, g/p) < 1.25")


def garg_crop(h, w):
    """The standard Eigen-split evaluation crop (Garg et al.). Returns a boolean H x W mask."""
    m = np.zeros((h, w), bool)
    m[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
    return m
