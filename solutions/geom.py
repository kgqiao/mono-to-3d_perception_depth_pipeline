# solutions/geom.py: REFERENCE answers. Try your own first, then compare line by line.
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
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    v, u = np.nonzero(depth > 0)          # v = row, u = column
    z = depth[v, u]
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    return np.stack([x, y, z], axis=1), u, v


def ransac_plane(pts, thresh=0.10, iters=500, seed=0):
    """Fit a*x + b*y + c*z + d = 0 with |(a, b, c)| = 1 and the normal pointing toward the camera (d > 0).
    Return (np.array([a, b, c, d]), inlier mask)."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        p = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ p[0]
        inl = np.abs(pts @ n + d) < thresh
        if inl.sum() > best_n:
            best, best_n = (n, d), inl.sum()
    n, d = best
    inl = np.abs(pts @ n + d) < thresh
    q = pts[inl]
    c = q.mean(0)
    _, _, vt = np.linalg.svd(q - c, full_matrices=False)
    n = vt[-1]
    d = -n @ c
    if d < 0:                             # the camera sits above the road, so make d (its height) positive
        n, d = -n, -d
    return np.append(n, d), np.abs(pts @ n + d) < thresh


def height_above(pts, plane):
    """Signed distance of each point to the plane (+ = on the camera's side, i.e. above the road)."""
    return pts @ plane[:3] + plane[3]


def metrics(pred, gt, max_depth=80.0):
    """Only where 0 < gt < max_depth and pred > 0. Return dict abs_rel, rmse, delta1, n_pixels."""
    m = (gt > 0) & (gt < max_depth) & (pred > 0)
    p, g = pred[m], gt[m]
    ratio = np.maximum(p / g, g / p)
    return {"abs_rel": float(np.mean(np.abs(p - g) / g)),
            "rmse": float(np.sqrt(np.mean((p - g) ** 2))),
            "delta1": float(np.mean(ratio < 1.25)),
            "n_pixels": int(m.sum())}


def garg_crop(h, w):
    """The standard Eigen-split evaluation crop (Garg et al.). Returns a boolean H x W mask."""
    m = np.zeros((h, w), bool)
    m[int(0.40810811 * h):int(0.99189189 * h), int(0.03594771 * w):int(0.96405229 * w)] = True
    return m
