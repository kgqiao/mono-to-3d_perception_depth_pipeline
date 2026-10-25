# solutions/scale.py: REFERENCE answers. Try your own first, then compare line by line.
# scale.py: giving each monocular step a length in meters (step D6). The scale ideas are yours.
import cv2
import numpy as np

from geom import ransac_plane


def ground_plane_scale(pts_unit, cam_height=1.65, rel_thresh=0.02, min_inliers=50, max_tilt_deg=15.0):
    """pts_unit: N x 3 points triangulated with a unit-length step (camera frame: y down).
    The road should sit `cam_height` meters below the camera. Return meters per unit (= the step length), or None."""
    p = pts_unit[(pts_unit[:, 1] > 0) & (pts_unit[:, 2] > 0)]
    if len(p) < min_inliers:
        return None
    zmed = np.median(p[:, 2])
    p = p[p[:, 2] < zmed]                           # far points triangulate badly with a 1 m baseline
    if len(p) < min_inliers:
        return None
    plane, inl = ransac_plane(p, thresh=rel_thresh * np.median(p[:, 2]), iters=300)
    if inl.sum() < min_inliers:
        return None
    up = np.array([0.0, -1.0, 0.0])                 # y points down, so "up" is -y
    if np.degrees(np.arccos(np.clip(plane[:3] @ up, -1, 1))) > max_tilt_deg:
        return None                                  # locked onto a wall or a curb
    return float(cam_height / plane[3])


def lift_matches(depth, uv, K, max_depth=40.0):
    """Metric 3D points (previous camera frame) for matched pixels, read from a depth map.
    Returns (N x 3 points, valid mask). Invalid where depth is 0, too far, or the pixel is off the image."""
    h, w = depth.shape
    ui = np.round(uv[:, 0]).astype(int)
    vi = np.round(uv[:, 1]).astype(int)
    ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    z = np.zeros(len(uv))
    z[ok] = depth[vi[ok], ui[ok]]
    ok &= (z > 0) & (z < max_depth)
    x = (uv[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1), ok


def pnp_rel(pts3d_prev, uv_cur, K, reproj_px=2.0):
    """Plumbing (done): PnP with RANSAC. Metric 3D points + 2D matches -> metric T_prev_cur, inlier count."""
    if len(pts3d_prev) < 6:
        return None, 0
    ok, rvec, tvec, inl = cv2.solvePnPRansac(pts3d_prev.astype(np.float64), uv_cur.astype(np.float64), K, None,
                                             reprojectionError=reproj_px, iterationsCount=200,
                                             flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None, 0
    R, _ = cv2.Rodrigues(rvec)
    T_cur_prev = np.eye(4)
    T_cur_prev[:3, :3] = R
    T_cur_prev[:3, 3] = tvec.ravel()
    return np.linalg.inv(T_cur_prev), 0 if inl is None else len(inl)


def smooth_scale(scales, k=5, max_jump=0.3):
    """scales: per-step estimates (None or nan = failed). Fill gaps with the last good value,
    reject jumps bigger than max_jump (relative) from the running median, then a rolling median of width k."""
    out, hist = [], []
    for s in scales:
        good = s is not None and np.isfinite(s) and s > 0
        if good and hist:
            med = np.median(hist[-k:])
            good = abs(s - med) / med <= max_jump
        if good:
            hist.append(float(s))
        val = np.median(hist[-k:]) if hist else np.nan
        out.append(float(val))
    out = np.array(out)
    if np.isnan(out).any() and hist:                 # gaps before the first good value
        out[np.isnan(out)] = hist[0]
    return out


def apply_scales(rel_poses, scales):
    """Return copies of the unit-length T_prev_cur with each translation multiplied by its scale."""
    out = []
    for T, s in zip(rel_poses, scales):
        T2 = np.array(T, float, copy=True)
        T2[:3, 3] = T2[:3, 3] / (np.linalg.norm(T2[:3, 3]) + 1e-12) * s
        out.append(T2)
    return out
