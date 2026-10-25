# traj.py: chaining poses and scoring a trajectory (step D5). The math is yours.
import numpy as np


def chain(rel_poses, T0=None):
    """rel_poses: list of 4x4 T_prev_cur. Returns N+1 camera-to-world poses starting at T0 (identity by default)."""
    raise NotImplementedError("You write this (D5): T_w_cur = T_w_prev @ T_prev_cur, starting from T0")


def umeyama(src, dst, with_scale=True):
    """Best s, R, t with dst ~= s * R @ src + t (least squares). src, dst: N x 3.
    with_scale=False gives a rigid fit (s = 1)."""
    raise NotImplementedError("You write this (D5): center both, SVD of the cross-covariance, fix a reflection with det, then s and t")


def ate(est_pos, gt_pos, align="se3"):
    """Absolute trajectory error: RMSE of positions after alignment.
    align: 'sim3' (with scale, only for a unit-scale baseline), 'se3' (rigid; use for metric variants), 'none'."""
    raise NotImplementedError("You write this (D5): align est onto gt with umeyama (or not), then RMSE of the position differences")


def rpe(est_T, gt_T, delta=1):
    """Relative pose error over `delta` frames: mean translation error (m) and mean rotation error (deg)."""
    raise NotImplementedError("You write this (D5): for each i: error = inv(gt motion i->i+delta) @ est motion i->i+delta; take |t| and the rotation angle")


def kitti_drift(est_T, gt_T, lengths=(100, 200, 300, 400, 500, 600, 700, 800), step=10):
    """Simplified KITTI odometry metric: average translation error as % of distance
    and rotation error in deg/100 m, over sub-paths of the given lengths. Use evo or the official devkit for final numbers."""
    pos = gt_T[:, :3, 3]
    dist = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    t_err, r_err = [], []
    for i in range(0, len(gt_T), step):
        for L in lengths:
            j = np.searchsorted(dist, dist[i] + L)
            if j >= len(gt_T):
                continue
            g = np.linalg.inv(gt_T[i]) @ gt_T[j]
            e = np.linalg.inv(est_T[i]) @ est_T[j]
            E = np.linalg.inv(e) @ g
            t_err.append(np.linalg.norm(E[:3, 3]) / L)
            c = (np.trace(E[:3, :3]) - 1.0) / 2.0
            r_err.append(np.degrees(np.arccos(np.clip(c, -1, 1))) / L * 100)
    if not t_err:
        return {"trans_pct": float("nan"), "rot_deg_per_100m": float("nan"), "n": 0}
    return {"trans_pct": 100 * float(np.mean(t_err)), "rot_deg_per_100m": float(np.mean(r_err)), "n": len(t_err)}
