# solutions/traj.py: REFERENCE answers. Try your own first, then compare line by line.
# traj.py: chaining poses and scoring a trajectory (step D5). The math is yours.
import numpy as np


def chain(rel_poses, T0=None):
    """rel_poses: list of 4x4 T_prev_cur. Returns N+1 camera-to-world poses starting at T0 (identity by default)."""
    T = np.eye(4) if T0 is None else np.asarray(T0, float)
    out = [T.copy()]
    for R in rel_poses:
        T = T @ R
        out.append(T.copy())
    return np.array(out)


def umeyama(src, dst, with_scale=True):
    """Best s, R, t with dst ~= s * R @ src + t (least squares). src, dst: N x 3.
    with_scale=False gives a rigid fit (s = 1)."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    n = len(src)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    S = xd.T @ xs / n
    U, D, Vt = np.linalg.svd(S)
    F = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        F[2, 2] = -1.0
    R = U @ F @ Vt
    s = float(np.trace(np.diag(D) @ F) / ((xs ** 2).sum() / n)) if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return s, R, t


def ate(est_pos, gt_pos, align="se3"):
    """Absolute trajectory error: RMSE of positions after alignment.
    align: 'sim3' (with scale, only for a unit-scale baseline), 'se3' (rigid; use for metric variants), 'none'."""
    est_pos = np.asarray(est_pos, float)
    gt_pos = np.asarray(gt_pos, float)
    if align == "none":
        aligned = est_pos
    else:
        s, R, t = umeyama(est_pos, gt_pos, with_scale=(align == "sim3"))
        aligned = (s * (R @ est_pos.T)).T + t
    return float(np.sqrt(np.mean(np.sum((aligned - gt_pos) ** 2, axis=1))))


def rpe(est_T, gt_T, delta=1):
    """Relative pose error over `delta` frames: mean translation error (m) and mean rotation error (deg)."""
    te, re = [], []
    for i in range(len(gt_T) - delta):
        g = np.linalg.inv(gt_T[i]) @ gt_T[i + delta]
        e = np.linalg.inv(est_T[i]) @ est_T[i + delta]
        E = np.linalg.inv(g) @ e
        te.append(np.linalg.norm(E[:3, 3]))
        c = (np.trace(E[:3, :3]) - 1.0) / 2.0
        re.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return {"trans": float(np.mean(te)), "rot_deg": float(np.mean(re))}


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
