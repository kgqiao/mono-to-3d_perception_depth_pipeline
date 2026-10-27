# traj.py: chaining poses and scoring a trajectory (step D5). The math is yours.
import numpy as np


def chain(rel_poses, T0=None):
    """
    Takes chain of individual relative steps (how the camera moved from one frame to the next)
    and accumulates them into a list of absolute positions in the world.
    If fed N relative movements --> returns a path containing N+1 physical locations.
    
    rel_poses: list of 4x4 T_prev_cur. Returns N+1 camera-to-world poses starting at T0 (identity by default)."""

    # 1. Initalize the starting post T0 (use identity matrix if None is provided)
    if T0 is None:
        T_w_cur = np.eye(4) #if none, set starting position at 0
    else:
        T_w_cur = np.copy(T0)

    # 2. Add the starting post as the first entry in trajectory list
    poses_w = [T_w_cur]

    # 3. Iteratively loop thorugh each relative transformation step
    for T_prev_cur in rel_poses: #Loop through each relative pose
        # Apply the repository chaining rule: T_w_cur = T_w_prev @ T_prev_cur
        T_w_cur = T_w_cur @ T_prev_cur

        # Append the newly calculated camera-to-world pose
        poses_w.append(T_w_cur)

    # 4. Return complete list of N+1 absolute global poses
    return poses_w


def umeyama(src, dst, with_scale=True):
    """
    Find best possible alignment between 2 matching sets of 3D points
    
    Best s, R, t with dst ~= s * R @ src + t (least squares). src, dst: N x 3.
    with_scale=False gives a rigid fit (s = 1)."""

    # 1. Ensure inputs are float arrays
    src = np.asarray(src, dtype = float)
    dst = np.asarray(dst, dtype = float)

    # 1a. Extract N and Dimension D from src source points
    N, D = src.shape # D = dimension (3)

    # 2. Compute the centers (means) of both point clouds
    src_mean = np.mean(src, axis = 0)
    dst_mean = np.mean(dst, axis = 0)

    # 3. Center both point clouds around (0, 0, 0)
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # 4. Compute the cross-covariance matrix (3x3)
    # Shape: (3, N) @ (N, 3) -> (3, 3)
    H = (dst_centered.T @ src_centered) / N

    # 5. Perform singular value decomposition (SVD) OF the covariance
    U, S, Vt = np.linalg.svd(H)

    # 6. Handle possible reflections using the determinant
    # Create an identity matrix to aassist with correcting reflection if needed
    S_matrix = np.eye(D) # Return matrix with 1s on diagonal and 0s everywhere, aka identity matrix

    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S_matrix[-1, -1] = -1
    
    # Calculate the optimal rotation matrix R
    R = U @ S_matrix @ Vt
    
    # 5. Calculate Scale (s)
    if with_scale:
        src_var = np.mean(np.sum(src_centered ** 2, axis=1))
        if src_var < 1e-8:  # Prevent division by zero
            s = 1.0
        else:
            s = np.trace(np.diag(S) @ S_matrix) / src_var
    else:
        s = 1.0
        
    # 6. Calculate Translation vector t
    # Derived from rearranging: dst_mean = s * R @ src_mean + t
    t = dst_mean - s * (R @ src_mean)
    
    return s, R, t


def ate(est_pos, gt_pos, align="se3"):
    """Absolute trajectory error: RMSE of positions after alignment.
    align: 'sim3' (with scale, only for a unit-scale baseline), 'se3' (rigid; use for metric variants), 'none'."""
    #D5): align est onto gt with umeyama (or not), then RMSE of the position differences")



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
