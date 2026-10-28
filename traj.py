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

    # Convert inputs to numpy arrays (Shape: N x 3)
    est = np.asarray(est_pos, dtype=float)
    gt = np.asarray(gt_pos, dtype=float)
    
    # 1. Align the estimated trajectory onto the ground truth based on the requested method
    if align == "sim3":
        # Similarity alignment (Scale + Rotation + Translation)
        s, R, t = umeyama(est, gt, with_scale=True)
        # Apply transformation to every estimated point: s * (R @ x) + t
        # Transpose and broadcasting tricks let us do this efficiently for all N points
        est_aligned = s * (est @ R.T) + t
        
    elif align == "se3":
        # Rigid body alignment (Rotation + Translation only, Scale is fixed to 1.0)
        s, R, t = umeyama(est, gt, with_scale=False)
        # Apply transformation: 1.0 * (R @ x) + t
        est_aligned = (est @ R.T) + t
        
    elif align == "none":
        # No alignment: evaluate the raw trajectories exactly as they are
        est_aligned = est
        
    else:
        raise ValueError(f"Unknown alignment type: {align}")
        
    # 2. Compute the 3D position differences for each frame
    # Difference vector for each coordinate (N x 3)
    differences = est_aligned - gt
    
    # 3. Calculate the Root Mean Squared Error (RMSE)
    # Compute squared distances for each point: dx^2 + dy^2 + dz^2
    squared_errors = np.sum(differences ** 2, axis=1)
    
    # Take the mean of those squares, then take the square root
    rmse = np.sqrt(np.mean(squared_errors))
    
    return float(rmse)


def rpe(est_T, gt_T, delta=1):
    """Relative pose error over `delta` frames: mean translation error (m) and mean rotation error (deg)."""
    N = len(est_T)
    
    # We can only compute relative errors if we have enough frames left to jump by delta
    if N <= delta:
        return 0.0, 0.0
        
    t_errors = []
    r_errors = []
    
    # Loop through all possible frame pairs separated by delta
    for i in range(N - delta):
        # 1. Compute the true relative motion from frame i to frame i+delta
        # Rule: T_w_idx2 = T_w_idx1 @ T_idx1_idx2 -> T_idx1_idx2 = inv(T_w_idx1) @ T_w_idx2
        T_gt_i = gt_T[i]
        T_gt_idelta = gt_T[i + delta]
        motion_gt = np.linalg.inv(T_gt_i) @ T_gt_idelta
        
        # 2. Compute the estimated relative motion over the same window
        T_est_i = est_T[i]
        T_est_idelta = est_T[i + delta]
        motion_est = np.linalg.inv(T_est_i) @ T_est_idelta
        
        # 3. Compute the error transformation between the two motions
        # Error = inv(gt_motion) @ est_motion
        T_err = np.linalg.inv(motion_gt) @ motion_est
        
        # 4. Extract translation error magnitude |t| (in meters)
        t_err_vec = T_err[:3, 3]
        t_err_mag = np.linalg.norm(t_err_vec)
        t_errors.append(t_err_mag)
        
        # 5. Extract rotation error angle (in degrees) using the trace shortcut
        R_err = T_err[:3, :3]
        trace = np.trace(R_err)
        cos_theta = (trace - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        r_err_deg = np.degrees(np.arccos(cos_theta))
        r_errors.append(r_err_deg)
        
    # 6. Calculate the average errors across all windows
    mean_t_error = np.mean(t_errors)
    mean_r_error = np.mean(r_errors)
    
    return float(mean_t_error), float(mean_r_error)


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
