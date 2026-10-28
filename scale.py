# scale.py: giving each monocular step a length in meters (step D6). The scale ideas are yours.
import cv2
import numpy as np

from geom import ransac_plane


def ground_plane_scale(pts_unit, cam_height=1.65, rel_thresh=0.02, min_inliers=50, max_tilt_deg=15.0):
    """
    keep points below the camera and not too far; 
    fit the plane (threshold relative to typical depth); 
    reject tilted planes; scale = cam_height / fitted height

    pts_unit: N x 3 points triangulated with a unit-length step (camera frame: y down).
    The road should sit `cam_height` meters below the camera. Return meters per unit (= the step length), or None."""

    if len(pts_unit) < min_inliers:
        return None
        
    # 1. Filter: Keep points below the camera (y > 0) and within a reasonable 
    # distance (e.g., z < 40)
    # In KITTI, Y is down, so road points must have positive Y coordinates
    valid_mask = (pts_unit[:, 1] > 0) & (pts_unit[:, 2] > 0) & (pts_unit[:, 2] < 40.0)
    pts_filtered = pts_unit[valid_mask]
    
    if len(pts_filtered) < min_inliers:
        return None
        
    # 2. Establish a data-dependent RANSAC threshold
    # Threshold scales dynamically with the typical (median) depth of your filtered points
    median_depth = np.median(pts_filtered[:, 2])
    absolute_thresh = rel_thresh * median_depth
    
    # 3. Fit the ground plane using your previously written RANSAC algorithm
    result = ransac_plane(pts_filtered, thresh=absolute_thresh)

    # Safe Check: before unpacking anything
    if result is None or result[0] is None:
        return None
    
    plane, inlier_mask = result
    
    if inlier_mask is None or np.sum(inlier_mask) < min_inliers:
        return None

    if plane is None:
        return None
    
    a, b, c, d = plane
    
    # 4. Enforce structural sanity: Reject heavily tilted planes
    # The road normal vector should theoretically point straight up, which is [0, -1, 0] in KITTI
    # The dot product between our normal and perfect up is simply -b
    cos_tilt = np.abs(-b) 
    tilt_rad = np.arccos(np.clip(cos_tilt, -1.0, 1.0))
    if np.degrees(tilt_rad) > max_tilt_deg:
        return None
        
    # 5. Calculate Scale factor: scale = cam_height / fitted height
    # The fitted height is the absolute distance from the camera origin (0,0,0) to the plane, which is |d|
    fitted_height = np.abs(d)
    if fitted_height < 1e-4:
        return None
        
    scale = cam_height / fitted_height
    return float(scale)


def lift_matches(depth, uv, K, max_depth=40.0):
    """
    You write this (D6): round uv to pixels, read depth there, then the same back-projection formula as D1
    
    Metric 3D points (previous camera frame) for matched pixels, read from a depth map.
    Returns (N x 3 points, valid mask). Invalid where depth is 0, too far, or the pixel is off the image."""
    H, W = depth.shape
    num_pts = len(uv)
    
    # Pre-allocate clean output containers to guarantee consistent types
    pts3d = np.zeros((num_pts, 3), dtype=float)
    valid_mask = np.zeros(num_pts, dtype=bool)
    
    if num_pts == 0: #if there are no points, then keep the arrays as 0
        return pts3d, valid_mask
        
    # 1. Round coordinates to grid indices
    u_idx = np.round(uv[:, 0]).astype(int)
    v_idx = np.round(uv[:, 1]).astype(int)
    
    # 2. Enforce image boundary guards
    in_bounds = (u_idx >= 0) & (u_idx < W) & (v_idx >= 0) & (v_idx < H)
    
    if not np.any(in_bounds):
        return pts3d, valid_mask
        
    # 3. Read depth values safely
    z_vals = np.zeros(num_pts, dtype=float)
    z_vals[in_bounds] = depth[v_idx[in_bounds], u_idx[in_bounds]]
    
    # 4. Filter validity boundaries (0 < depth < max_depth)
    valid_mask = in_bounds & (z_vals > 0) & (z_vals < max_depth)
    
    if not np.any(valid_mask):
        return pts3d, valid_mask
        
    # 5. Extract only coordinates passing filters for vectorized back-projection
    u_vld = u_idx[valid_mask]
    v_vld = v_idx[valid_mask]
    z_vld = z_vals[valid_mask]
    
    # 6. Apply backprojection formula: Z * K_inv @ [u, v, 1]^T
    hom_pixels = np.stack([u_vld, v_vld, np.ones_like(u_vld)], axis=0)
    K_inv = np.linalg.inv(K)
    
    # Un-project to 3D rays and multiply by depth metric scale
    rays = K_inv @ hom_pixels
    pts3d[valid_mask] = (rays * z_vld).T
    
    return pts3d, valid_mask


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
    """
    walk forward; keep a list of accepted values; a value is accepted if it is near the median of the last k accepted
    
    scales: per-step estimates (None or nan = failed). Fill gaps with the last good value,
    reject jumps bigger than max_jump (relative) from the running median, then a rolling median of width k."""
    cleaned_scales = []
    accepted_history = []
    
    # 1. Walk forward through time to handle anomalies, missing data, and abrupt jumps
    for s in scales:
        # Check if current estimation element failed completely
        if s is None or np.isnan(s) or s <= 0:
            current_val = accepted_history[-1] if accepted_history else 1.0
        else:
            current_val = float(s)
            
        # 2. Outlier rejection using localized window history comparison
        if accepted_history:
            history_window = accepted_history[-k:]
            running_median = np.median(history_window)
            
            # Compute percentage variance jump from historical trendline
            relative_jump = np.abs(current_val - running_median) / running_median
            
            # If tracking registers a spike, force value down to the last safe step
            if relative_jump > max_jump:
                current_val = accepted_history[-1]
                
        accepted_history.append(current_val)
        cleaned_scales.append(current_val)
        
    # 3. Apply rolling median window smoothing pass
    smoothed_scales = []
    n = len(cleaned_scales)
    
    for i in range(n):
        # Calculate moving bounds symmetrically balanced around index position i
        start_idx = max(0, i - k // 2)
        end_idx = min(n, i + k // 2 + 1)
        smoothed_scales.append(np.median(cleaned_scales[start_idx:end_idx]))
        
    return smoothed_scales


def apply_scales(rel_poses, scales):
    """
    only the translation column changes; rotation has no scale

    Return copies of the unit-length T_prev_cur with each translation multiplied by its scale."""
    scaled_poses = []
    
    # Parallel map transitions alongside calculated metric scales
    for T_unit, s in zip(rel_poses, scales):
        # Protect memory array space integrity by isolating deep array copies
        T_scaled = np.copy(T_unit)
        
        # Scale only the linear column position vector values (Column 3)
        T_scaled[:3, 3] = T_unit[:3, 3] * s
        
        scaled_poses.append(T_scaled)
        
    return scaled_poses