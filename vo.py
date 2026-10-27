# vo.py: two-view geometry (step D4). Matching and the OpenCV solvers are plumbing (done).
# The pose conventions and the error functions are yours.
#
# Convention used everywhere in this repo (write it in your README too):
#   T_w_c  = camera-to-world pose (KITTI ground truth is this, world = first camera).
#   T_prev_cur = pose of the current camera expressed in the previous camera's frame.
#   Chaining:  T_w_cur = T_w_prev @ T_prev_cur
import cv2
import numpy as np


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def match_orb(img1, img2, n_features=3000, ratio=0.75):
    """ORB features + brute-force Hamming matching + Lowe's ratio test. Returns N x 2 float pixel arrays."""
    orb = cv2.ORB.create(nfeatures = n_features)

    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)

    # Perform check: knnMatch with k=2 requires at least 2 descriptors to work
    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return np.zeros((0, 2)), np.zeros((0, 2))

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    good = [m for m, n in (p for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < ratio * n.distance]

    # Create lists of match indices
    idx1 = [m.queryIdx for m in good]
    idx2 = [m.trainIdx for m in good]

    # Let OpenCV convert them natively at C++ speed
    p1 = cv2.KeyPoint.convert(k1, idx1) # Natively outputs NumPy array with shape (N, 2)
    p2 = cv2.KeyPoint.convert(k2, idx2)

    return p1.astype(np.float64), p2.astype(np.float64)


def relative_pose(p1, p2, K, thresh_px=1.0):
    """Essential matrix with RANSAC, then recoverPose (cheirality check picks 1 of 4 solutions).
    OpenCV returns R, t with x_cur = R x_prev + t (a world-to-camera style transform, t has length 1).
    We return T_prev_cur (this repo's convention) plus the inlier mask."""
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=thresh_px)
    if E is None or E.shape != (3, 3):
        return None, np.zeros(len(p1), bool)
    _, R, t, mpose = cv2.recoverPose(E, p1, p2, K, mask=mask)
    T_cur_prev = to_T(R, t)
    return np.linalg.inv(T_cur_prev), mpose.ravel() > 0


def triangulate(T_prev_cur, p1, p2, K):
    """3D points in the previous camera's frame from two views (same scale as T_prev_cur's translation)."""
    T_cur_prev = np.linalg.inv(T_prev_cur)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ T_cur_prev[:3, :]
    X = cv2.triangulatePoints(P1, P2, p1.T.astype(float), p2.T.astype(float))
    return (X[:3] / X[3]).T


'''D4: Work out how the camera moved between two frames (vo.py)
You compare two photos to find the rotation and the direction the 
camera moved. The distance it moved isn’t in the photos, 
so you can’t get it from two frames alone.'''

def gt_relative(T_w_prev, T_w_cur):
    """
    # Express the current pose in the previous camera's frame: one inverse and one product
    Ground-truth relative pose T_prev_cur from two camera-to-world poses."""
    # T_w_cur = T_w_prev @ T_prev_cur
    T_prev_cur = np.linalg.inv(T_w_prev) @ T_w_cur
    return T_prev_cur


def rotation_error_deg(R_est, R_gt):
    """
    Measure how far off your estimated camera rotation is from real thing
    Calculate the angle (degrees) of the rotation that takes R_est to R_gt."""

    # 1. Calculate the relative rotation matrix between estimation and ground truth
    R_rel = R_est.T & R_gt

    # 2. Extract the trace (sum of the main diagonal elements)
    trace = np.trace(R_rel)

    # 3. Extract the cosine value = cos(theta) = (trace - 1) / 2
    cos_theta = (trace - 1.0) / 2.0

    # 4. Clip to prevent floating point inaccuracies from going out of [-1.0, 1.0] bounds
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # 5. Convert from radians to degrees
    angle_deg = np.degrees(np.arccos(cos_theta))
    
    return angle_deg


def tdir_error_deg(t_est, t_gt):
    """Angle (degrees) between two translation directions. Length is ignored on purpose"""

    # 1. Flatten the inputs to ensure they are 1D arrays
    t1 = np.ravel(t_est)
    t2 = np.ravel(t_gt)

    # 2. Compute magnitudes (lengths) of both translations vectors
    norm1 = np.linalg.norm(t1)
    norm2 = np.linalg.norm(t2)

    # 3. Avoid division by zero if a vector has zero length
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    # 4. Normalize both vectors to unit length
    t1_norm = t1 / norm1
    t2_norm = t2 / norm2

    # 4. Compute dot product to find the cosine of the angle between them
    cos_alpha = np.dot(t1_norm, t2_norm)
    cos_alpha = np.clip(cos_alpha, -1.0, 1.0) #Clip the possible value

    # 5. Convert from radians to degrees
    angle_deg = np.degrees(np.arccos(cos_alpha))

    return angle_deg