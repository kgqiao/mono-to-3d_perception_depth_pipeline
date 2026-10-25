# solutions/vo.py: REFERENCE answers. Try your own first, then compare line by line.
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
    orb = cv2.ORB_create(n_features)
    k1, d1 = orb.detectAndCompute(img1, None)
    k2, d2 = orb.detectAndCompute(img2, None)
    if d1 is None or d2 is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    good = [m for m, n in (p for p in bf.knnMatch(d1, d2, k=2) if len(p) == 2) if m.distance < ratio * n.distance]
    p1 = np.float64([k1[m.queryIdx].pt for m in good])
    p2 = np.float64([k2[m.trainIdx].pt for m in good])
    return p1.reshape(-1, 2), p2.reshape(-1, 2)


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


def gt_relative(T_w_prev, T_w_cur):
    """Ground-truth relative pose T_prev_cur from two camera-to-world poses."""
    return np.linalg.inv(T_w_prev) @ T_w_cur


def rotation_error_deg(R_est, R_gt):
    """Angle (degrees) of the rotation that takes R_est to R_gt."""
    c = (np.trace(R_est.T @ R_gt) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def tdir_error_deg(t_est, t_gt):
    """Angle (degrees) between two translation directions. Length is ignored on purpose (monocular)."""
    a = np.ravel(t_est) / (np.linalg.norm(t_est) + 1e-12)
    b = np.ravel(t_gt) / (np.linalg.norm(t_gt) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(a @ b, -1.0, 1.0))))
