# boxes.py: geometric 3D boxes from a point cloud (D8).
import cv2
import numpy as np
from sklearn.cluster import DBSCAN


def cluster_objects(pts, eps=0.5, min_samples=20):
    """Plumbing (done): DBSCAN on 3D points (meters). Returns a label per point, -1 = noise."""
    if len(pts) == 0:
        return np.zeros(0, int)
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pts)


def fit_box_bev(xz, heights):
    """Oriented box from a cluster.
    xz: N x 2 ground-plane coordinates (x right, z forward), heights: N heights above the road (m).
    Return dict: cx, cz, length (long side), width, height, heading (radians in [0, pi), 
    angle of the long side from +x)."""

    xz = np.asarray(xz, float); heights = np.asarray(heights, float)

    if len(xz) < 2:
        raise ValueError("need at least 2 points to fit a box")
    
    mean = xz.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((xz - mean).T))   # ascending eigenvalues
    basis = np.c_[evecs[:, 1], evecs[:, 0]]                 # main axis (long side), then minor axis
    local = (xz - mean) @ basis
    lo, hi = local.min(axis=0), local.max(axis=0)
    center = mean + ((lo + hi) / 2) @ basis.T
    heading = np.arctan2(basis[1, 0], basis[0, 0]) % np.pi  # long side's angle from +x, in [0, pi)

    return {"cx": float(center[0]), "cz": float(center[1]),
            "length": float(hi[0] - lo[0]), "width": float(hi[1] - lo[1]),
            "height": float(heights.max() - heights.min()), "heading": float(heading)}


def bev_iou(a, b):
    """Plumbing (done): bird's-eye-view IoU of two boxes from fit_box_bev (OpenCV rotated rectangles)."""

    ra = ((a["cx"], a["cz"]), (a["length"], a["width"]), np.degrees(a["heading"]))
    rb = ((b["cx"], b["cz"]), (b["length"], b["width"]), np.degrees(b["heading"]))

    ret, pts = cv2.rotatedRectangleIntersection(ra, rb)

    if ret == cv2.INTERSECT_NONE or pts is None:
        return 0.0
    
    inter = cv2.contourArea(cv2.convexHull(pts))
    union = a["length"] * a["width"] + b["length"] * b["width"] - inter

    return float(inter / union) if union > 0 else 0.0


def car_sized(box, length=(2.5, 6.0), width=(1.2, 2.6), height=(1.0, 2.5)):
    """Plumbing (done): a plausible-car filter. Tune the ranges on 10 tuning frames, then freeze them."""
    return (length[0] <= box["length"] <= length[1] and width[0] <= box["width"] <= width[1]
            and height[0] <= box["height"] <= height[1])
