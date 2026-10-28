# sensors.py: LiDAR projection (D10), IMU speed scale (D12), and loop checks for your own footage (D9).
import numpy as np
from scipy.spatial import cKDTree


def project_velo(pts_velo, Tr, P, image_shape):
    """pts_velo: N x 3 (or N x 4) Velodyne points. Tr: 4x4 Velodyne -> camera 0. 
    P: 3x4 projection of the camera you use.
    Return (uv N x 2, z N depth along the camera axis, keep mask: in front of the 
    camera and inside the image)."""
    pts = np.asarray(pts_velo, float)[:, :3]
    h, w = image_shape[:2]
    Xc = (Tr @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]          # Velodyne -> camera 0 frame
    proj = (P @ np.c_[Xc, np.ones(len(Xc))].T).T                   # 3 x 4 projection, homogeneous
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = proj[:, :2] / proj[:, 2:3]
    keep = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, z, keep


def lidar_points_for_matches(uv_matches, uv_lidar, xyz_lidar_cam, radius_px=2.0):
    """Plumbing: for each matched pixel, the camera-frame 3D LiDAR point 
    within radius_px, if any.
    Returns (N x 3, valid mask)."""

    out = np.zeros((len(uv_matches), 3))
    ok = np.zeros(len(uv_matches), bool)

    if len(uv_lidar) == 0:
        return out, ok
    d, j = cKDTree(uv_lidar).query(uv_matches, k=1)

    ok = d <= radius_px
    out[ok] = xyz_lidar_cam[j[ok]]
    
    return out, ok


def step_lengths(v_forward, dt):
    """IMU / OXTS speed to step length in meters: |v| * dt for each frame interval."""
    return np.abs(np.asarray(v_forward, float)) * np.asarray(dt, float) # reversing gives negative v; distance is |v|


def add_imu_noise(v, dt, sigma=0.05, bias_rate=0.002, seed=0):
    """Make the speed look like a cheap phone IMU: white noise (sigma, m/s) plus a bias that grows
    slowly (bias_rate, m/s per second of driving). Same seed -> same noise."""
    v = np.asarray(v, float); t = np.cumsum(np.asarray(dt, float))
    rng = np.random.default_rng(seed)
    return v + rng.normal(0.0, sigma, size=v.shape) + bias_rate * t


def path_length(positions):
    """Total length of a path (N x 3)."""
    p = np.asarray(positions, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def closure_gap(positions):
    """How far the end of a loop is from its start (m). For a closed loop the truth is 0."""
    p = np.asarray(positions, float)
    return float(np.linalg.norm(p[-1] - p[0]))
