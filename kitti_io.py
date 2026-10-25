# kitti_io.py: read KITTI odometry and raw files. Plumbing, complete.
# Odometry layout:  <root>/sequences/00/image_0/000000.png, calib.txt, times.txt
#                   <root>/poses/00.txt   (ground truth for sequences 00-10)
#                   <root>/sequences/00/velodyne/000000.bin   (only if you download the Velodyne set, D10)
import glob
import os
import numpy as np


def read_poses(path):
    """Each line: 12 numbers, a 3x4 [R|t] = camera-to-world pose of camera 0 (world = first camera).
    Returns N x 4 x 4."""
    a = np.loadtxt(path).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(a), 1, 1))
    T[:, :3, :] = a
    return T


def save_poses(path, T):
    """Write N x 4 x 4 poses in the same 12-numbers-per-line format (evo and the KITTI devkit read this)."""
    np.savetxt(path, np.asarray(T)[:, :3, :].reshape(len(T), 12), fmt="%.6e")


def read_calib(path):
    """calib.txt lines like 'P0: 12 numbers' and 'Tr: 12 numbers'. Returns dict of 3x4 P* and 4x4 Tr (Velodyne -> camera 0)."""
    out = {}
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            vals = np.array([float(x) for x in v.split()])
            if vals.size != 12:
                continue
            m = vals.reshape(3, 4)
            if k.strip() == "Tr":
                T = np.eye(4)
                T[:3] = m
                out["Tr"] = T
            else:
                out[k.strip()] = m
    return out


def read_times(path):
    return np.loadtxt(path)


def image_paths(seq_dir, cam=0):
    return sorted(glob.glob(os.path.join(seq_dir, f"image_{cam}", "*.png")))


def read_velo(path):
    """Velodyne scan: float32 x, y, z, reflectance."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_oxts(drive_dir):
    """KITTI raw drive: oxts/data/*.txt (30 numbers each) and oxts/timestamps.txt.
    Column 8 is vf, forward velocity in m/s. Returns (vf array, times in seconds)."""
    from datetime import datetime
    files = sorted(glob.glob(os.path.join(drive_dir, "oxts", "data", "*.txt")))
    vf = np.array([np.loadtxt(f)[8] for f in files])
    with open(os.path.join(drive_dir, "oxts", "timestamps.txt")) as f:
        ts = [datetime.strptime(s.strip()[:26], "%Y-%m-%d %H:%M:%S.%f") for s in f if s.strip()]
    t = np.array([(x - ts[0]).total_seconds() for x in ts])
    return vf, t
