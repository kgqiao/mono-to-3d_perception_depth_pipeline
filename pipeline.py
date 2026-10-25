# pipeline.py: frame sources and the VO loop shared by d5_vo.py and d6_scale.py.
# It only calls your functions; it holds no math of its own except bookkeeping.
import glob
import os
import numpy as np

import synth
from vo import match_orb, relative_pose, triangulate, gt_relative
from scale import ground_plane_scale, lift_matches, pnp_rel, smooth_scale, apply_scales
from traj import chain


class SyntheticSource:
    """The synthetic world from synth.py. Gives matches, depth maps, LiDAR-like depth, speeds and ground truth."""
    def __init__(self, n=80, seed=0, noise_px=0.5):
        self.T = synth.make_trajectory(n, seed=seed)
        self.world = synth.make_world(self.T, seed=seed)
        self.K, self.noise, self.seed = synth.K, noise_px, seed
        self.dt = np.full(n - 1, 0.1)                                   # 10 Hz like KITTI
        steps = np.linalg.norm(np.diff(self.T[:, :3, 3], axis=0), axis=1)
        self.speed = steps / self.dt                                    # what an IMU / OXTS would report

    def __len__(self):
        return len(self.T)

    def matches(self, i):
        p1, p2, _ = synth.pair(self.world, self.T[i], self.T[i + 1], self.noise, seed=self.seed * 1000 + i)
        return p1, p2

    def depth(self, i):
        return synth.depth_image(self.world, self.T[i])

    lidar_depth = depth                                                 # synthetic: same sparse depth


class KittiSource:
    """KITTI odometry sequence. depth_dir: one <frame>.npy metric depth map per image (from depth_infer.py).
    lidar_dir: one <frame>.npy sparse LiDAR depth map per image (from d10_lidar_depth.py). oxts: (speed, dt) arrays."""
    def __init__(self, root, seq="00", cam=0, depth_dir=None, lidar_dir=None, oxts=None, start=0, n=None, images=None):
        import cv2
        from kitti_io import read_poses, read_calib, image_paths
        from geom import K_from_P
        self.cv2 = cv2
        sdir = os.path.join(root, "sequences", seq)
        # images: optional folder that replaces image_<cam> (e.g. a corrupted copy from corrupt.py, same file names)
        all_imgs = sorted(glob.glob(os.path.join(images, "*.png"))) if images else image_paths(sdir, cam)
        self.imgs = all_imgs[start:None if n is None else start + n]
        self.T = read_poses(os.path.join(root, "poses", f"{seq}.txt"))[start:start + len(self.imgs)]
        self.K = K_from_P(read_calib(os.path.join(sdir, "calib.txt"))[f"P{cam}"])
        self.depth_dir, self.lidar_dir = depth_dir, lidar_dir
        self.speed, self.dt = (None, None) if oxts is None else oxts

    def __len__(self):
        return len(self.imgs)

    def matches(self, i):
        a = self.cv2.imread(self.imgs[i], 0)
        b = self.cv2.imread(self.imgs[i + 1], 0)
        return match_orb(a, b)

    def _npy(self, folder, i):
        name = os.path.splitext(os.path.basename(self.imgs[i]))[0] + ".npy"
        return np.load(os.path.join(folder, name))

    def depth(self, i):
        return self._npy(self.depth_dir, i)

    def lidar_depth(self, i):
        return self._npy(self.lidar_dir, i)


class FolderSource:
    """Your own footage (D9): a folder of frames plus a phone calibration (d9_calibrate.py output).
    No ground truth: T holds identity placeholders, used only as the starting pose."""
    def __init__(self, images, calib_npz, step=1):
        import cv2
        c = np.load(calib_npz)
        self.cv2, self.K0, self.dist = cv2, c["K"], c["dist"]
        self.imgs = sorted(glob.glob(os.path.join(images, "*.png")) + glob.glob(os.path.join(images, "*.jpg")))[::step]
        h, w = cv2.imread(self.imgs[0], 0).shape
        self.K, _ = cv2.getOptimalNewCameraMatrix(self.K0, self.dist, (w, h), 0)   # undistorted camera
        self.T = np.tile(np.eye(4), (len(self.imgs), 1, 1))

    def __len__(self):
        return len(self.imgs)

    def _load(self, i):
        g = self.cv2.imread(self.imgs[i], 0)
        return self.cv2.undistort(g, self.K0, self.dist, None, self.K)

    def matches(self, i):
        return match_orb(self._load(i), self._load(i + 1))


def run_vo(src, variant="unit", imu_noise=None, smooth=True, log=None, cam_height=1.65):
    """variant: unit | oracle | plane | pnp | lidar | imu. Returns (estimated camera-to-world poses, raw scales).
    log: optional list; per-frame dicts (inliers, scale) are appended for the failure analysis (D7)."""
    from sensors import step_lengths, add_imu_noise
    rel_unit, scales, rel_metric = [], [], {}
    imu_steps = None
    if variant == "imu":
        v = src.speed if imu_noise is None else add_imu_noise(src.speed, src.dt, **imu_noise)
        imu_steps = step_lengths(v[:len(src) - 1], src.dt[:len(src) - 1])
    last = np.eye(4)
    for i in range(len(src) - 1):
        p1, p2 = src.matches(i)
        T_u, inl = relative_pose(p1, p2, src.K) if len(p1) >= 8 else (None, np.zeros(len(p1), bool))
        if T_u is None:                                                 # keep going with the last motion
            T_u = last.copy()
        last = T_u
        rel_unit.append(T_u)
        s = None
        if variant == "unit":
            s = 1.0
        elif variant == "oracle":
            s = float(np.linalg.norm(gt_relative(src.T[i], src.T[i + 1])[:3, 3]))
        elif variant == "plane" and inl.sum() >= 8:
            s = ground_plane_scale(triangulate(T_u, p1[inl], p2[inl], src.K), cam_height=cam_height)
        elif variant in ("pnp", "lidar"):
            d = src.depth(i) if variant == "pnp" else src.lidar_depth(i)
            pts, ok = lift_matches(d, p1, src.K)
            T_m, n_in = pnp_rel(pts[ok], p2[ok], src.K)
            if T_m is not None and n_in >= 15:
                rel_metric[i] = T_m
                s = float(np.linalg.norm(T_m[:3, 3]))
        elif variant == "imu":
            s = float(imu_steps[i])
        scales.append(s)
        if log is not None:
            log.append({"frame": i, "inliers": int(inl.sum()), "scale": s})
    raw = np.array([np.nan if s is None else s for s in scales])
    use = smooth_scale(scales) if (smooth and variant in ("plane", "pnp", "lidar")) else np.nan_to_num(raw, nan=1.0)
    rel = apply_scales(rel_unit, use)
    return chain(rel, T0=src.T[0]), raw
