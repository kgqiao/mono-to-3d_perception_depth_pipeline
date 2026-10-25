# synth.py: a synthetic driving world with known answers, for tests and for --synthetic runs.
# Camera frame like KITTI: x right, y down, z forward. The camera rides 1.65 m above a flat road
# and only turns left/right (yaw), so the road is always at y = +1.65 in camera coordinates.
import numpy as np

K = np.array([[718.856, 0, 607.1928], [0, 718.856, 185.2157], [0, 0, 1.0]])   # KITTI odometry seq 00, camera 0
W, H = 1241, 376
CAM_H = 1.65


def yaw_T(yaw, pos):
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])   # rotation about the y axis
    T[:3, 3] = pos
    return T


def make_trajectory(n=60, seed=0, step=(0.8, 1.3), turn_at=(20, 35), turn_deg=3.0):
    """Camera-to-world poses. World = first camera frame. Moves forward with a left turn between turn_at frames."""
    rng = np.random.default_rng(seed)
    T, yaw, pos = [], 0.0, np.zeros(3)
    for i in range(n):
        T.append(yaw_T(yaw, pos.copy()))
        d = rng.uniform(*step)
        if turn_at[0] <= i < turn_at[1]:
            yaw += np.radians(turn_deg)
        fwd = np.array([np.sin(yaw), 0, np.cos(yaw)])
        pos = pos + d * fwd
    return np.array(T)


def make_world(T, seed=0, n_ground=40000, n_wall=30000, half_width=12.0, reach=40.0):
    """Points on the road and on two building walls along the path."""
    rng = np.random.default_rng(seed)
    # road: points between the walls, from just in front of each pose to `reach` meters ahead
    gi = rng.integers(0, len(T), n_ground)
    g = []
    for k in range(n_ground):
        local = np.array([rng.uniform(-half_width, half_width), CAM_H, rng.uniform(1.0, reach)])
        g.append(T[gi[k], :3, :3] @ local + T[gi[k], :3, 3])
    g = np.array(g)
    # walls: for random path samples, offset sideways by +-half_width, heights up to 6 m above the road
    idx = rng.integers(0, len(T), n_wall)
    side = rng.choice([-1, 1], n_wall)
    fwd_off = rng.uniform(-3, reach, n_wall)
    pts = []
    for k in range(n_wall):
        Rw = T[idx[k], :3, :3]
        local = np.array([side[k] * half_width, rng.uniform(CAM_H - 6, CAM_H), fwd_off[k]])
        pts.append(Rw @ local + T[idx[k], :3, 3])
    return np.vstack([g, np.array(pts)])


def observe(world, T_w_c, noise_px=0.0, seed=0, max_depth=60.0):
    """Project world points into a camera. Returns ids of visible points, their pixels (N x 2) and depths."""
    rng = np.random.default_rng(seed)
    T_c_w = np.linalg.inv(T_w_c)
    Xc = (T_c_w[:3, :3] @ world.T).T + T_c_w[:3, 3]
    z = Xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    vis = (z > 1.0) & (z < max_depth) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    ids = np.nonzero(vis)[0]
    uv = np.c_[u[ids], v[ids]] + rng.normal(0, noise_px, (len(ids), 2)) * (noise_px > 0)
    return ids, uv, z[ids]


def pair(world, Ta, Tb, noise_px=0.5, seed=0, max_n=1500):
    """Matched pixels between two cameras (perfect data association), plus true depth in camera a."""
    ia, uva, za = observe(world, Ta, noise_px, seed)
    ib, uvb, _ = observe(world, Tb, noise_px, seed + 1)
    common, ka, kb = np.intersect1d(ia, ib, return_indices=True)
    if len(common) > max_n:
        sel = np.random.default_rng(seed).choice(len(common), max_n, replace=False)
        ka, kb = ka[sel], kb[sel]
    return uva[ka], uvb[kb], za[ka]


def depth_image(world, T_w_c):
    """Sparse depth image (0 = no point), nearest point wins. Like a LiDAR depth map."""
    ids, uv, z = observe(world, T_w_c, 0.0)
    d = np.zeros((H, W))
    ui, vi = np.round(uv[:, 0]).astype(int).clip(0, W - 1), np.round(uv[:, 1]).astype(int).clip(0, H - 1)
    order = np.argsort(-z)                         # far first, so near points overwrite
    d[vi[order], ui[order]] = z[order]
    return d
