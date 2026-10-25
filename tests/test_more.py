# D7-D12: failure buckets, boxes, LiDAR projection, IMU scale, loop checks
import numpy as np
import synth
from failure import bucket_errors
from boxes import fit_box_bev, bev_iou
from sensors import project_velo, step_lengths, add_imu_noise, path_length, closure_gap


def test_bucket_errors():
    rows = bucket_errors([1, 2, 3, 10], [5, 15, 15, 150], [0, 10, 100, np.inf])
    assert [r["n"] for r in rows] == [1, 2, 1]
    assert rows[1]["median_error"] == 2.5


def test_fit_box_recovers_rotated_car():
    rng = np.random.default_rng(0)
    L, Wd, Hh, th = 4.2, 1.8, 1.5, np.radians(30)
    local = np.c_[rng.uniform(-L / 2, L / 2, 3000), rng.uniform(-Wd / 2, Wd / 2, 3000)]
    c, s = np.cos(th), np.sin(th)
    xz = local @ np.array([[c, s], [-s, c]]) + np.array([3.0, 15.0])
    b = fit_box_bev(xz, rng.uniform(0, Hh, 3000))
    assert abs(b["length"] - L) < 0.1 and abs(b["width"] - Wd) < 0.1 and abs(b["height"] - Hh) < 0.01
    assert abs(b["heading"] - th) < np.radians(2) and abs(b["cx"] - 3) < 0.05 and abs(b["cz"] - 15) < 0.05
    assert bev_iou(b, b) > 0.999


def test_project_velo_known_point():
    # Velodyne: x forward, y left, z up. A transform that maps it to camera axes (x right, y down, z forward).
    Tr = np.eye(4); Tr[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    P = np.hstack([synth.K, np.zeros((3, 1))])
    pts = np.array([[10.0, 0.0, 0.0], [-5.0, 0.0, 0.0], [10.0, -1.0, 0.5]])
    uv, z, keep = project_velo(pts, Tr, P, (synth.H, synth.W))
    assert keep.tolist() == [True, False, True]
    assert np.allclose(uv[0], [synth.K[0, 2], synth.K[1, 2]]) and abs(z[0] - 10) < 1e-9
    assert np.allclose(uv[2], [synth.K[0, 2] + synth.K[0, 0] * 0.1, synth.K[1, 2] - synth.K[1, 1] * 0.05])


def test_imu_speed_scale():
    v, dt = np.full(10, 10.0), np.full(10, 0.1)
    assert np.allclose(step_lengths(v, dt), 1.0)
    a = add_imu_noise(v, dt, sigma=0.0, bias_rate=0.5)
    assert np.allclose(a - v, 0.5 * np.cumsum(dt))
    assert np.allclose(add_imu_noise(v, dt, seed=3), add_imu_noise(v, dt, seed=3))


def test_loop_checks():
    sq = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0, 0, 0]], float)
    assert path_length(sq) == 4.0 and closure_gap(sq) == 0.0
