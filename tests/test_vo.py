# D4: two-view geometry on a synthetic world with known poses
import numpy as np
import synth
from vo import relative_pose, gt_relative, rotation_error_deg, tdir_error_deg, to_T

T = synth.make_trajectory(40, seed=1)
WORLD = synth.make_world(T, seed=1)


def test_gt_relative_convention():
    a = synth.yaw_T(0.3, [1.0, 0.0, 2.0])
    rel = synth.yaw_T(0.1, [0.2, 0.0, 1.0])
    assert np.allclose(gt_relative(a, a @ rel), rel)


def test_error_functions():
    R = synth.yaw_T(np.radians(5), [0, 0, 0])[:3, :3]
    assert abs(rotation_error_deg(np.eye(3), R) - 5.0) < 1e-6
    assert abs(tdir_error_deg([1, 0, 0], [0, 0, 7]) - 90.0) < 1e-6
    assert tdir_error_deg([0, 0, 1], [0, 0, 5]) < 1e-3


def test_relative_pose_matches_ground_truth_on_a_turn():
    for i in (5, 25):                                  # straight and turning
        p1, p2, _ = synth.pair(WORLD, T[i], T[i + 1], noise_px=0.5, seed=i)
        T_est, inl = relative_pose(p1, p2, synth.K)
        g = gt_relative(T[i], T[i + 1])
        assert inl.sum() > 100
        assert rotation_error_deg(T_est[:3, :3], g[:3, :3]) < 0.3
        assert tdir_error_deg(T_est[:3, 3], g[:3, 3]) < 3.0
