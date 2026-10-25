# D5: oracle tests for chaining and trajectory metrics
import numpy as np
import synth
from vo import gt_relative
from traj import chain, umeyama, ate, rpe

T = synth.make_trajectory(60, seed=2)


def test_chain_reproduces_ground_truth():
    rel = [gt_relative(T[i], T[i + 1]) for i in range(len(T) - 1)]
    assert np.allclose(chain(rel), T, atol=1e-9)


def test_umeyama_recovers_known_similarity():
    rng = np.random.default_rng(0)
    src = rng.normal(size=(100, 3))
    R = synth.yaw_T(0.7, [0, 0, 0])[:3, :3]
    dst = 2.5 * (R @ src.T).T + np.array([1.0, -2.0, 3.0])
    s, R2, t2 = umeyama(src, dst)
    assert abs(s - 2.5) < 1e-9 and np.allclose(R2, R) and np.allclose(t2, [1, -2, 3])
    s1, _, _ = umeyama(src, dst, with_scale=False)
    assert s1 == 1.0


def test_ate_zero_and_scale_alignment():
    p = T[:, :3, 3]
    assert ate(p, p) < 1e-9
    half = 0.5 * p                                  # right shape, wrong scale
    assert ate(half, p, align="sim3") < 1e-9
    assert ate(half, p, align="se3") > 1.0         # a metric claim must not hide the scale error


def test_rpe_zero_for_identical():
    r = rpe(T, T)
    assert r["trans"] < 1e-9 and r["rot_deg"] < 1e-6
