# D6: each scale source must recover the true step length on the synthetic world
import numpy as np
import synth
from vo import gt_relative, relative_pose, triangulate
from scale import ground_plane_scale, lift_matches, pnp_rel, smooth_scale, apply_scales

T = synth.make_trajectory(40, seed=3)
WORLD = synth.make_world(T, seed=3)


def test_ground_plane_scale_recovers_step_length():
    # one frame pair is noisy (a 1 m baseline triangulates the road roughly), so judge many pairs
    errs, fails = [], 0
    for i in range(2, 36, 2):
        p1, p2, _ = synth.pair(WORLD, T[i], T[i + 1], noise_px=0.3, seed=i)
        T_unit, inl = relative_pose(p1, p2, synth.K)
        s = ground_plane_scale(triangulate(T_unit, p1[inl], p2[inl], synth.K))
        true = np.linalg.norm(gt_relative(T[i], T[i + 1])[:3, 3])
        if s is None:
            fails += 1
        else:
            errs.append(abs(s - true) / true)
    assert fails <= 3
    assert np.median(errs) < 0.06                        # typical pair: within 6%
    assert np.mean(np.array(errs) < 0.15) >= 0.8         # a few bad pairs remain: that is why you smooth


def test_pnp_with_true_depth_is_metric():
    i = 10
    p1, p2, _ = synth.pair(WORLD, T[i], T[i + 1], noise_px=0.3, seed=7)
    pts, ok = lift_matches(synth.depth_image(WORLD, T[i]), p1, synth.K)
    assert ok.sum() > 50
    T_m, n_inl = pnp_rel(pts[ok], p2[ok], synth.K)
    true = np.linalg.norm(gt_relative(T[i], T[i + 1])[:3, 3])
    assert abs(np.linalg.norm(T_m[:3, 3]) - true) / true < 0.05


def test_smooth_scale_rejects_spikes_and_gaps():
    s = smooth_scale([1.0, 1.02, None, 5.0, 0.98, 1.01, float("nan"), 1.0])
    assert np.all(np.abs(s - 1.0) < 0.05)


def test_apply_scales_changes_only_translation():
    rel = [gt_relative(T[i], T[i + 1]) for i in range(5)]
    unit = [r.copy() for r in rel]
    for u in unit:
        u[:3, 3] /= np.linalg.norm(u[:3, 3])
    out = apply_scales(unit, [np.linalg.norm(r[:3, 3]) for r in rel])
    assert all(np.allclose(a, b) for a, b in zip(out, rel))
