# Plumbing that is already done. These pass from the start; if one fails, your setup is broken, not your code.
import numpy as np
import synth
from kitti_io import save_poses, read_poses, read_calib
from vo import relative_pose
from boxes import bev_iou
from fastmatch_ref import hamming_match_py


def test_pose_file_roundtrip(tmp_path):
    T = synth.make_trajectory(5)
    save_poses(tmp_path / "p.txt", T)
    assert np.allclose(read_poses(tmp_path / "p.txt"), T, atol=1e-5)


def test_read_calib(tmp_path):
    f = tmp_path / "calib.txt"
    f.write_text("P0: " + " ".join(["1"] * 12) + "\nTr: " + " ".join(map(str, np.eye(4)[:3].ravel())) + "\n")
    c = read_calib(f)
    assert c["P0"].shape == (3, 4) and np.allclose(c["Tr"], np.eye(4))


def test_synth_projects_inside_image():
    T = synth.make_trajectory(3); W = synth.make_world(T)
    ids, uv, z = synth.observe(W, T[0])
    assert len(ids) > 1000 and uv[:, 0].max() < synth.W and uv[:, 1].max() < synth.H and z.min() > 1


def test_relative_pose_gives_unit_translation():
    T = synth.make_trajectory(5); W = synth.make_world(T)
    p1, p2, _ = synth.pair(W, T[1], T[2])
    T_est, inl = relative_pose(p1, p2, synth.K)
    assert T_est.shape == (4, 4) and abs(np.linalg.norm(T_est[:3, 3]) - 1) < 1e-6 and inl.sum() > 100


def test_bev_iou_identical_and_disjoint():
    a = {"cx": 0, "cz": 10, "length": 4, "width": 2, "height": 1.5, "heading": 0.5}
    b = dict(a, cx=50)
    assert bev_iou(a, a) > 0.999 and bev_iou(a, b) == 0.0


def test_hamming_reference():
    q = np.zeros((1, 32), np.uint8); db = np.zeros((3, 32), np.uint8); db[0, 0] = 0b111; db[2, 5] = 1
    i, d = hamming_match_py(q, db)
    assert i[0] == 1 and d[0] == 0
