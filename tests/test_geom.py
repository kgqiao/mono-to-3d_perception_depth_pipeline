# D1-D3: a ray-cast flat road and a box where the right answers are known
import numpy as np
from geom import backproject, ransac_plane, height_above, metrics

K = np.array([[721.5, 0, 609.6], [0, 721.5, 172.9], [0, 0, 1.0]])
H, W, CAM_H = 352, 1216, 1.65


def flat_road(box=True):
    v, u = np.mgrid[0:H, 0:W].astype(float)
    ry, rx = (v - K[1, 2]) / K[1, 1], (u - K[0, 2]) / K[0, 0]
    d = np.zeros((H, W)); g = ry > 1e-3; d[g] = CAM_H / ry[g]
    if box:
        face = (np.abs(rx * 15) < 1) & (ry * 15 > CAM_H - 1.5) & (ry * 15 < CAM_H)
        d[face & ((d == 0) | (d > 15))] = 15.0
    d[d > 80] = 0
    return d


def test_backproject_puts_road_at_camera_height():
    pts, _, _ = backproject(flat_road(box=False), K)
    assert np.allclose(pts[:, 1], CAM_H, atol=1e-6)


def test_ransac_finds_road_and_box_height():
    pts, _, _ = backproject(flat_road(), K)
    plane, inl = ransac_plane(pts, thresh=0.05)
    assert np.allclose(plane[:3], [0, -1, 0], atol=0.01)
    assert abs(plane[3] - CAM_H) < 0.02
    assert abs(height_above(pts, plane).max() - 1.5) < 0.05


def test_metrics_oracle():
    gt = flat_road()
    assert metrics(gt, gt)["abs_rel"] == 0.0
    r = metrics(gt * 2, gt)
    assert abs(r["abs_rel"] - 1.0) < 1e-9 and r["delta1"] == 0.0
    r = metrics(gt * 1.1, gt)
    assert abs(r["abs_rel"] - 0.10) < 1e-6 and r["delta1"] == 1.0
