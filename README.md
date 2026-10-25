# Monocular visual odometry with scale recovery (KITTI)

**Question:** can depth and the ground plane fix the scale drift that breaks monocular visual odometry, and when do they fail?

Status: in progress. No results yet. Every number here will come from my own runs.

## What is mine and what is a library
- Mine: pose conventions and chaining, trajectory metrics (ATE, RPE), the scale-recovery variants (ground plane, depth + PnP, LiDAR, IMU speed), smoothing, the evaluation and failure analysis.
- Libraries: OpenCV (ORB features, the five-point essential-matrix solver, PnP), Depth Anything V2 (pretrained depth model), Open3D, scikit-learn. The approach follows published hybrid methods (for example DF-VO); it is not a new method.

## Pose convention
T_w_c = camera-to-world (KITTI ground truth). T_prev_cur = current camera in the previous camera's frame. T_w_cur = T_w_prev @ T_prev_cur. Camera axes: x right, y down, z forward.

## Quickstart
    pip install -r requirements.txt
    python -m pytest -q
    python d6_scale.py --root data/kitti --seq 04 --depth-dir out/depth/04

## Results
(to fill from my runs: variant table with ATE (rigid alignment for metric variants) and RPE on at least 2 sequences, scale-over-time plot, failure analysis)

## Limitations
(to fill)

The plumbing (loading, plotting, command-line wiring, test fixtures) was written with AI help. The functions marked for you are mine.
