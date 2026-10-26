# Start here

Monocular visual odometry on KITTI, plus scale fixes from depth, the ground plane, LiDAR and an IMU.
The lessons, tasks and checks for each step are in the Project lab, Perception tab (steps D1 to D12).

## How this starter works
- Functions YOU own are stubbed: `raise NotImplementedError("You write this (D4): ...")`. The hint says which step.
- `tests/` already knows the right answers on a synthetic driving world (`synth.py`). Implement, run the tests, repeat.
- `solutions/` has reference versions. Peek only after you've tried, then compare line by line.
- Loading, matching, OpenCV solvers, plotting and bookkeeping are complete, so your time goes to the geometry.
- Every script has `--synthetic`, so you can run each step end to end before KITTI finishes downloading.

## Setup (Windows, PowerShell, from this folder)
    py -3.11 -m venv .venv
    .venv\Scripts\Activate.ps1
    pip install -r requirements.txt
    python -m pytest -q          # at first: 19 fail, 6 pass, 1 skipped (the C++ test, until you build it). The failures are your to-do list.

## Data (put it under data/)
- KITTI odometry: grayscale images (about 22 GB), calibration, ground-truth poses: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
  Layout: data/kitti/sequences/00/image_0/*.png, data/kitti/sequences/00/calib.txt, data/kitti/poses/00.txt
  Tip: start with one short sequence (04 is about 270 frames, mostly straight; 07 has turns).
- D1-D3 depth evaluation uses KITTI depth (val_selection_cropped), the same data as the Wing depth-to-3D build.
- D10 Velodyne (very large) and D12 KITTI raw OXTS: only when you reach those optional steps.

## Order
1. D1-D3: `geom.py` (Same as Wing build)
   `python -m pytest -q tests/test_geom.py`
    then `python d1_cloud.py --synthetic`, 
    `d2_height.py --synthetic`, 
    `d3_eval.py --synthetic`

2. D4: `vo.py` owned functions 
   -> `python -m pytest -q tests/test_vo.py` 
   -> `python d4_two_view.py --synthetic`

3. D5: `traj.py` 
   -> `tests/test_traj.py` 
   -> `python d5_vo.py --synthetic`, 
   then `--root data/kitti --seq 04`

4. D6: `scale.py` 
   -> `tests/test_scale.py` 
   -> `python d6_scale.py --synthetic`, 
   then on 2 KITTI sequences

5. D7: `failure.py` 
   -> `python d7_failure.py --frames out/d5_04_frames.csv`; 
   `python corrupt.py ...` 
   and re-run D5

6. Optional: D8 `boxes.py`, D9 `sensors.py` loop checks (`d9_calibrate.py`, `d9_run.py`, `d9_loop.py`), D10 `project_velo`, D11 `cpp/fastmatch.cpp`, D12 IMU functions.

## C++ step (D11)
    # Windows: install Visual Studio Build Tools ("Desktop development with C++") first
    python setup.py build_ext --inplace
    python -m pytest -q tests/test_cpp.py
    python bench_match.py



# project_3d_perception_pipeline
Monocular to 3D Perception Pipeline

### Primary Reference: Depth Anything V2 Codebase

- https://depth-anything-v2.github.io/
- Depth Anything V2 Paper
- https://github.com/DepthAnything/Depth-Anything-V2
- Live Simulation: https://huggingface.co/spaces/depth-anything/Depth-Anything-V2

### References for Tutorials:

- https://visionbrick.com/depth-anything-v2-generating-depth-maps-from-rgb-images/
- https://www.youtube.com/watch?v=1Lt4p-cf_6Y
- https://www.youtube.com/watch?v=BoibnBIw5DA

### References for Supplementary Libraries:

- Color Maps for Matplotlib: https://matplotlib.org/stable/users/explain/colors/colormaps.html





### ABOUT CONCEPTS:

Tutorial: https://www.youtube.com/watch?v=BoibnBIw5DA
- Monocular Depth Estimation = single camera, showing how far or close something is
- Training Method: Synthetic labeled dataset, pseudo label dataset