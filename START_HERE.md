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

    ```bash
    py -3.11 -m venv mono3d_venv
    mono3d_venv\Scripts\Activate.ps1
    mono3d_venv\Scripts\activate
    pip install -r requirements.txt
    python -m pytest -q   #Expected Output R FIRST: 19 fail, 6 pass, 1 skipped (the C++ test, until you build it).
    ```
## Data (put it under data/)
- KITTI odometry: grayscale images (about 22 GB), calibration, ground-truth poses: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- 
  Layout: data/kitti/sequences/00/image_0/*.png, data/kitti/sequences/00/calib.txt, data/kitti/poses/00.txt
  Tip: start with one short sequence (04 is about 270 frames, mostly straight; 07 has turns).

- D1-D3 depth evaluation uses KITTI depth (val_selection_cropped), the same data as the Wing depth-to-3D build.
- 
- D10 Velodyne (very large) and D12 KITTI raw OXTS: only when reach those optional steps.

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

## Testing ##

1. Full test suite

`python -m pytest -q`

Everything should pass.

1. Each step end to end on synthetic data

`python d1_cloud.py --synthetic`
`python d2_height.py --synthetic`
`python d3_eval.py --synthetic`
`python d4_two_view.py --synthetic`
`python d5_vo.py --synthetic`
`python d6_scale.py --synthetic`
`python d7_failure.py --frames out/d5_04_frames.csv`
`python d8_boxes.py --synthetic`

This only runs once D1 to D7 pass their tests, since the synthetic runs use all of them. Any script that errors shows you which function is responsible.

3. Real KITTI data
Run d5_vo.py and d6_scale.py on at least two sequences (for example 04 and 00). The README needs these numbers, and synthetic results don’t count.

4. Write-up
Fill in the README results and failure-analysis sections from your KITTI runs, following the D7 outline. Remove the placeholders and the “to fill” notes.


D9: phone calibration and loop test (optional)

Print a checkerboard and take 15 to 20 photos of it at different angles, with zoom, focus and stabilization locked.
Run python d9_calibrate.py --images data/checker/*.jpg --cols 9 --rows 6 --square 0.025. 
Check that the reprojection error is under about 0.5 px.

Film a loop of known length, extract frames, then run d9_run.py and d9_loop.py.

D10: LiDAR depth (optional)

You need the KITTI Velodyne files for the sequence, which are large.
Run python d10_lidar_depth.py --root data/kitti --seq 04 --cam 0 --out out/lidar/04 --n 500.

Then run d6_scale.py with --lidar-dir out/lidar/04, so the LiDAR variant is included.

D11: C++ matcher (optional)

On Windows, install Visual Studio Build Tools with the “Desktop development with C++” workload first.

Run python setup.py build_ext --inplace, then python -m pytest -q tests/test_cpp.py.

Run python bench_match.py to see the speed comparison.

D12: IMU speed (optional)

KITTI raw OXTS data for the same drive is needed.
Run d6_scale.py with --oxts pointing at that drive, and check that the F imu clean and F imu noisy rows appear in the table.