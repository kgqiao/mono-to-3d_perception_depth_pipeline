# d7_failure.py (D7): where does the odometry break? Buckets per-frame errors by situation.
#   python d7_failure.py --frames out/d5_00_frames.csv
#   python d7_failure.py --frames out/d5_synthetic_frames.csv
import argparse
import numpy as np
import pandas as pd
from failure import bucket_errors, print_buckets

ap = argparse.ArgumentParser()
ap.add_argument("--frames", required=True)
a = ap.parse_args()
df = pd.read_csv(a.frames)
err = df["rot_err_deg"].values
print_buckets("by speed (true step length, m per frame)", bucket_errors(err, df["true_step_m"], [0, 0.3, 0.8, 1.2, np.inf]))
print_buckets("by turn rate (deg per frame)", bucket_errors(err, df["turn_deg"], [0, 0.5, 1.5, 3, np.inf]))
print_buckets("by RANSAC inliers", bucket_errors(err, df["inliers"], [0, 100, 300, 800, np.inf]))
print("\nWrite ONE hypothesis (e.g. 'fewer inliers drives the error') and ONE experiment that would test it.")
