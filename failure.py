# failure.py: where does the odometry break? Bucket per frame errors by situation: speed, turning rate, and # of inlier matches
import numpy as np


def bucket_errors(errors, values, edges):
    """
    For each [lo, hi) pick the frames whose value falls inside, then the 
    median and 90th percentile of their errors
    
    Group per-frame errors by a situation variable (speed, turn rate, inlier count).
    edges: bucket boundaries, e.g. [0, 50, 100, 200, inf]. Returns a list of dicts:
    lo, hi, n, median_error, p90_error (nan when a bucket is empty).
    """
    # Ensure inputs are standard numpy arrays for fast boolean indexing
    errs = np.asarray(errors, dtype=float)
    vals = np.asarray(values, dtype=float)
    
    bucketed_errors = []

    # Loop through the edges to create pairs of [lo, hi)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        
        # 1. Create a mask picking frames whose situation value falls inside [lo, hi)
        mask = (vals >= lo) & (vals < hi)
        
        # Extract the errors specific to this bucket's situation
        bucket_errs = errs[mask]
        n_frames = int(len(bucket_errs))
        
        # 2. Initialize default statistical properties for empty buckets
        median_error = np.nan
        p90_error = np.nan
        
        # 3. Calculate statistics only if the bucket actually caught frames
        if n_frames > 0:
            median_error = float(np.median(bucket_errs))
            p90_error = float(np.percentile(bucket_errs, 90))
            
        # 4. Package structural property parameters
        bucket = {
            "lo": lo, 
            "hi": hi, 
            "n": n_frames, 
            "median_error": median_error, 
            "p90_error": p90_error
        }
        
        bucketed_errors.append(bucket)
        
    return bucketed_errors



def print_buckets(name, rows):
    """Print bucketed errors"""
    print(f"\n{name}")
    print(f"{'bucket':>16} {'n':>6} {'median':>9} {'p90':>9}")
    for r in rows:
        print(f"{r['lo']:>7.3g}-{r['hi']:<8.3g} {r['n']:>6} {r['median_error']:>9.4f} {r['p90_error']:>9.4f}")
