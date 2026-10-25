# solutions/failure.py: REFERENCE answers. Try your own first, then compare line by line.
# failure.py: where does the odometry break? (step D7) The bucketing is yours; the hypothesis is yours.
import numpy as np


def bucket_errors(errors, values, edges):
    """Group per-frame errors by a situation variable (speed, turn rate, inlier count).
    edges: bucket boundaries, e.g. [0, 50, 100, 200, inf]. Returns a list of dicts:
    lo, hi, n, median_error, p90_error (nan when a bucket is empty)."""
    errors = np.asarray(errors, float)
    values = np.asarray(values, float)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (values >= lo) & (values < hi)
        e = errors[m]
        out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()),
                    "median_error": float(np.median(e)) if len(e) else float("nan"),
                    "p90_error": float(np.percentile(e, 90)) if len(e) else float("nan")})
    return out


def print_buckets(name, rows):
    """Plumbing (done)."""
    print(f"\n{name}")
    print(f"{'bucket':>16} {'n':>6} {'median':>9} {'p90':>9}")
    for r in rows:
        print(f"{r['lo']:>7.3g}-{r['hi']:<8.3g} {r['n']:>6} {r['median_error']:>9.4f} {r['p90_error']:>9.4f}")
