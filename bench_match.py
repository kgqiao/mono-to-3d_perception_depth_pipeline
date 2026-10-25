# bench_match.py (D11, optional): time the Python reference against your C++ module.
#   python setup.py build_ext --inplace ; python bench_match.py
import time
import numpy as np
from fastmatch_ref import hamming_match_py
import fastmatch


def t(fn, *args, reps=3):
    best = 1e9
    for _ in range(reps):
        s = time.perf_counter(); fn(*args); best = min(best, time.perf_counter() - s)
    return best


rng = np.random.default_rng(0)
print(f"{'N x M':>12} {'python s':>10} {'c++ s':>10} {'speedup':>8}")
for n in (250, 500, 1000, 2000):
    q = rng.integers(0, 256, (n, 32), dtype=np.uint8); db = rng.integers(0, 256, (n, 32), dtype=np.uint8)
    a_, b_ = t(hamming_match_py, q, db), t(fastmatch.hamming_match, q, db)
    print(f"{f'{n} x {n}':>12} {a_:10.4f} {b_:10.4f} {a_ / b_:8.2f}")
