# D11: your C++ matcher must agree with the Python reference on many random inputs.
# Skipped until you build it:  python setup.py build_ext --inplace
import numpy as np
import pytest
from fastmatch_ref import hamming_match_py


def test_cpp_matches_python_reference():
    fm = pytest.importorskip("fastmatch")
    rng = np.random.default_rng(0)
    for n, m in [(1, 1), (5, 300), (200, 500)]:
        q = rng.integers(0, 256, (n, 32), dtype=np.uint8)
        db = rng.integers(0, 256, (m, 32), dtype=np.uint8)
        db[0] = q[0]                               # an exact match: distance 0
        i_py, d_py = hamming_match_py(q, db)
        i_c, d_c = fm.hamming_match(q, db)
        assert np.array_equal(i_py, i_c) and np.array_equal(d_py, d_c)
        assert d_c[0] == 0
