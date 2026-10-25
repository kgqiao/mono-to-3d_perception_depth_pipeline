# fastmatch_ref.py: the Python reference for D11 (done). Your C++ version must give the same answers.
# Brute-force nearest neighbour of binary descriptors (ORB: 32 bytes = 256 bits) by Hamming distance.
import numpy as np

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)   # bits set in each byte value


def hamming_match_py(q, db):
    """q: N x 32 uint8, db: M x 32 uint8. Returns (best index per query, its Hamming distance), both int32.
    Ties go to the lowest index."""
    q = np.ascontiguousarray(q, np.uint8)
    db = np.ascontiguousarray(db, np.uint8)
    idx = np.empty(len(q), np.int32)
    dist = np.empty(len(q), np.int32)
    for i in range(len(q)):
        d = _POP[np.bitwise_xor(db, q[i])].sum(1)
        j = int(np.argmin(d))
        idx[i], dist[i] = j, d[j]
    return idx, dist
