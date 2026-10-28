// fastmatch.cpp: brute-force Hamming matching in C++ (D11), exposed to Python with pybind11.
// Build:  python setup.py build_ext --inplace     then:  python -m pytest -q -k cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <limits>
#include <utility>

namespace py = pybind11;

// Bits set in one byte. (std::popcount needs C++20; this works everywhere.)
static const struct PopTable {
    uint8_t t[256];
    PopTable() { for (int i = 0; i < 256; ++i) { int c = 0, x = i; while (x) { x &= x - 1; ++c; } t[i] = static_cast<uint8_t>(c); } }
} kPop;
static inline int popcount8(uint8_t x) { return kPop.t[x]; }

// q: N x B uint8, db: M x B uint8 (B = 32 for ORB). Returns (idx, dist) as int32 arrays of length N.
// Ties go to the lowest index, exactly like the Python reference.
std::pair<py::array_t<int32_t>, py::array_t<int32_t>>
hamming_match(py::array_t<uint8_t, py::array::c_style | py::array::forcecast> q,
              py::array_t<uint8_t, py::array::c_style | py::array::forcecast> db) {
    if (q.ndim() != 2 || db.ndim() != 2 || q.shape(1) != db.shape(1))
        throw std::invalid_argument("q and db must be 2-D with the same number of bytes per row");
    const ssize_t N = q.shape(0), M = db.shape(0), B = q.shape(1);
    if (M == 0) throw std::invalid_argument("db is empty");
    py::array_t<int32_t> idx(N), dist(N);
    const uint8_t* Q = q.data();
    const uint8_t* D = db.data();
    int32_t* I = idx.mutable_data();
    int32_t* S = dist.mutable_data();
    for (ssize_t i = 0; i < N; ++i) {
        int32_t best_d = std::numeric_limits<int32_t>::max();
        ssize_t best_j = 0;
        for (ssize_t j = 0; j < M; ++j) {
            int32_t d = 0;
            for (ssize_t k = 0; k < B; ++k)
                d += popcount8(static_cast<uint8_t>(Q[i * B + k] ^ D[j * B + k]));
            if (d < best_d) { best_d = d; best_j = j; }   // strictly smaller: ties keep the lowest j
        }
        I[i] = static_cast<int32_t>(best_j);
        S[i] = best_d;
    }
    return {idx, dist};
}

PYBIND11_MODULE(fastmatch, m) {
    m.doc() = "Brute-force Hamming matching for binary descriptors";
    m.def("hamming_match", &hamming_match, py::arg("q"), py::arg("db"));
}
