// fastmatch.cpp: brute-force Hamming matching in C++ (optional step D11), exposed to Python with pybind11.
// Build:  python setup.py build_ext --inplace     then:  python -m pytest -q -k cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace py = pybind11;

// Bits set in one byte. (std::popcount needs C++20; this works everywhere.)
static inline int popcount8(uint8_t x) {
    int c = 0;
    while (x) { x &= static_cast<uint8_t>(x - 1); ++c; }
    return c;
}

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
    // TODO (D11), you write this: for each query row i, loop over db rows j, sum popcount8(Q[i*B+k] ^ D[j*B+k]) over k; keep the smallest (strictly smaller wins, so ties keep the lowest j)
    throw std::runtime_error("TODO D11: write hamming_match in cpp/fastmatch.cpp");
    return {idx, dist};
}

PYBIND11_MODULE(fastmatch, m) {
    m.doc() = "Brute-force Hamming matching for binary descriptors";
    m.def("hamming_match", &hamming_match, py::arg("q"), py::arg("db"));
}
