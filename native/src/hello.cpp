// pybind11 extension module: FR-17 performance-critical kernels (Viterbi,
// LDPC min-sum BP, bitstream correlation), plus the original hello-world
// ping/add proving the C++/Python build+import path works end to end.
//
// Covers: FR-17, SR-05 (hardened build).

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

#include "correlation.hpp"
#include "ldpc.hpp"
#include "viterbi.hpp"

namespace {

std::string ping() { return "pong"; }

// Bounds-checked on purpose: overflow of int addition is undefined behaviour in
// C++, so this rejects operands that would overflow rather than wrapping
// silently -- the kind of discipline SR-05/UBSan is meant to catch early.
int add(int a, int b) {
  long long result = static_cast<long long>(a) + static_cast<long long>(b);
  if (result > INT32_MAX || result < INT32_MIN) {
    throw std::overflow_error("add: result does not fit in a 32-bit int");
  }
  return static_cast<int>(result);
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "sigscope native (C++) extension module";
  m.def("ping", &ping, "Return 'pong' -- proves the extension built and imported.");
  m.def("add", &add, "Add two ints with overflow checking.");

  m.def("viterbi_decode", &sigscope::viterbi_decode, pybind11::arg("llrs"), pybind11::arg("constraint_length"),
        pybind11::arg("generators"), "Soft-decision Viterbi decode of a rate-1/n convolutional code.");

  pybind11::class_<sigscope::LdpcDecodeResult>(m, "LdpcDecodeResult")
      .def_readonly("bits", &sigscope::LdpcDecodeResult::bits)
      .def_readonly("converged", &sigscope::LdpcDecodeResult::converged)
      .def_readonly("iterations", &sigscope::LdpcDecodeResult::iterations);

  m.def("ldpc_decode_min_sum", &sigscope::ldpc_decode_min_sum, pybind11::arg("llrs"), pybind11::arg("h_rows"),
        pybind11::arg("h_cols"), pybind11::arg("num_checks"), pybind11::arg("num_bits"),
        pybind11::arg("max_iterations") = 50, "Min-sum belief-propagation LDPC decode over a sparse H matrix.");

  m.def("bitstream_correlate", &sigscope::bitstream_correlate, pybind11::arg("bits"), pybind11::arg("pattern"),
        "Bipolar cross-correlation of a bit sequence against a pattern (sync-word search).");
  m.def("bitstream_autocorrelation", &sigscope::bitstream_autocorrelation, pybind11::arg("bits"),
        pybind11::arg("max_lag"), "Bipolar autocorrelation of a bit sequence up to max_lag (frame-period search).");
}
