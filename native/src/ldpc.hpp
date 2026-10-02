#pragma once
#include <cstdint>
#include <vector>

namespace sigscope {

struct LdpcDecodeResult {
  std::vector<uint8_t> bits;
  bool converged;
  int iterations;
};

// Min-sum belief-propagation LDPC decoder over a sparse parity-check matrix H,
// given as parallel arrays of (row, col) 1-entries. `llrs` holds one channel
// LLR per codeword bit (positive => bit 0 more likely, project-wide
// convention). Standard or user-supplied H matrices both use this same path.
LdpcDecodeResult ldpc_decode_min_sum(const std::vector<double>& llrs, const std::vector<int>& h_rows,
                                     const std::vector<int>& h_cols, int num_checks, int num_bits,
                                     int max_iterations);

}  // namespace sigscope
