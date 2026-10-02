#pragma once
#include <cstdint>
#include <vector>

namespace sigscope {

// Bipolar (+1/-1) cross-correlation of `bits` against a shorter `pattern` at
// every valid starting offset -- used for sync-word search.
std::vector<int> bitstream_correlate(const std::vector<uint8_t>& bits, const std::vector<uint8_t>& pattern);

// Bipolar autocorrelation of `bits` at every lag in [1, max_lag] -- used to
// find a repeating frame period.
std::vector<double> bitstream_autocorrelation(const std::vector<uint8_t>& bits, int max_lag);

}  // namespace sigscope
