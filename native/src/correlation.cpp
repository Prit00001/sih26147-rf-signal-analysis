#include "correlation.hpp"

#include <stdexcept>

namespace sigscope {
namespace {
inline int bipolar(uint8_t b) { return b ? -1 : 1; }
}  // namespace

std::vector<int> bitstream_correlate(const std::vector<uint8_t>& bits, const std::vector<uint8_t>& pattern) {
  if (pattern.empty()) throw std::invalid_argument("pattern must not be empty");
  if (bits.size() < pattern.size()) return {};
  const size_t num_offsets = bits.size() - pattern.size() + 1;
  std::vector<int> scores(num_offsets, 0);
  for (size_t offset = 0; offset < num_offsets; ++offset) {
    int score = 0;
    for (size_t k = 0; k < pattern.size(); ++k) {
      score += bipolar(bits[offset + k]) * bipolar(pattern[k]);
    }
    scores[offset] = score;
  }
  return scores;
}

std::vector<double> bitstream_autocorrelation(const std::vector<uint8_t>& bits, int max_lag) {
  if (max_lag < 1) throw std::invalid_argument("max_lag must be >= 1");
  std::vector<double> result(static_cast<size_t>(max_lag), 0.0);
  const size_t n = bits.size();
  for (int lag = 1; lag <= max_lag; ++lag) {
    if (static_cast<size_t>(lag) >= n) break;
    long long sum = 0;
    size_t count = n - static_cast<size_t>(lag);
    for (size_t i = 0; i < count; ++i) {
      sum += bipolar(bits[i]) * bipolar(bits[i + static_cast<size_t>(lag)]);
    }
    result[static_cast<size_t>(lag - 1)] = count > 0 ? static_cast<double>(sum) / static_cast<double>(count) : 0.0;
  }
  return result;
}

}  // namespace sigscope
