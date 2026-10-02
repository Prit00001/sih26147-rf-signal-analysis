// GoogleTest unit tests for the C++ kernels, independent of the Python
// bindings (SR-05/Phase 4 requirement for C++-level unit tests).
//
// Covers: FR-12 (Viterbi, LDPC), FR-13 support (correlation).

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "correlation.hpp"
#include "ldpc.hpp"
#include "viterbi.hpp"

namespace {

std::vector<uint8_t> ConvEncode(const std::vector<uint8_t>& bits, int K, const std::vector<int>& generators) {
  int mem = 0;
  std::vector<uint8_t> out;
  for (uint8_t b : bits) {
    int window = (static_cast<int>(b) << (K - 1)) | mem;
    for (int g : generators) {
      int parity = 0;
      for (int bitpos = 0; bitpos < 32; ++bitpos) {
        if ((window & g) & (1 << bitpos)) parity ^= 1;
      }
      out.push_back(static_cast<uint8_t>(parity));
    }
    mem = (static_cast<int>(b) << (K - 2)) | (mem >> 1);
  }
  return out;
}

}  // namespace

TEST(Viterbi, NoiselessRoundTrip) {
  const int K = 7;
  const std::vector<int> generators = {0171, 0133};
  std::vector<uint8_t> bits = {1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1};
  auto coded = ConvEncode(bits, K, generators);
  std::vector<double> llrs;
  llrs.reserve(coded.size());
  for (uint8_t c : coded) llrs.push_back(c == 0 ? 5.0 : -5.0);
  auto decoded = sigscope::viterbi_decode(llrs, K, generators);
  ASSERT_EQ(decoded.size(), bits.size());
  for (size_t i = 0; i < bits.size(); ++i) {
    EXPECT_EQ(decoded[i], bits[i]) << "at index " << i;
  }
}

TEST(Viterbi, RejectsBadConstraintLength) {
  EXPECT_THROW(sigscope::viterbi_decode({1.0, 2.0}, 1, {0171, 0133}), std::invalid_argument);
  EXPECT_THROW(sigscope::viterbi_decode({1.0, 2.0}, 7, {}), std::invalid_argument);
}

TEST(Ldpc, DecodesCleanCodeword) {
  // Tiny H: 4 checks over 8 bits, each check covering 2 bits (simple parity pairs).
  std::vector<int> h_rows = {0, 0, 1, 1, 2, 2, 3, 3};
  std::vector<int> h_cols = {0, 1, 2, 3, 4, 5, 6, 7};
  // All-zero is trivially a valid codeword for this H (even parity per pair).
  std::vector<double> llrs(8, 5.0);  // strongly favors bit=0 everywhere
  auto result = sigscope::ldpc_decode_min_sum(llrs, h_rows, h_cols, 4, 8, 20);
  EXPECT_TRUE(result.converged);
  for (uint8_t b : result.bits) EXPECT_EQ(b, 0);
}

TEST(Ldpc, RejectsMismatchedLlrSize) {
  std::vector<int> h_rows = {0};
  std::vector<int> h_cols = {0};
  EXPECT_THROW(sigscope::ldpc_decode_min_sum({1.0, 2.0}, h_rows, h_cols, 1, 1, 10), std::invalid_argument);
}

TEST(Correlation, FindsExactPatternMatch) {
  std::vector<uint8_t> pattern = {1, 0, 1, 1};
  std::vector<uint8_t> bits = {0, 0, 1, 0, 1, 1, 0, 0};
  auto scores = sigscope::bitstream_correlate(bits, pattern);
  int best = 0;
  for (size_t i = 1; i < scores.size(); ++i) {
    if (scores[i] > scores[static_cast<size_t>(best)]) best = static_cast<int>(i);
  }
  EXPECT_EQ(best, 2);
  EXPECT_EQ(scores[static_cast<size_t>(best)], static_cast<int>(pattern.size()));
}

TEST(Correlation, AutocorrelationFindsPeriod) {
  std::vector<uint8_t> bits;
  for (int i = 0; i < 20; ++i) {
    bits.push_back(0);
    bits.push_back(1);
    bits.push_back(1);
    bits.push_back(0);
  }
  auto autoc = sigscope::bitstream_autocorrelation(bits, 10);
  int best_lag = 1;
  for (int lag = 2; lag <= 10; ++lag) {
    if (autoc[static_cast<size_t>(lag - 1)] > autoc[static_cast<size_t>(best_lag - 1)]) best_lag = lag;
  }
  EXPECT_EQ(best_lag, 4);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
