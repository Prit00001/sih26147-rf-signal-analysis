#include "viterbi.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace sigscope {
namespace {

inline int popcount32(unsigned int x) {
  int c = 0;
  while (x) {
    c += static_cast<int>(x & 1u);
    x >>= 1;
  }
  return c;
}

}  // namespace

std::vector<uint8_t> viterbi_decode(const std::vector<double>& llrs, int constraint_length,
                                     const std::vector<int>& generators) {
  if (constraint_length < 2 || constraint_length > 16) {
    throw std::invalid_argument("constraint_length must be in [2, 16]");
  }
  const int K = constraint_length;
  const int n = static_cast<int>(generators.size());
  if (n < 1) {
    throw std::invalid_argument("at least one generator polynomial is required");
  }
  if (llrs.size() % static_cast<size_t>(n) != 0) {
    throw std::invalid_argument("llrs.size() must be a multiple of the number of generators");
  }
  const int num_states = 1 << (K - 1);
  const int num_symbols = static_cast<int>(llrs.size() / static_cast<size_t>(n));

  std::vector<int> next_state(static_cast<size_t>(num_states) * 2);
  std::vector<uint8_t> out_bits(static_cast<size_t>(num_states) * 2 * static_cast<size_t>(n));
  for (int s = 0; s < num_states; ++s) {
    for (int b = 0; b < 2; ++b) {
      int window = (b << (K - 1)) | s;
      int ns = (b << (K - 2)) | (s >> 1);
      next_state[static_cast<size_t>(s) * 2 + b] = ns;
      for (int k = 0; k < n; ++k) {
        out_bits[(static_cast<size_t>(s) * 2 + b) * n + k] =
            static_cast<uint8_t>(popcount32(static_cast<unsigned int>(window & generators[static_cast<size_t>(k)])) & 1);
      }
    }
  }

  const double NEG_INF = -std::numeric_limits<double>::infinity();
  std::vector<double> path_metric(num_states, NEG_INF);
  path_metric[0] = 0.0;  // encoder assumed to start in the all-zero memory state

  std::vector<uint8_t> tb_bit(static_cast<size_t>(num_symbols) * static_cast<size_t>(num_states));
  std::vector<int> tb_prev(static_cast<size_t>(num_symbols) * static_cast<size_t>(num_states));
  std::vector<double> new_metric(num_states);

  for (int t = 0; t < num_symbols; ++t) {
    std::fill(new_metric.begin(), new_metric.end(), NEG_INF);
    for (int s = 0; s < num_states; ++s) {
      if (path_metric[static_cast<size_t>(s)] == NEG_INF) continue;
      for (int b = 0; b < 2; ++b) {
        int ns = next_state[static_cast<size_t>(s) * 2 + b];
        double bm = 0.0;
        for (int k = 0; k < n; ++k) {
          uint8_t expected = out_bits[(static_cast<size_t>(s) * 2 + b) * n + k];
          double llr = llrs[static_cast<size_t>(t) * n + k];
          bm += (expected == 0) ? llr : -llr;
        }
        double cand = path_metric[static_cast<size_t>(s)] + bm;
        if (cand > new_metric[static_cast<size_t>(ns)]) {
          new_metric[static_cast<size_t>(ns)] = cand;
          tb_bit[static_cast<size_t>(t) * num_states + ns] = static_cast<uint8_t>(b);
          tb_prev[static_cast<size_t>(t) * num_states + ns] = s;
        }
      }
    }
    path_metric = new_metric;
  }

  int best_state = 0;
  double best_metric = NEG_INF;
  for (int s = 0; s < num_states; ++s) {
    if (path_metric[static_cast<size_t>(s)] > best_metric) {
      best_metric = path_metric[static_cast<size_t>(s)];
      best_state = s;
    }
  }

  std::vector<uint8_t> decoded(static_cast<size_t>(num_symbols));
  int state = best_state;
  for (int t = num_symbols - 1; t >= 0; --t) {
    uint8_t b = tb_bit[static_cast<size_t>(t) * num_states + state];
    decoded[static_cast<size_t>(t)] = b;
    state = tb_prev[static_cast<size_t>(t) * num_states + state];
  }
  return decoded;
}

}  // namespace sigscope
