#include "ldpc.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace sigscope {

LdpcDecodeResult ldpc_decode_min_sum(const std::vector<double>& llrs, const std::vector<int>& h_rows,
                                     const std::vector<int>& h_cols, int num_checks, int num_bits,
                                     int max_iterations) {
  if (h_rows.size() != h_cols.size()) {
    throw std::invalid_argument("h_rows and h_cols must be the same length");
  }
  if (static_cast<int>(llrs.size()) != num_bits) {
    throw std::invalid_argument("llrs.size() must equal num_bits");
  }
  if (max_iterations < 1) {
    throw std::invalid_argument("max_iterations must be >= 1");
  }

  const size_t num_edges = h_rows.size();
  std::vector<std::vector<size_t>> check_edges(static_cast<size_t>(num_checks));
  std::vector<std::vector<size_t>> var_edges(static_cast<size_t>(num_bits));
  for (size_t e = 0; e < num_edges; ++e) {
    int r = h_rows[e];
    int c = h_cols[e];
    if (r < 0 || r >= num_checks || c < 0 || c >= num_bits) {
      throw std::invalid_argument("h_rows/h_cols index out of range for the given num_checks/num_bits");
    }
    check_edges[static_cast<size_t>(r)].push_back(e);
    var_edges[static_cast<size_t>(c)].push_back(e);
  }

  std::vector<double> var_to_check(num_edges, 0.0);
  std::vector<double> check_to_var(num_edges, 0.0);
  for (size_t v = 0; v < static_cast<size_t>(num_bits); ++v) {
    for (size_t e : var_edges[v]) var_to_check[e] = llrs[v];
  }

  std::vector<uint8_t> hard_bits(static_cast<size_t>(num_bits), 0);
  bool converged = false;
  int iterations_used = 0;

  for (int iter = 0; iter < max_iterations; ++iter) {
    iterations_used = iter + 1;

    // Check-node update (min-sum), excluding each edge's own incoming message.
    for (size_t c = 0; c < static_cast<size_t>(num_checks); ++c) {
      const auto& edges = check_edges[c];
      for (size_t ei : edges) {
        double prod_sign = 1.0;
        double min_abs = std::numeric_limits<double>::infinity();
        for (size_t ej : edges) {
          if (ej == ei) continue;
          double m = var_to_check[ej];
          prod_sign *= (m < 0.0) ? -1.0 : 1.0;
          double a = std::fabs(m);
          if (a < min_abs) min_abs = a;
        }
        check_to_var[ei] = prod_sign * min_abs;
      }
    }

    // Variable-node update: total (a-posteriori) LLR, hard decision, and the
    // extrinsic message sent back out to each connected check.
    for (size_t v = 0; v < static_cast<size_t>(num_bits); ++v) {
      double total = llrs[v];
      // cppcheck-suppress useStlAlgorithm
      // A raw loop reads more clearly here than std::accumulate with a
      // capturing lambda for a one-line running sum; not a correctness
      // concern, cppcheck's style suggestion overruled deliberately.
      for (size_t e : var_edges[v]) total += check_to_var[e];
      hard_bits[v] = (total < 0.0) ? uint8_t{1} : uint8_t{0};
      for (size_t e : var_edges[v]) {
        var_to_check[e] = total - check_to_var[e];
      }
    }

    // Convergence check: every parity-check equation satisfied.
    bool ok = true;
    for (size_t c = 0; c < static_cast<size_t>(num_checks) && ok; ++c) {
      int parity = 0;
      // cppcheck-suppress useStlAlgorithm
      // XOR-reduction, not a sum -- std::accumulate would need the same
      // capturing-lambda indirection with none of the clarity benefit.
      for (size_t e : check_edges[c]) parity ^= hard_bits[static_cast<size_t>(h_cols[e])];
      if (parity != 0) ok = false;
    }
    if (ok) {
      converged = true;
      break;
    }
  }

  return LdpcDecodeResult{hard_bits, converged, iterations_used};
}

}  // namespace sigscope
