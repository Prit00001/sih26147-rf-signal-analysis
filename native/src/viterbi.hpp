#pragma once
#include <cstdint>
#include <vector>

namespace sigscope {

// Soft-decision Viterbi decoder for a rate-1/n, constraint-length-K
// convolutional code. `llrs` holds n_symbols*n soft LLR values (positive =>
// coded bit more likely 0, matching the project-wide convention), grouped
// per symbol. `generators` holds n integers, each a K-bit generator
// polynomial (bit K-1 = current input bit, bits K-2..0 = shift-register
// memory, MSB-first). Returns one decoded bit per input symbol (non-
// terminated trellis: traceback starts from the best-metric final state).
std::vector<uint8_t> viterbi_decode(const std::vector<double>& llrs, int constraint_length,
                                     const std::vector<int>& generators);

}  // namespace sigscope
