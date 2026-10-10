#pragma once

// Lodestone: a 1-bit transformer foundation.
// Weights are poles, +1 or -1. The matmul is popcount of an xor.

#include "lodestone/binary.hpp"
#include "lodestone/nn.hpp"

namespace lodestone {

inline constexpr int kVersionMajor = 0;
inline constexpr int kVersionMinor = 1;
inline constexpr int kVersionPatch = 0;

}  // namespace lodestone
