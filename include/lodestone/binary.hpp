#pragma once

// 1-bit gemm.
//
// A weight or activation of +1 is stored as bit 1. A value of -1 is stored as
// bit 0. Within each uint64_t, feature i lives at bit (i % 64) of word i/64.
// Padding bits past K are zero.
//
// For vectors a, b in {+1, -1}^K:
//
//   dot(a, b) = K - 2 * popcount(a XOR b)
//             = 2 * popcount(a XNOR b) - K
//
// binary_gemm computes that dot for every pair of rows. It does not scale.

#include "lodestone/check.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace lodestone {

inline constexpr std::size_t kWordBits = 64;

inline constexpr std::size_t packed_words(std::size_t k) {
    return (k + kWordBits - 1) / kWordBits;
}

enum class Isa : std::uint8_t { Scalar, Avx2, Avx512 };

[[nodiscard]] const char* isa_name(Isa isa);
[[nodiscard]] bool isa_supported(Isa isa);
[[nodiscard]] Isa active_isa();

// Pack signs. `words` must hold packed_words(values.size()) entries.
// Positive and zero become +1 (bit 1). Negative becomes -1 (bit 0).
inline void pack_signs(std::span<const float> values, std::span<std::uint64_t> words) {
    static_assert(kWordBits == 64, "bit addressing assumes 64-bit words");
    require(words.size() == packed_words(values.size()), "pack_signs: word count does not match K");
    for (std::uint64_t& word : words) {
        word = 0;
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (values[i] >= 0.f) {
            words[i / kWordBits] |= std::uint64_t{1} << (i % kWordBits);
        }
    }
}

[[nodiscard]] inline float unpack_pole(std::span<const std::uint64_t> words, std::size_t index) {
    require((index / kWordBits) < words.size(), "unpack_pole: index past packed buffer");
    const std::uint64_t bit = (words[index / kWordBits] >> (index % kWordBits)) & std::uint64_t{1};
    return bit != 0 ? 1.f : -1.f;
}

// C[m, n] = dot(A[m, :], B[n, :]) with A shaped [rows, K] and B shaped [cols, K],
// both packed row-major. K may be any length; it does not have to be a multiple of 64.
void binary_gemm(const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                 std::size_t rows, std::size_t cols, std::size_t k);

void binary_gemm(Isa isa, const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                 std::size_t rows, std::size_t cols, std::size_t k);

inline void binary_gemm(std::span<const std::uint64_t> a, std::span<const std::uint64_t> b,
                        std::span<std::int32_t> c, std::size_t rows, std::size_t cols, std::size_t k) {
    const std::size_t words = packed_words(k);
    require(a.size() == rows * words, "binary_gemm: packed A has the wrong size");
    require(b.size() == cols * words, "binary_gemm: packed B has the wrong size");
    require(c.size() == rows * cols, "binary_gemm: output has the wrong size");
    binary_gemm(a.data(), b.data(), c.data(), rows, cols, k);
}

}  // namespace lodestone
