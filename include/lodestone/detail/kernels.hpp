#pragma once

#include <cstddef>
#include <cstdint>

namespace lodestone::detail {

// dot(±1, ±1) = K - 2 * popcount(xor). Popcount is taken over the packed
// words; padding bits must be zero on both sides so they add nothing.
inline std::int32_t dot_from_pop(std::uint64_t pop, std::size_t k) {
    return static_cast<std::int32_t>(
        static_cast<std::int64_t>(k) - (static_cast<std::int64_t>(pop) << 1));
}

void binary_gemm_scalar(const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                        std::size_t rows, std::size_t cols, std::size_t k);

void binary_gemm_avx2(const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                      std::size_t rows, std::size_t cols, std::size_t k);

void binary_gemm_avx512(const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                        std::size_t rows, std::size_t cols, std::size_t k);

}  // namespace lodestone::detail
