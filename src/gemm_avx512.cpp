#include "lodestone/detail/kernels.hpp"

#include <bit>
#include <immintrin.h>

namespace lodestone::detail {
namespace {

void dot8(const std::uint64_t* arow, const std::uint64_t* const brows[8], std::int32_t* dst,
          std::size_t words, std::size_t words8, std::size_t k) {
    __m512i acc[8];
    for (int r = 0; r < 8; ++r) {
        acc[r] = _mm512_setzero_si512();
    }

    std::size_t w = 0;
    for (; w < words8; w += 8) {
        const __m512i va = _mm512_loadu_si512(arow + w);
        for (int r = 0; r < 8; ++r) {
            const __m512i vb = _mm512_loadu_si512(brows[r] + w);
            acc[r] = _mm512_add_epi64(acc[r], _mm512_popcnt_epi64(_mm512_xor_si512(va, vb)));
        }
    }

    for (int r = 0; r < 8; ++r) {
        std::uint64_t pop = static_cast<std::uint64_t>(_mm512_reduce_add_epi64(acc[r]));
        for (std::size_t t = w; t < words; ++t) {
            pop += static_cast<std::uint64_t>(std::popcount(arow[t] ^ brows[r][t]));
        }
        dst[r] = dot_from_pop(pop, k);
    }
}

}  // namespace

void binary_gemm_avx512(const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                        std::size_t rows, std::size_t cols, std::size_t k) {
    const std::size_t words = (k + 63) / 64;

    // K <= 64 is one word per row, and those words are contiguous across columns.
    // Broadcast the activation and popcount eight weight rows at once.
    if (words == 1) {
        for (std::size_t row = 0; row < rows; ++row) {
            const __m512i va = _mm512_set1_epi64(static_cast<long long>(a[row]));
            std::int32_t* crow = c + row * cols;
            std::size_t col = 0;
            for (; col + 8 <= cols; col += 8) {
                const __m512i pc = _mm512_popcnt_epi64(_mm512_xor_si512(va, _mm512_loadu_si512(b + col)));
                alignas(64) std::uint64_t pops[8];
                _mm512_store_si512(pops, pc);
                for (int r = 0; r < 8; ++r) {
                    crow[col + static_cast<std::size_t>(r)] = dot_from_pop(pops[r], k);
                }
            }
            for (; col < cols; ++col) {
                const std::uint64_t pop = static_cast<std::uint64_t>(std::popcount(a[row] ^ b[col]));
                crow[col] = dot_from_pop(pop, k);
            }
        }
        return;
    }

    const std::size_t words8 = words & ~std::size_t{7};

    std::size_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        const std::uint64_t* brows[8];
        for (int r = 0; r < 8; ++r) {
            brows[r] = b + (col + static_cast<std::size_t>(r)) * words;
        }
        for (std::size_t row = 0; row < rows; ++row) {
            dot8(a + row * words, brows, c + row * cols + col, words, words8, k);
        }
    }
    for (; col < cols; ++col) {
        const std::uint64_t* brow = b + col * words;
        for (std::size_t row = 0; row < rows; ++row) {
            const std::uint64_t* arow = a + row * words;
            std::uint64_t pop = 0;
            for (std::size_t w = 0; w < words; ++w) {
                pop += static_cast<std::uint64_t>(std::popcount(arow[w] ^ brow[w]));
            }
            c[row * cols + col] = dot_from_pop(pop, k);
        }
    }
}

}  // namespace lodestone::detail
