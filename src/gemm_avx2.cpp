#include "lodestone/detail/kernels.hpp"

#include <bit>
#include <immintrin.h>

namespace lodestone::detail {
namespace {

__m256i avx2_popcount_xor(const __m256i a, const __m256i b) {
    const __m256i lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
    const __m256i nibble = _mm256_set1_epi8(0x0f);
    const __m256i x = _mm256_xor_si256(a, b);
    const __m256i lo = _mm256_and_si256(x, nibble);
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), nibble);
    const __m256i bytes = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));
    return _mm256_sad_epu8(bytes, _mm256_setzero_si256());
}

std::uint64_t hsum_epu64(const __m256i v) {
    alignas(32) std::uint64_t lanes[4];
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lanes), v);
    return lanes[0] + lanes[1] + lanes[2] + lanes[3];
}

void dot4(const std::uint64_t* arow, const std::uint64_t* const brows[4], std::int32_t* dst,
          std::size_t words, std::size_t words4, std::size_t k) {
    __m256i acc[4] = {
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
        _mm256_setzero_si256(),
    };
    std::size_t w = 0;
    for (; w < words4; w += 4) {
        const __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(arow + w));
        for (int r = 0; r < 4; ++r) {
            const __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(brows[r] + w));
            acc[r] = _mm256_add_epi64(acc[r], avx2_popcount_xor(va, vb));
        }
    }
    for (int r = 0; r < 4; ++r) {
        std::uint64_t pop = hsum_epu64(acc[r]);
        for (std::size_t t = w; t < words; ++t) {
            pop += static_cast<std::uint64_t>(std::popcount(arow[t] ^ brows[r][t]));
        }
        dst[r] = dot_from_pop(pop, k);
    }
}

}  // namespace

void binary_gemm_avx2(const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                      std::size_t rows, std::size_t cols, std::size_t k) {
    const std::size_t words = (k + 63) / 64;
    const std::size_t words4 = words & ~std::size_t{3};

    // Walk weight tiles in the outer loop so each tile stays in L1 while it is
    // applied to every activation row.
    std::size_t col = 0;
    for (; col + 4 <= cols; col += 4) {
        const std::uint64_t* brows[4] = {
            b + (col + 0) * words,
            b + (col + 1) * words,
            b + (col + 2) * words,
            b + (col + 3) * words,
        };
        for (std::size_t row = 0; row < rows; ++row) {
            dot4(a + row * words, brows, c + row * cols + col, words, words4, k);
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
