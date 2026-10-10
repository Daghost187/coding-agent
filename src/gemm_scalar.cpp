#include "lodestone/detail/kernels.hpp"

#include <bit>

namespace lodestone::detail {

void binary_gemm_scalar(const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                        std::size_t rows, std::size_t cols, std::size_t k) {
    const std::size_t words = (k + 63) / 64;
    for (std::size_t row = 0; row < rows; ++row) {
        const std::uint64_t* arow = a + row * words;
        std::int32_t* crow = c + row * cols;
        for (std::size_t col = 0; col < cols; ++col) {
            const std::uint64_t* brow = b + col * words;
            std::uint64_t pop = 0;
            for (std::size_t w = 0; w < words; ++w) {
                pop += static_cast<std::uint64_t>(std::popcount(arow[w] ^ brow[w]));
            }
            crow[col] = dot_from_pop(pop, k);
        }
    }
}

}  // namespace lodestone::detail
