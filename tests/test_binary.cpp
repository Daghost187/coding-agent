#include "harness.hpp"
#include "lodestone/binary.hpp"

#include <cstdint>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

int sign_of(float value) {
    return value >= 0.f ? 1 : -1;
}

int naive_dot(const std::vector<float>& a, const std::vector<float>& b) {
    int dot = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        dot += sign_of(a[i]) * sign_of(b[i]);
    }
    return dot;
}

void check_shape(int rows, int cols, int k, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    std::vector<float> a(static_cast<std::size_t>(rows) * k);
    std::vector<float> b(static_cast<std::size_t>(cols) * k);
    for (float& value : a) {
        value = dist(rng);
    }
    for (float& value : b) {
        value = dist(rng);
    }
    // Sprinkle exact zeros so the +1 pole for zero is tested.
    if (k > 0 && rows > 0) {
        a[static_cast<std::size_t>(k) / 2] = 0.f;
    }

    const std::size_t words = lodestone::packed_words(static_cast<std::size_t>(k));
    std::vector<std::uint64_t> packed_a(static_cast<std::size_t>(rows) * words);
    std::vector<std::uint64_t> packed_b(static_cast<std::size_t>(cols) * words);
    for (int row = 0; row < rows; ++row) {
        lodestone::pack_signs(
            std::span<const float>(a.data() + static_cast<std::size_t>(row) * k, static_cast<std::size_t>(k)),
            std::span<std::uint64_t>(packed_a.data() + static_cast<std::size_t>(row) * words, words));
    }
    for (int col = 0; col < cols; ++col) {
        lodestone::pack_signs(
            std::span<const float>(b.data() + static_cast<std::size_t>(col) * k, static_cast<std::size_t>(k)),
            std::span<std::uint64_t>(packed_b.data() + static_cast<std::size_t>(col) * words, words));
    }

    std::vector<std::int32_t> got(static_cast<std::size_t>(rows) * cols);
    lodestone::binary_gemm(packed_a.data(), packed_b.data(), got.data(),
                           static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                           static_cast<std::size_t>(k));

    for (int row = 0; row < rows; ++row) {
        std::vector<float> arow(a.begin() + static_cast<std::ptrdiff_t>(row) * k,
                                a.begin() + static_cast<std::ptrdiff_t>(row + 1) * k);
        for (int col = 0; col < cols; ++col) {
            std::vector<float> brow(b.begin() + static_cast<std::ptrdiff_t>(col) * k,
                                    b.begin() + static_cast<std::ptrdiff_t>(col + 1) * k);
            const int want = naive_dot(arow, brow);
            EXPECT(got[static_cast<std::size_t>(row) * cols + col] == want);
        }
    }

    const lodestone::Isa isas[] = {lodestone::Isa::Scalar, lodestone::Isa::Avx2, lodestone::Isa::Avx512};
    for (lodestone::Isa isa : isas) {
        if (!lodestone::isa_supported(isa)) {
            continue;
        }
        std::vector<std::int32_t> other(got.size());
        lodestone::binary_gemm(isa, packed_a.data(), packed_b.data(), other.data(),
                               static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                               static_cast<std::size_t>(k));
        EXPECT(other == got);
    }
}

}  // namespace

int main() {
    using lodestone::Isa;

    Isa best = Isa::Scalar;
    if (lodestone::isa_supported(Isa::Avx512)) {
        best = Isa::Avx512;
    } else if (lodestone::isa_supported(Isa::Avx2)) {
        best = Isa::Avx2;
    }
    EXPECT(lodestone::active_isa() == best);
    EXPECT(lodestone::isa_supported(Isa::Scalar));

    // bit 63 and bit 64 fall in different words. +1 is bit 1.
    {
        std::vector<float> values(128, -1.f);
        values[63] = 1.f;
        values[64] = 1.f;
        std::vector<std::uint64_t> words(lodestone::packed_words(values.size()));
        lodestone::pack_signs(values, words);
        EXPECT(words[0] == (std::uint64_t{1} << 63));
        EXPECT(words[1] == std::uint64_t{1});
        EXPECT(lodestone::unpack_pole(words, 63) == 1.f);
        EXPECT(lodestone::unpack_pole(words, 62) == -1.f);
        EXPECT(lodestone::unpack_pole(words, 64) == 1.f);
    }

    // Hand dot: [+ - + +] · [+ + - +] = 1 - 1 - 1 + 1 = 0, and K - 2*popcount(xor) agrees.
    {
        const std::vector<float> a = {1.f, -1.f, 1.f, 1.f};
        const std::vector<float> b = {1.f, 1.f, -1.f, 1.f};
        std::vector<std::uint64_t> pa(1);
        std::vector<std::uint64_t> pb(1);
        lodestone::pack_signs(a, pa);
        lodestone::pack_signs(b, pb);
        std::int32_t dot = 99;
        lodestone::binary_gemm(pa.data(), pb.data(), &dot, 1, 1, 4);
        EXPECT(dot == 0);
        EXPECT(naive_dot(a, b) == 0);
    }

    {
        std::int32_t out[4] = {7, 7, 7, 7};
        lodestone::binary_gemm(nullptr, nullptr, out, 2, 2, 0);
        EXPECT(out[0] == 0 && out[1] == 0 && out[2] == 0 && out[3] == 0);
    }

    {
        bool threw = false;
        try {
            std::vector<std::uint64_t> a(1, 0);
            std::vector<std::uint64_t> b(1, 0);
            std::vector<std::int32_t> c(1, 0);
            lodestone::binary_gemm(std::span<const std::uint64_t>(a), std::span<const std::uint64_t>(b),
                                   std::span<std::int32_t>(c), 2, 2, 64);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        EXPECT(threw);
    }

    std::mt19937 rng(123);
    const int shapes[][3] = {
        {1, 1, 1},   {1, 1, 63}, {1, 1, 64}, {1, 1, 65}, {2, 3, 70},
        {1, 8, 128}, {3, 7, 200}, {8, 8, 256}, {9, 17, 96}, {4, 13, 512},
    };
    for (const auto& shape : shapes) {
        check_shape(shape[0], shape[1], shape[2], rng);
    }

    if (lodestone_failures() != 0) {
        std::fprintf(stderr, "%d failure(s)\n", lodestone_failures());
        return 1;
    }
    std::printf("ok isa %s\n", lodestone::isa_name(lodestone::active_isa()));
    return 0;
}
