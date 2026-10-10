#include "lodestone/binary.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

std::int64_t checksum(const std::vector<std::int32_t>& values) {
    std::int64_t sum = 0;
    for (std::int32_t value : values) {
        sum += value;
    }
    return sum;
}

bool kernels_agree(int rows, int cols, int k) {
    const std::size_t words = lodestone::packed_words(static_cast<std::size_t>(k));
    std::mt19937 rng(7);
    std::vector<std::uint64_t> a(static_cast<std::size_t>(rows) * words);
    std::vector<std::uint64_t> b(static_cast<std::size_t>(cols) * words);
    for (std::uint64_t& word : a) {
        word = (static_cast<std::uint64_t>(rng()) << 32) ^ rng();
    }
    for (std::uint64_t& word : b) {
        word = (static_cast<std::uint64_t>(rng()) << 32) ^ rng();
    }
    // Clear padding bits in the tail word so every kernel sees the same K.
    if (k % 64 != 0) {
        const std::uint64_t mask = (std::uint64_t{1} << (k % 64)) - 1;
        for (std::size_t row = 0; row < static_cast<std::size_t>(rows); ++row) {
            a[row * words + words - 1] &= mask;
        }
        for (std::size_t col = 0; col < static_cast<std::size_t>(cols); ++col) {
            b[col * words + words - 1] &= mask;
        }
    }

    std::vector<std::int32_t> reference(static_cast<std::size_t>(rows) * cols);
    lodestone::binary_gemm(lodestone::Isa::Scalar, a.data(), b.data(), reference.data(),
                           static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                           static_cast<std::size_t>(k));
    const lodestone::Isa isas[] = {lodestone::Isa::Avx2, lodestone::Isa::Avx512};
    for (lodestone::Isa isa : isas) {
        if (!lodestone::isa_supported(isa)) {
            continue;
        }
        std::vector<std::int32_t> got(reference.size());
        lodestone::binary_gemm(isa, a.data(), b.data(), got.data(),
                               static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                               static_cast<std::size_t>(k));
        if (got != reference) {
            std::fprintf(stderr, "mismatch %s on %d x %d x %d\n", lodestone::isa_name(isa), rows, cols, k);
            return false;
        }
    }
    return true;
}

double seconds_per_call(lodestone::Isa isa, int rows, int cols, int k, std::int64_t* sink) {
    const std::size_t words = lodestone::packed_words(static_cast<std::size_t>(k));
    std::mt19937 rng(1);
    std::vector<std::uint64_t> a(static_cast<std::size_t>(rows) * words);
    std::vector<std::uint64_t> b(static_cast<std::size_t>(cols) * words);
    std::vector<std::int32_t> c(static_cast<std::size_t>(rows) * cols);
    for (std::uint64_t& word : a) {
        word = (static_cast<std::uint64_t>(rng()) << 32) ^ rng();
    }
    for (std::uint64_t& word : b) {
        word = (static_cast<std::uint64_t>(rng()) << 32) ^ rng();
    }

    lodestone::binary_gemm(isa, a.data(), b.data(), c.data(),
                           static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                           static_cast<std::size_t>(k));

    int iters = 1;
    double elapsed = 0.0;
    do {
        const auto start = Clock::now();
        for (int i = 0; i < iters; ++i) {
            lodestone::binary_gemm(isa, a.data(), b.data(), c.data(),
                                   static_cast<std::size_t>(rows), static_cast<std::size_t>(cols),
                                   static_cast<std::size_t>(k));
        }
        elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        if (elapsed < 0.20 && iters < 4096) {
            iters *= 2;
        } else {
            break;
        }
    } while (true);

    *sink += checksum(c);
    return elapsed / static_cast<double>(iters);
}

}  // namespace

int main() {
    if (!kernels_agree(3, 5, 70) || !kernels_agree(17, 19, 512)) {
        return 1;
    }

    std::printf("active isa: %s\n", lodestone::isa_name(lodestone::active_isa()));
    std::printf("%8s %8s %8s %22s %14s %12s\n", "M", "N", "K", "kernel", "us", "GMAC/s");

    const int shapes[][3] = {
        {64, 256, 64},
        {64, 256, 256},
        {256, 256, 1024},
        {512, 512, 2048},
    };
    const lodestone::Isa isas[] = {lodestone::Isa::Scalar, lodestone::Isa::Avx2, lodestone::Isa::Avx512};
    std::int64_t sink = 0;
    for (const auto& shape : shapes) {
        for (lodestone::Isa isa : isas) {
            if (!lodestone::isa_supported(isa)) {
                continue;
            }
            const double seconds = seconds_per_call(isa, shape[0], shape[1], shape[2], &sink);
            const double macs = static_cast<double>(shape[0]) * shape[1] * shape[2];
            const double gmacs = macs / seconds / 1e9;
            std::printf("%8d %8d %8d %22s %14.2f %12.2f\n",
                        shape[0], shape[1], shape[2], lodestone::isa_name(isa), seconds * 1e6, gmacs);
        }
    }
    if (sink == 0x7fffffffffffffffLL) {
        std::printf("sink\n");
    }
    return 0;
}
