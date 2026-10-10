#include "lodestone/binary.hpp"
#include "lodestone/detail/kernels.hpp"

#include <algorithm>

#if defined(LODESTONE_X86)
#include <cpuid.h>
#endif

namespace lodestone {
namespace {

#if defined(LODESTONE_X86)

unsigned long long read_xcr0() {
    unsigned int eax = 0;
    unsigned int edx = 0;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (static_cast<unsigned long long>(edx) << 32) | eax;
}

bool os_avx() {
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }
    // OSXSAVE must be set, and the OS must have enabled xmm and ymm via XCR0.
    if ((ecx & (1u << 27)) == 0) {
        return false;
    }
    return (read_xcr0() & 0x6ull) == 0x6ull;
}

bool os_avx512() {
    if (!os_avx()) {
        return false;
    }
    // xmm, ymm, opmask, zmm_hi256, hi16_zmm.
    return (read_xcr0() & 0xE6ull) == 0xE6ull;
}

bool cpu_avx2() {
    if (!os_avx()) {
        return false;
    }
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }
    return (ebx & (1u << 5)) != 0;
}

bool cpu_avx512_vpopcnt() {
    if (!os_avx512()) {
        return false;
    }
    unsigned eax = 0;
    unsigned ebx = 0;
    unsigned ecx = 0;
    unsigned edx = 0;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0) {
        return false;
    }
    const bool avx512f = (ebx & (1u << 16)) != 0;
    const bool avx512bw = (ebx & (1u << 30)) != 0;
    const bool vpopcntdq = (ecx & (1u << 14)) != 0;
    return avx512f && avx512bw && vpopcntdq;
}

#endif

using GemmFn = void (*)(const std::uint64_t*, const std::uint64_t*, std::int32_t*,
                        std::size_t, std::size_t, std::size_t);

GemmFn kernel_for(Isa isa) {
#if defined(LODESTONE_X86)
    if (isa == Isa::Avx512) {
        return detail::binary_gemm_avx512;
    }
    if (isa == Isa::Avx2) {
        return detail::binary_gemm_avx2;
    }
#else
    (void)isa;
#endif
    return detail::binary_gemm_scalar;
}

}  // namespace

const char* isa_name(Isa isa) {
    switch (isa) {
    case Isa::Scalar:
        return "scalar";
    case Isa::Avx2:
        return "avx2";
    case Isa::Avx512:
        return "avx512-vpopcntdq";
    }
    return "scalar";
}

bool isa_supported(Isa isa) {
    switch (isa) {
    case Isa::Scalar:
        return true;
#if defined(LODESTONE_X86)
    case Isa::Avx2:
        return cpu_avx2();
    case Isa::Avx512:
        return cpu_avx512_vpopcnt();
#else
    case Isa::Avx2:
    case Isa::Avx512:
        return false;
#endif
    }
    return false;
}

Isa active_isa() {
#if defined(LODESTONE_X86)
    if (cpu_avx512_vpopcnt()) {
        return Isa::Avx512;
    }
    if (cpu_avx2()) {
        return Isa::Avx2;
    }
#endif
    return Isa::Scalar;
}

void binary_gemm(Isa isa, const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                 std::size_t rows, std::size_t cols, std::size_t k) {
    if (rows == 0 || cols == 0) {
        return;
    }
    require(c != nullptr, "binary_gemm: null output");
    require(k <= 0x3fffffff, "binary_gemm: K exceeds the int32 dot range");
    const std::size_t words = packed_words(k);
    if (words == 0) {
        std::fill(c, c + rows * cols, 0);
        return;
    }
    require(a != nullptr && b != nullptr, "binary_gemm: null packed input");
    require(isa_supported(isa), "binary_gemm: ISA is not supported on this CPU");
    kernel_for(isa)(a, b, c, rows, cols, k);
}

static Isa isa_for_width(std::size_t k) {
    const std::size_t words = packed_words(k);
    Isa isa = active_isa();
    // One word per row has its own broadcast kernel. Two to seven words do not
    // fill a zmm, so they drop to AVX2 once a ymm is full, else scalar.
    if (isa == Isa::Avx512 && words != 1 && words < 8) {
        isa = (words >= 4 && isa_supported(Isa::Avx2)) ? Isa::Avx2 : Isa::Scalar;
    }
    if (isa == Isa::Avx2 && words < 4) {
        isa = Isa::Scalar;
    }
    return isa;
}

void binary_gemm(const std::uint64_t* a, const std::uint64_t* b, std::int32_t* c,
                 std::size_t rows, std::size_t cols, std::size_t k) {
    binary_gemm(isa_for_width(k), a, b, c, rows, cols, k);
}

}  // namespace lodestone
