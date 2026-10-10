#pragma once

#include <cstddef>
#include <new>
#include <vector>

namespace lodestone {

// 64-byte alignment so packed bit rows can sit on a cache line and a zmm load.
template <typename T>
struct AlignedAlloc {
    using value_type = T;
    static constexpr std::align_val_t kAlign{64};

    AlignedAlloc() noexcept = default;

    template <typename U>
    AlignedAlloc(const AlignedAlloc<U>&) noexcept {}

    template <typename U>
    struct rebind {
        using other = AlignedAlloc<U>;
    };

    T* allocate(std::size_t n) {
        if (n == 0) {
            n = 1;
        }
        return static_cast<T*>(::operator new[](n * sizeof(T), kAlign));
    }

    void deallocate(T* pointer, std::size_t) noexcept {
        ::operator delete[](pointer, kAlign);
    }

    bool operator==(const AlignedAlloc&) const noexcept { return true; }
};

template <typename T>
using AlignedVector = std::vector<T, AlignedAlloc<T>>;

}  // namespace lodestone
