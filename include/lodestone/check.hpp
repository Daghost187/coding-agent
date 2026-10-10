#pragma once

#include <stdexcept>
#include <string>

namespace lodestone {

[[noreturn]] inline void fail(const std::string& message) {
    throw std::invalid_argument(message);
}

inline void require(bool condition, const char* message) {
    if (!condition) {
        fail(message);
    }
}

}  // namespace lodestone
