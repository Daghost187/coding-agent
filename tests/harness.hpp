#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>

inline int& lodestone_failures() {
    static int failures = 0;
    return failures;
}

inline void lodestone_expect(bool condition, const char* expr, const char* file, int line) {
    if (!condition) {
        std::fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
        ++lodestone_failures();
    }
}

#define EXPECT(cond) ::lodestone_expect(static_cast<bool>(cond), #cond, __FILE__, __LINE__)

inline bool near(float a, float b, float tol) {
    const float delta = std::fabs(a - b);
    const float scale = std::max(std::fabs(a), std::fabs(b));
    return delta <= tol || delta <= tol * scale;
}
