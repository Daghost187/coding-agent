#pragma once

#include <cstdint>
#include <string>

namespace onebit {

struct ModelConfig {
    std::string name{"onebit-code-2b"};
    std::uint64_t vocabulary_size{32'768};
    std::uint64_t context_length{4'096};
    std::uint64_t model_width{2'048};
    std::uint64_t layer_count{40};
    std::uint64_t query_head_count{16};
    std::uint64_t key_value_head_count{8};
    std::uint64_t feed_forward_width{5'544};
    double rotary_base{10'000.0};

    void validate() const;
    [[nodiscard]] std::uint64_t head_width() const;
    [[nodiscard]] std::uint64_t key_value_width() const;
};

}  // namespace onebit
