#include "onebit/config.hpp"

#include <stdexcept>

namespace onebit {

void ModelConfig::validate() const {
    if (name.empty()) {
        throw std::invalid_argument("name must not be empty");
    }
    if (
        vocabulary_size == 0 || context_length == 0 || model_width == 0 ||
        layer_count == 0 || query_head_count == 0 ||
        key_value_head_count == 0 || feed_forward_width == 0
    ) {
        throw std::invalid_argument("model dimensions must be positive");
    }
    if (model_width % query_head_count != 0) {
        throw std::invalid_argument(
            "model_width must be divisible by query_head_count"
        );
    }
    if (query_head_count % key_value_head_count != 0) {
        throw std::invalid_argument(
            "query_head_count must be divisible by key_value_head_count"
        );
    }
    if (rotary_base <= 0.0) {
        throw std::invalid_argument("rotary_base must be positive");
    }
}

std::uint64_t ModelConfig::head_width() const {
    validate();
    return model_width / query_head_count;
}

std::uint64_t ModelConfig::key_value_width() const {
    return key_value_head_count * head_width();
}

}  // namespace onebit
