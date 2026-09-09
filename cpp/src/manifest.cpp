#include "onebit/manifest.hpp"

#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace onebit {
namespace {

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) {
        throw std::overflow_error("parameter count exceeds uint64");
    }
    return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right) {
    if (
        left != 0 &&
        right > std::numeric_limits<std::uint64_t>::max() / left
    ) {
        throw std::overflow_error("tensor size exceeds uint64");
    }
    return left * right;
}

std::string layer_prefix(std::uint64_t layer_index) {
    std::ostringstream output;
    output << "layers." << std::setfill('0') << std::setw(2) << layer_index;
    return output.str();
}

}  // namespace

std::uint64_t TensorShape::element_count() const {
    if (rows == 0 || columns == 0) {
        throw std::invalid_argument("tensor dimensions must be positive");
    }
    return checked_multiply(rows, columns);
}

std::uint64_t TensorShape::packed_byte_count() const {
    const auto elements = element_count();
    return elements / 8 + (elements % 8 == 0 ? 0 : 1);
}

ModelManifest ModelManifest::build(ModelConfig config) {
    config.validate();
    const auto width = config.model_width;
    const auto kv_width = config.key_value_width();
    const auto ff_width = config.feed_forward_width;

    std::vector<ParameterSpec> parameters;
    parameters.reserve(static_cast<std::size_t>(config.layer_count * 7 + 2));
    parameters.push_back({
        "token_embedding.weight",
        {config.vocabulary_size, width},
    });
    for (
        std::uint64_t layer_index = 0;
        layer_index < config.layer_count;
        ++layer_index
    ) {
        const auto prefix = layer_prefix(layer_index);
        parameters.push_back({
            prefix + ".attention.query.weight",
            {width, width},
        });
        parameters.push_back({
            prefix + ".attention.key.weight",
            {kv_width, width},
        });
        parameters.push_back({
            prefix + ".attention.value.weight",
            {kv_width, width},
        });
        parameters.push_back({
            prefix + ".attention.output.weight",
            {width, width},
        });
        parameters.push_back({
            prefix + ".feed_forward.gate.weight",
            {ff_width, width},
        });
        parameters.push_back({
            prefix + ".feed_forward.up.weight",
            {ff_width, width},
        });
        parameters.push_back({
            prefix + ".feed_forward.down.weight",
            {width, ff_width},
        });
    }
    parameters.push_back({
        "output.weight",
        {config.vocabulary_size, width},
    });
    return ModelManifest(std::move(config), std::move(parameters));
}

ModelManifest::ModelManifest(
    ModelConfig config,
    std::vector<ParameterSpec> parameters
)
    : config_(std::move(config)), parameters_(std::move(parameters)) {}

const ModelConfig& ModelManifest::config() const noexcept {
    return config_;
}

const std::vector<ParameterSpec>& ModelManifest::parameters() const noexcept {
    return parameters_;
}

std::uint64_t ModelManifest::parameter_count() const {
    std::uint64_t total = 0;
    for (const auto& parameter : parameters_) {
        total = checked_add(total, parameter.shape.element_count());
    }
    return total;
}

std::uint64_t ModelManifest::packed_weight_bytes() const {
    std::uint64_t total = 0;
    for (const auto& parameter : parameters_) {
        total = checked_add(total, parameter.shape.packed_byte_count());
    }
    return total;
}

const ParameterSpec& ModelManifest::find(std::string_view name) const {
    for (const auto& parameter : parameters_) {
        if (parameter.name == name) {
            return parameter;
        }
    }
    throw std::out_of_range("parameter not found");
}

}  // namespace onebit
