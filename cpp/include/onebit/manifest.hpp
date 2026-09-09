#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "onebit/config.hpp"

namespace onebit {

struct TensorShape {
    std::uint64_t rows;
    std::uint64_t columns;

    [[nodiscard]] std::uint64_t element_count() const;
    [[nodiscard]] std::uint64_t packed_byte_count() const;
};

struct ParameterSpec {
    std::string name;
    TensorShape shape;
};

class ModelManifest {
public:
    static ModelManifest build(ModelConfig config = {});

    [[nodiscard]] const ModelConfig& config() const noexcept;
    [[nodiscard]] const std::vector<ParameterSpec>& parameters() const noexcept;
    [[nodiscard]] std::uint64_t parameter_count() const;
    [[nodiscard]] std::uint64_t packed_weight_bytes() const;
    [[nodiscard]] const ParameterSpec& find(std::string_view name) const;

private:
    ModelManifest(ModelConfig config, std::vector<ParameterSpec> parameters);

    ModelConfig config_;
    std::vector<ParameterSpec> parameters_;
};

}  // namespace onebit
