#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include "onebit/manifest.hpp"

namespace onebit {

class BinaryTensor {
public:
    BinaryTensor(TensorShape shape, std::vector<std::uint8_t> packed_data);

    [[nodiscard]] const TensorShape& shape() const noexcept;
    [[nodiscard]] const std::vector<std::uint8_t>& packed_data() const noexcept;
    [[nodiscard]] bool is_positive(
        std::uint64_t row,
        std::uint64_t column
    ) const;
    [[nodiscard]] int sign(std::uint64_t row, std::uint64_t column) const;

private:
    TensorShape shape_;
    std::vector<std::uint8_t> packed_data_;
};

class WeightProvider {
public:
    virtual ~WeightProvider() = default;
    [[nodiscard]] virtual BinaryTensor load(
        const ParameterSpec& parameter
    ) = 0;
};

class DirectoryWeightProvider final : public WeightProvider {
public:
    explicit DirectoryWeightProvider(std::filesystem::path root);
    [[nodiscard]] BinaryTensor load(
        const ParameterSpec& parameter
    ) override;

private:
    std::filesystem::path root_;
};

}  // namespace onebit
