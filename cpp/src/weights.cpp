#include "onebit/weights.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>

namespace onebit {
namespace {

bool is_safe_parameter_name(const std::string& name) {
    if (name.empty()) {
        return false;
    }
    return std::all_of(
        name.begin(),
        name.end(),
        [](unsigned char character) {
            return (
                (character >= 'a' && character <= 'z') ||
                (character >= 'A' && character <= 'Z') ||
                (character >= '0' && character <= '9') ||
                character == '.' || character == '_' || character == '-'
            );
        }
    );
}

}  // namespace

BinaryTensor::BinaryTensor(
    TensorShape shape,
    std::vector<std::uint8_t> packed_data
)
    : shape_(shape), packed_data_(std::move(packed_data)) {
    if (packed_data_.size() != shape_.packed_byte_count()) {
        throw std::invalid_argument("packed tensor byte count is incorrect");
    }
}

const TensorShape& BinaryTensor::shape() const noexcept {
    return shape_;
}

const std::vector<std::uint8_t>& BinaryTensor::packed_data() const noexcept {
    return packed_data_;
}

bool BinaryTensor::is_positive(
    std::uint64_t row,
    std::uint64_t column
) const {
    if (row >= shape_.rows) {
        throw std::out_of_range("row is outside the tensor");
    }
    if (column >= shape_.columns) {
        throw std::out_of_range("column is outside the tensor");
    }
    const auto bit_index = row * shape_.columns + column;
    const auto byte = packed_data_.at(
        static_cast<std::size_t>(bit_index / 8)
    );
    return ((byte >> (bit_index % 8)) & 1U) != 0;
}

int BinaryTensor::sign(
    std::uint64_t row,
    std::uint64_t column
) const {
    return is_positive(row, column) ? 1 : -1;
}

DirectoryWeightProvider::DirectoryWeightProvider(
    std::filesystem::path root
)
    : root_(std::move(root)) {}

BinaryTensor DirectoryWeightProvider::load(
    const ParameterSpec& parameter
) {
    if (!is_safe_parameter_name(parameter.name)) {
        throw std::invalid_argument("unsafe parameter name");
    }
    const auto path = root_ / (parameter.name + ".bin");
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error(
            "unable to read binary tensor: " + path.string()
        );
    }
    std::vector<std::uint8_t> data(
        (std::istreambuf_iterator<char>(input)),
        std::istreambuf_iterator<char>()
    );
    if (!input.eof()) {
        throw std::runtime_error(
            "failed while reading binary tensor: " + path.string()
        );
    }
    return BinaryTensor(parameter.shape, std::move(data));
}

}  // namespace onebit
