#include <exception>
#include <iostream>
#include <string>

#include "onebit/manifest.hpp"
#include "onebit/weights.hpp"

namespace {

void print_description(const onebit::ModelManifest& manifest) {
    const auto& config = manifest.config();
    std::cout
        << "name=" << config.name << '\n'
        << "weight_encoding=binary-sign-lsb-first\n"
        << "vocabulary_size=" << config.vocabulary_size << '\n'
        << "context_length=" << config.context_length << '\n'
        << "model_width=" << config.model_width << '\n'
        << "layer_count=" << config.layer_count << '\n'
        << "query_head_count=" << config.query_head_count << '\n'
        << "key_value_head_count=" << config.key_value_head_count << '\n'
        << "head_width=" << config.head_width() << '\n'
        << "feed_forward_width=" << config.feed_forward_width << '\n'
        << "tensor_count=" << manifest.parameters().size() << '\n'
        << "parameter_count=" << manifest.parameter_count() << '\n'
        << "packed_weight_bytes=" << manifest.packed_weight_bytes() << '\n';
}

int run(int argument_count, char** arguments) {
    if (argument_count < 2) {
        throw std::invalid_argument(
            "usage: onebit-llm <describe|list-tensors|validate-weights>"
        );
    }
    const std::string command = arguments[1];
    const auto manifest = onebit::ModelManifest::build();

    if (command == "describe") {
        print_description(manifest);
        return 0;
    }
    if (command == "list-tensors") {
        for (const auto& parameter : manifest.parameters()) {
            std::cout
                << parameter.name << '\t'
                << parameter.shape.rows << '\t'
                << parameter.shape.columns << '\t'
                << parameter.shape.packed_byte_count() << '\n';
        }
        return 0;
    }
    if (command == "validate-weights") {
        if (argument_count != 3) {
            throw std::invalid_argument(
                "usage: onebit-llm validate-weights <directory>"
            );
        }
        onebit::DirectoryWeightProvider provider(arguments[2]);
        for (const auto& parameter : manifest.parameters()) {
            static_cast<void>(provider.load(parameter));
        }
        std::cout
            << "validated_tensor_count="
            << manifest.parameters().size()
            << '\n';
        return 0;
    }
    throw std::invalid_argument("unknown command: " + command);
}

}  // namespace

int main(int argument_count, char** arguments) {
    try {
        return run(argument_count, arguments);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
