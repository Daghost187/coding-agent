#include "onebit/runtime.hpp"

#include <stdexcept>
#include <utility>

namespace onebit {

void GenerationOptions::validate() const {
    if (max_new_tokens == 0) {
        throw std::invalid_argument("max_new_tokens must be positive");
    }
    if (temperature < 0.0) {
        throw std::invalid_argument("temperature must not be negative");
    }
    if (top_k == 0) {
        throw std::invalid_argument("top_k must be positive");
    }
}

OneBitModel::OneBitModel(ModelConfig config)
    : manifest_(ModelManifest::build(std::move(config))) {}

const ModelManifest& OneBitModel::manifest() const noexcept {
    return manifest_;
}

std::string OneBitModel::complete(
    std::string_view prompt,
    const Tokenizer& tokenizer,
    WeightProvider& weights,
    InferenceBackend& backend,
    const GenerationOptions& options
) const {
    options.validate();
    const auto prompt_tokens = tokenizer.encode(prompt);
    validate_tokens(prompt_tokens);
    const auto generated_tokens = backend.generate(
        manifest_,
        weights,
        prompt_tokens,
        options
    );
    validate_tokens(generated_tokens);
    return tokenizer.decode(generated_tokens);
}

void OneBitModel::validate_tokens(
    std::span<const std::uint32_t> token_ids
) const {
    const auto vocabulary_size = manifest_.config().vocabulary_size;
    for (const auto token_id : token_ids) {
        if (token_id >= vocabulary_size) {
            throw std::out_of_range(
                "token identifier is outside the vocabulary"
            );
        }
    }
}

}  // namespace onebit
