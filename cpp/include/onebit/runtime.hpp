#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "onebit/manifest.hpp"
#include "onebit/weights.hpp"

namespace onebit {

struct GenerationOptions {
    std::uint64_t max_new_tokens{256};
    double temperature{0.0};
    std::uint64_t top_k{1};

    void validate() const;
};

class Tokenizer {
public:
    virtual ~Tokenizer() = default;
    [[nodiscard]] virtual std::vector<std::uint32_t> encode(
        std::string_view text
    ) const = 0;
    [[nodiscard]] virtual std::string decode(
        std::span<const std::uint32_t> token_ids
    ) const = 0;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    [[nodiscard]] virtual std::vector<std::uint32_t> generate(
        const ModelManifest& manifest,
        WeightProvider& weights,
        std::span<const std::uint32_t> prompt_token_ids,
        const GenerationOptions& options
    ) = 0;
};

class OneBitModel {
public:
    explicit OneBitModel(ModelConfig config = {});

    [[nodiscard]] const ModelManifest& manifest() const noexcept;
    [[nodiscard]] std::string complete(
        std::string_view prompt,
        const Tokenizer& tokenizer,
        WeightProvider& weights,
        InferenceBackend& backend,
        const GenerationOptions& options = {}
    ) const;

private:
    void validate_tokens(std::span<const std::uint32_t> token_ids) const;

    ModelManifest manifest_;
};

}  // namespace onebit
