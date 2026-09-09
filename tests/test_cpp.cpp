#include <cassert>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "onebit/manifest.hpp"
#include "onebit/runtime.hpp"
#include "onebit/weights.hpp"

namespace {

class StaticTokenizer final : public onebit::Tokenizer {
public:
    std::vector<std::uint32_t> encode(
        std::string_view text
    ) const override {
        return {static_cast<std::uint32_t>(text.size())};
    }

    std::string decode(
        std::span<const std::uint32_t> token_ids
    ) const override {
        assert(token_ids.size() == 2);
        return std::to_string(token_ids[0]) + "," +
            std::to_string(token_ids[1]);
    }
};

class UnusedWeightProvider final : public onebit::WeightProvider {
public:
    onebit::BinaryTensor load(
        const onebit::ParameterSpec&
    ) override {
        assert(false);
        return {{1, 1}, {0}};
    }
};

class StaticBackend final : public onebit::InferenceBackend {
public:
    std::vector<std::uint32_t> generate(
        const onebit::ModelManifest& manifest,
        onebit::WeightProvider&,
        std::span<const std::uint32_t> prompt_token_ids,
        const onebit::GenerationOptions& options
    ) override {
        assert(manifest.parameter_count() == 2'000'027'648ULL);
        assert(prompt_token_ids.size() == 1);
        assert(prompt_token_ids[0] == 4);
        assert(options.max_new_tokens == 12);
        return {7, 8};
    }
};

}  // namespace

int main() {
    const auto manifest = onebit::ModelManifest::build();
    assert(manifest.parameter_count() == 2'000'027'648ULL);
    assert(manifest.packed_weight_bytes() == 250'003'456ULL);
    assert(manifest.parameters().size() == 282);
    assert(manifest.config().head_width() == 128);
    assert(manifest.config().key_value_width() == 1'024);

    const onebit::BinaryTensor tensor(
        {2, 4},
        {0b1010'0101}
    );
    const std::vector<int> expected{1, -1, 1, -1, -1, 1, -1, 1};
    std::vector<int> actual;
    for (std::uint64_t row = 0; row < 2; ++row) {
        for (std::uint64_t column = 0; column < 4; ++column) {
            actual.push_back(tensor.sign(row, column));
        }
    }
    assert(actual == expected);

    const onebit::OneBitModel model;
    const StaticTokenizer tokenizer;
    UnusedWeightProvider weights;
    StaticBackend backend;
    onebit::GenerationOptions options;
    options.max_new_tokens = 12;
    assert(
        model.complete("code", tokenizer, weights, backend, options) == "7,8"
    );
}
