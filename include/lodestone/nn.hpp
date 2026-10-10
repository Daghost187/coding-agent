#pragma once

// A small pre-norm transformer whose linear maps are 1-bit.
//
// The residual stream stays fp32. Each BitLinear sign-quantizes its input,
// multiplies by packed ±1 weights, and rescales by mean(|x|) * mean(|W_row|).
// Attention scores and the value mix stay fp32. Squared ReLU is the feed-forward
// nonlinearity (the BitNet choice: one up-projection, no gate).
//
// Scales are stop-gradient. Latent weights train with the straight-through
// estimator: the forward uses sign(w), the backward treats that sign as the
// identity. Call forward, then backward, then adamw. backward matches the last
// forward on this object; the model is not re-entrant.

#include "lodestone/aligned.hpp"
#include "lodestone/binary.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace lodestone {

struct Tensor {
    std::vector<float> value;
    std::vector<float> grad;
    std::vector<float> m;
    std::vector<float> v;
    // BitLinear latents are clamped after the optimizer step so the row scale
    // cannot run away. Embeddings and RMSNorm gains are not.
    bool clamp = false;
    float clamp_abs = 1.f;

    void resize(std::size_t n) {
        value.assign(n, 0.f);
        grad.assign(n, 0.f);
        m.assign(n, 0.f);
        v.assign(n, 0.f);
    }

    void zero_grad() {
        for (float& g : grad) {
            g = 0.f;
        }
    }

    [[nodiscard]] double grad_norm_sq() const {
        double sum = 0.0;
        for (float g : grad) {
            sum += static_cast<double>(g) * static_cast<double>(g);
        }
        return sum;
    }
};

struct Config {
    int vocab = 256;
    int d_model = 64;
    int n_layers = 1;
    int n_heads = 4;
    int d_ff = 256;
    int max_seq = 128;
    float rms_eps = 1e-5f;
    float rope_base = 10000.f;
};

struct TrainConfig {
    float lr = 0.001f;
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    float adam_eps = 1e-8f;
    float weight_decay = 0.f;
    float grad_clip = 1.f;
    // Adam normalizes the step, so 1-bit latents keep moving by about `lr`
    // after the loss is already small. SGD follows the raw gradient and
    // settles when the cross entropy does.
    bool use_adam = true;
};

struct StepResult {
    float loss = 0.f;
    float accuracy = 0.f;
};

enum class RopeDirection : int { Forward = 1, Inverse = -1 };

// In-place rotary embedding. `states` is [tokens, heads, head_dim], head_dim even.
void apply_rope(float* states, int tokens, int heads, int head_dim, float base, RopeDirection direction);

// Causal softmax attention. q, k, v, out are [tokens, heads, head_dim].
// prob is [heads, tokens, tokens], row-softmaxed with a causal mask, and is
// required again by attention_mix_backward.
void attention_mix(const float* q, const float* k, const float* v, float* out, float* prob,
                   int tokens, int heads, int head_dim);

void attention_mix_backward(const float* q, const float* k, const float* v, const float* prob,
                            const float* d_out, float* d_q, float* d_k, float* d_v,
                            int tokens, int heads, int head_dim);

void relu2(const float* x, float* y, int n);
void relu2_backward(const float* x, const float* dy, float* dx, int n);

// Next-token cross entropy. Labels are tokens[1] .. tokens[T-1].
// When d_logits is non-null it must hold T * vocab floats; gradients are added
// into positions 0 .. T-2 and the last row is left untouched.
[[nodiscard]] float cross_entropy(const float* logits, const int* tokens, int tokens_n, int vocab,
                                  float* d_logits);

struct RmsNorm {
    int width = 0;
    float eps = 1e-5f;
    Tensor weight;

    void configure(int width, float eps);
    void forward(const float* x, float* y, float* inv_rms, int rows) const;
    void backward(const float* x, const float* inv_rms, const float* dy, float* dx, int rows);
};

struct BitLinear {
    int in_features = 0;
    int out_features = 0;
    Tensor weight;  // [out, in], row-major latent values
    std::vector<float> scale;  // [out], mean(|w|), stop-gradient
    AlignedVector<std::uint64_t> packed;
    std::size_t words = 0;

    void configure(int in_features, int out_features, std::mt19937& rng);
    void pack();
    void forward(const float* x, float* y, float* alpha, int rows);
    void backward(const float* x, const float* alpha, const float* dy, float* dx, int rows);

private:
    AlignedVector<std::uint64_t> act_packed_;
    std::vector<std::int32_t> dots_;
};

struct Block {
    RmsNorm attn_norm;
    BitLinear qkv;
    BitLinear proj;
    RmsNorm ffn_norm;
    BitLinear up;
    BitLinear down;

    int heads = 0;
    int width = 0;
    int head_dim = 0;
    int ffn_width = 0;
    float rope_base = 10000.f;

    void configure(int width, int heads, int ffn, float eps, float rope_base, std::mt19937& rng);
    void forward(const float* x, int tokens);
    // Adds the input gradient into d_in. d_in and d_out must not alias.
    void backward(const float* d_out, float* d_in, int tokens);
    [[nodiscard]] const float* output() const { return out_.data(); }

private:
    std::vector<float> in_;
    std::vector<float> normed_;
    std::vector<float> inv_attn_;
    std::vector<float> qkv_act_;
    std::vector<float> alpha_qkv_;
    std::vector<float> q_;
    std::vector<float> k_;
    std::vector<float> v_;
    std::vector<float> prob_;
    std::vector<float> mixed_;
    std::vector<float> proj_out_;
    std::vector<float> alpha_proj_;
    std::vector<float> resid_;
    std::vector<float> ffn_normed_;
    std::vector<float> inv_ffn_;
    std::vector<float> up_act_;
    std::vector<float> alpha_up_;
    std::vector<float> hidden_;
    std::vector<float> down_out_;
    std::vector<float> alpha_down_;
    std::vector<float> out_;
};

class Transformer {
public:
    Config cfg;
    Tensor token_embed;
    std::vector<Block> blocks;
    RmsNorm final_norm;
    BitLinear head;

    explicit Transformer(Config config, std::uint32_t seed = 1);

    void forward(std::span<const int> tokens);
    // Teacher-forced next-token loss for the context last passed to forward.
    // Accumulates parameter gradients. Does not apply them.
    [[nodiscard]] float backward(std::span<const int> tokens);
    [[nodiscard]] float accuracy(std::span<const int> tokens) const;

    void zero_grad();
    void adamw(const TrainConfig& opt);
    [[nodiscard]] StepResult train_step(std::span<const int> tokens, const TrainConfig& opt);

    [[nodiscard]] int greedy_next(std::span<const int> tokens);
    [[nodiscard]] std::vector<int> greedy(std::span<const int> prefix, int total_len);

    void save(const std::string& path) const;
    void load(const std::string& path);

    [[nodiscard]] std::span<const float> logits() const {
        return std::span<const float>(logits_.data(), logits_.size());
    }
    [[nodiscard]] int context() const { return context_; }

    [[nodiscard]] std::size_t parameter_count() const {
        std::size_t n = 0;
        for_each_tensor([&](const Tensor& tensor) { n += tensor.value.size(); });
        return n;
    }

    template <class F>
    void for_each_tensor(F&& fn) {
        visit(*this, std::forward<F>(fn));
    }

    template <class F>
    void for_each_tensor(F&& fn) const {
        visit(*this, std::forward<F>(fn));
    }

private:
    template <class Self, class F>
    static void visit(Self& self, F&& fn) {
        fn(self.token_embed);
        for (auto& block : self.blocks) {
            fn(block.attn_norm.weight);
            fn(block.qkv.weight);
            fn(block.proj.weight);
            fn(block.ffn_norm.weight);
            fn(block.up.weight);
            fn(block.down.weight);
        }
        fn(self.final_norm.weight);
        fn(self.head.weight);
    }

    int context_ = 0;
    float b1_pow_ = 1.f;
    float b2_pow_ = 1.f;
    std::vector<float> hidden_;
    std::vector<float> final_in_;
    std::vector<float> final_normed_;
    std::vector<float> inv_final_;
    std::vector<float> head_alpha_;
    std::vector<float> logits_;
};

}  // namespace lodestone
