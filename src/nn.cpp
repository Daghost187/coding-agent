#include "lodestone/nn.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace lodestone {
namespace {

float pole(float value) {
    return value >= 0.f ? 1.f : -1.f;
}

void fit(std::vector<float>& values, int n) {
    if (static_cast<int>(values.size()) != n) {
        values.resize(static_cast<std::size_t>(n));
    }
}

int argmax(const float* row, int n) {
    int best = 0;
    float best_value = row[0];
    for (int i = 1; i < n; ++i) {
        if (row[i] > best_value) {
            best_value = row[i];
            best = i;
        }
    }
    return best;
}

void write_u32(std::ostream& out, std::uint32_t value) {
    if constexpr (std::endian::native != std::endian::little) {
        value = std::byteswap(value);
    }
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

std::uint32_t read_u32(std::istream& in) {
    std::uint32_t value = 0;
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("checkpoint: truncated");
    }
    if constexpr (std::endian::native != std::endian::little) {
        value = std::byteswap(value);
    }
    return value;
}

void write_f32(std::ostream& out, float value) {
    write_u32(out, std::bit_cast<std::uint32_t>(value));
}

float read_f32(std::istream& in) {
    return std::bit_cast<float>(read_u32(in));
}

void write_i32(std::ostream& out, int value) {
    write_u32(out, static_cast<std::uint32_t>(value));
}

void clamp_weight(Tensor& tensor, std::size_t i) {
    if (!tensor.clamp) {
        return;
    }
    const float limit = tensor.clamp_abs;
    tensor.value[i] = std::min(limit, std::max(-limit, tensor.value[i]));
}

void adamw_update(Tensor& tensor, const TrainConfig& opt, float b1_correction, float b2_correction) {
    const std::size_t n = tensor.value.size();
    for (std::size_t i = 0; i < n; ++i) {
        const float g = tensor.grad[i];
        float step = 0.f;
        if (opt.use_adam) {
            tensor.m[i] = opt.beta1 * tensor.m[i] + (1.f - opt.beta1) * g;
            tensor.v[i] = opt.beta2 * tensor.v[i] + (1.f - opt.beta2) * g * g;
            const float mhat = tensor.m[i] / b1_correction;
            const float vhat = tensor.v[i] / b2_correction;
            step = mhat / (std::sqrt(vhat) + opt.adam_eps) + opt.weight_decay * tensor.value[i];
        } else {
            step = g + opt.weight_decay * tensor.value[i];
        }
        tensor.value[i] -= opt.lr * step;
        clamp_weight(tensor, i);
    }
}

}  // namespace

void apply_rope(float* states, int tokens, int heads, int head_dim, float base, RopeDirection direction) {
    require(states != nullptr, "rope: null state");
    require(tokens > 0 && heads > 0 && head_dim > 0, "rope: empty shape");
    require(head_dim % 2 == 0, "rope: head dim must be even");
    require(base > 0.f, "rope: base must be positive");
    const float dir = static_cast<float>(static_cast<int>(direction));
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < heads; ++h) {
            float* pair = states + (static_cast<std::size_t>(t) * heads + h) * static_cast<std::size_t>(head_dim);
            for (int i = 0; i < head_dim; i += 2) {
                const float theta = std::pow(base, -static_cast<float>(i) / static_cast<float>(head_dim));
                const float angle = static_cast<float>(t) * theta;
                const float c = std::cos(angle);
                const float s = std::sin(angle) * dir;
                const float x0 = pair[i];
                const float x1 = pair[i + 1];
                pair[i] = x0 * c - x1 * s;
                pair[i + 1] = x0 * s + x1 * c;
            }
        }
    }
}

void attention_mix(const float* q, const float* k, const float* v, float* out, float* prob,
                   int tokens, int heads, int head_dim) {
    require(q && k && v && out && prob, "attention_mix: null buffer");
    require(tokens > 0 && heads > 0 && head_dim > 0, "attention_mix: empty shape");
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    std::fill(out, out + static_cast<std::size_t>(tokens) * heads * head_dim, 0.f);

    for (int h = 0; h < heads; ++h) {
        for (int t = 0; t < tokens; ++t) {
            const float* query = q + (static_cast<std::size_t>(t) * heads + h) * head_dim;
            float* row = prob + (static_cast<std::size_t>(h) * tokens + t) * tokens;
            float max_score = -std::numeric_limits<float>::infinity();
            for (int j = 0; j <= t; ++j) {
                const float* key = k + (static_cast<std::size_t>(j) * heads + h) * head_dim;
                float dot = 0.f;
                for (int i = 0; i < head_dim; ++i) {
                    dot += query[i] * key[i];
                }
                const float score = dot * scale;
                row[j] = score;
                max_score = std::max(max_score, score);
            }
            float sum = 0.f;
            for (int j = 0; j <= t; ++j) {
                const float e = std::exp(row[j] - max_score);
                row[j] = e;
                sum += e;
            }
            const float inv = 1.f / sum;
            for (int j = 0; j <= t; ++j) {
                row[j] *= inv;
            }
            for (int j = t + 1; j < tokens; ++j) {
                row[j] = 0.f;
            }

            float* mixed = out + (static_cast<std::size_t>(t) * heads + h) * head_dim;
            for (int j = 0; j <= t; ++j) {
                const float* value = v + (static_cast<std::size_t>(j) * heads + h) * head_dim;
                const float weight = row[j];
                for (int i = 0; i < head_dim; ++i) {
                    mixed[i] += weight * value[i];
                }
            }
        }
    }
}

void attention_mix_backward(const float* q, const float* k, const float* v, const float* prob,
                            const float* d_out, float* d_q, float* d_k, float* d_v,
                            int tokens, int heads, int head_dim) {
    require(q && k && v && prob && d_out && d_q && d_k && d_v, "attention_mix_backward: null buffer");
    const float scale = 1.f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> dprob(static_cast<std::size_t>(tokens));

    for (int h = 0; h < heads; ++h) {
        for (int t = 0; t < tokens; ++t) {
            const float* grad = d_out + (static_cast<std::size_t>(t) * heads + h) * head_dim;
            const float* row = prob + (static_cast<std::size_t>(h) * tokens + t) * tokens;
            float dot_pg = 0.f;
            for (int j = 0; j <= t; ++j) {
                const float* value = v + (static_cast<std::size_t>(j) * heads + h) * head_dim;
                float dp = 0.f;
                for (int i = 0; i < head_dim; ++i) {
                    dp += grad[i] * value[i];
                }
                dprob[static_cast<std::size_t>(j)] = dp;
                dot_pg += dp * row[j];
            }

            const float* query = q + (static_cast<std::size_t>(t) * heads + h) * head_dim;
            float* dq = d_q + (static_cast<std::size_t>(t) * heads + h) * head_dim;
            for (int j = 0; j <= t; ++j) {
                const float ds = row[j] * (dprob[static_cast<std::size_t>(j)] - dot_pg) * scale;
                const float* key = k + (static_cast<std::size_t>(j) * heads + h) * head_dim;
                float* dk = d_k + (static_cast<std::size_t>(j) * heads + h) * head_dim;
                float* dv = d_v + (static_cast<std::size_t>(j) * heads + h) * head_dim;
                for (int i = 0; i < head_dim; ++i) {
                    dq[i] += ds * key[i];
                    dk[i] += ds * query[i];
                    dv[i] += row[j] * grad[i];
                }
            }
        }
    }
}

void relu2(const float* x, float* y, int n) {
    for (int i = 0; i < n; ++i) {
        const float v = x[i] > 0.f ? x[i] : 0.f;
        y[i] = v * v;
    }
}

void relu2_backward(const float* x, const float* dy, float* dx, int n) {
    for (int i = 0; i < n; ++i) {
        if (x[i] > 0.f) {
            dx[i] += dy[i] * (2.f * x[i]);
        }
    }
}

float cross_entropy(const float* logits, const int* tokens, int tokens_n, int vocab, float* d_logits) {
    require(tokens_n >= 2, "cross_entropy: need at least two tokens");
    require(vocab >= 1, "cross_entropy: empty vocab");
    require(logits != nullptr && tokens != nullptr, "cross_entropy: null buffer");
    const int predictions = tokens_n - 1;
    const float inv = 1.f / static_cast<float>(predictions);
    double loss = 0.0;
    for (int t = 0; t < predictions; ++t) {
        const int target = tokens[t + 1];
        require(target >= 0 && target < vocab, "cross_entropy: target out of range");
        const float* row = logits + static_cast<std::size_t>(t) * static_cast<std::size_t>(vocab);
        float max_logit = row[0];
        for (int v = 1; v < vocab; ++v) {
            max_logit = std::max(max_logit, row[v]);
        }
        double sum = 0.0;
        for (int v = 0; v < vocab; ++v) {
            sum += std::exp(static_cast<double>(row[v]) - static_cast<double>(max_logit));
        }
        const double log_z = static_cast<double>(max_logit) + std::log(sum);
        loss += log_z - static_cast<double>(row[target]);
        if (d_logits != nullptr) {
            float* drow = d_logits + static_cast<std::size_t>(t) * static_cast<std::size_t>(vocab);
            for (int v = 0; v < vocab; ++v) {
                const float p = static_cast<float>(std::exp(static_cast<double>(row[v]) - log_z));
                drow[v] += p * inv;
            }
            drow[target] -= inv;
        }
    }
    return static_cast<float>(loss / static_cast<double>(predictions));
}

void RmsNorm::configure(int w, float e) {
    require(w > 0, "RmsNorm: width must be positive");
    require(e > 0.f, "RmsNorm: eps must be positive");
    width = w;
    eps = e;
    weight.resize(static_cast<std::size_t>(w));
    std::fill(weight.value.begin(), weight.value.end(), 1.f);
}

void RmsNorm::forward(const float* x, float* y, float* inv_rms, int rows) const {
    require(x && y && inv_rms, "RmsNorm: null buffer");
    for (int row = 0; row < rows; ++row) {
        const float* in = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
        double square = 0.0;
        for (int i = 0; i < width; ++i) {
            square += static_cast<double>(in[i]) * static_cast<double>(in[i]);
        }
        const float inv = 1.f / std::sqrt(static_cast<float>(square / static_cast<double>(width)) + eps);
        inv_rms[row] = inv;
        float* out = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
        for (int i = 0; i < width; ++i) {
            out[i] = in[i] * inv * weight.value[static_cast<std::size_t>(i)];
        }
    }
}

void RmsNorm::backward(const float* x, const float* inv_rms, const float* dy, float* dx, int rows) {
    require(x && inv_rms && dy && dx, "RmsNorm: null buffer");
    for (int row = 0; row < rows; ++row) {
        const float* in = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
        const float* grad = dy + static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
        float* out = dx + static_cast<std::size_t>(row) * static_cast<std::size_t>(width);
        const float inv = inv_rms[row];
        double dot = 0.0;
        for (int i = 0; i < width; ++i) {
            const float xhat = in[i] * inv;
            const float dxhat = weight.value[static_cast<std::size_t>(i)] * grad[i];
            dot += static_cast<double>(dxhat) * static_cast<double>(xhat);
            weight.grad[static_cast<std::size_t>(i)] += xhat * grad[i];
        }
        const float mean = static_cast<float>(dot / static_cast<double>(width));
        for (int i = 0; i < width; ++i) {
            const float xhat = in[i] * inv;
            const float dxhat = weight.value[static_cast<std::size_t>(i)] * grad[i];
            out[i] += inv * (dxhat - xhat * mean);
        }
    }
}

void BitLinear::configure(int in, int out, std::mt19937& rng) {
    require(in > 0 && out > 0, "BitLinear: features must be positive");
    in_features = in;
    out_features = out;
    weight.resize(static_cast<std::size_t>(in) * static_cast<std::size_t>(out));
    const float limit = std::sqrt(3.f / static_cast<float>(in));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& value : weight.value) {
        value = dist(rng);
    }
    weight.clamp = true;
    weight.clamp_abs = 1.f;
    pack();
}

void BitLinear::pack() {
    words = packed_words(static_cast<std::size_t>(in_features));
    packed.assign(static_cast<std::size_t>(out_features) * words, 0);
    scale.resize(static_cast<std::size_t>(out_features));
    for (int n = 0; n < out_features; ++n) {
        const float* row = weight.value.data() + static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features);
        double abs_sum = 0.0;
        std::uint64_t* dst = packed.data() + static_cast<std::size_t>(n) * words;
        for (int i = 0; i < in_features; ++i) {
            const float value = row[i];
            abs_sum += std::fabs(value);
            if (value >= 0.f) {
                dst[static_cast<std::size_t>(i) >> 6] |= std::uint64_t{1} << (i & 63);
            }
        }
        scale[static_cast<std::size_t>(n)] = static_cast<float>(abs_sum / static_cast<double>(in_features));
    }
}

void BitLinear::forward(const float* x, float* y, float* alpha, int rows) {
    require(rows > 0, "BitLinear: empty batch");
    require(x && y && alpha, "BitLinear: null buffer");
    pack();
    act_packed_.resize(static_cast<std::size_t>(rows) * words);
    dots_.resize(static_cast<std::size_t>(rows) * static_cast<std::size_t>(out_features));
    for (int r = 0; r < rows; ++r) {
        const float* row = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features);
        double abs_sum = 0.0;
        for (int i = 0; i < in_features; ++i) {
            abs_sum += std::fabs(row[i]);
        }
        alpha[r] = static_cast<float>(abs_sum / static_cast<double>(in_features));
        pack_signs(std::span<const float>(row, static_cast<std::size_t>(in_features)),
                   std::span<std::uint64_t>(act_packed_.data() + static_cast<std::size_t>(r) * words, words));
    }
    binary_gemm(act_packed_.data(), packed.data(), dots_.data(),
                static_cast<std::size_t>(rows), static_cast<std::size_t>(out_features),
                static_cast<std::size_t>(in_features));
    for (int r = 0; r < rows; ++r) {
        const float act_scale = alpha[r];
        for (int n = 0; n < out_features; ++n) {
            y[static_cast<std::size_t>(r) * static_cast<std::size_t>(out_features) + static_cast<std::size_t>(n)] =
                act_scale * scale[static_cast<std::size_t>(n)] *
                static_cast<float>(dots_[static_cast<std::size_t>(r) * static_cast<std::size_t>(out_features) +
                                         static_cast<std::size_t>(n)]);
        }
    }
}

void BitLinear::backward(const float* x, const float* alpha, const float* dy, float* dx, int rows) {
    require(x && alpha && dy && dx, "BitLinear: null buffer");
    // alpha and per-row weight scales are detached. sign() contributes the
    // straight-through surrogate of 1, so each latent gets ±coeff.
    for (int r = 0; r < rows; ++r) {
        const float* xrow = x + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features);
        const float* grow = dy + static_cast<std::size_t>(r) * static_cast<std::size_t>(out_features);
        float* dxrow = dx + static_cast<std::size_t>(r) * static_cast<std::size_t>(in_features);
        const float act_scale = alpha[r];
        for (int n = 0; n < out_features; ++n) {
            const float coeff = grow[n] * act_scale * scale[static_cast<std::size_t>(n)];
            const float* wrow = weight.value.data() + static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features);
            float* gw = weight.grad.data() + static_cast<std::size_t>(n) * static_cast<std::size_t>(in_features);
            for (int i = 0; i < in_features; ++i) {
                gw[i] += coeff * pole(xrow[i]);
                dxrow[i] += coeff * pole(wrow[i]);
            }
        }
    }
}

void Block::configure(int w, int h, int ffn, float eps, float base, std::mt19937& rng) {
    require(w > 0 && h > 0 && w % h == 0, "block: width must be divisible by heads");
    require((w / h) % 2 == 0, "block: head dim must be even");
    require(ffn > 0, "block: ffn width must be positive");
    require(base > 0.f, "block: rope base must be positive");
    heads = h;
    width = w;
    head_dim = w / h;
    ffn_width = ffn;
    rope_base = base;
    attn_norm.configure(w, eps);
    qkv.configure(w, 3 * w, rng);
    proj.configure(w, w, rng);
    ffn_norm.configure(w, eps);
    up.configure(w, ffn, rng);
    down.configure(ffn, w, rng);
}

void Block::forward(const float* x, int tokens) {
    require(x != nullptr && tokens > 0, "block: empty forward");
    const int features = tokens * width;
    fit(in_, features);
    std::copy(x, x + features, in_.begin());

    fit(normed_, features);
    fit(inv_attn_, tokens);
    attn_norm.forward(in_.data(), normed_.data(), inv_attn_.data(), tokens);

    fit(qkv_act_, tokens * 3 * width);
    fit(alpha_qkv_, tokens);
    qkv.forward(normed_.data(), qkv_act_.data(), alpha_qkv_.data(), tokens);

    fit(q_, features);
    fit(k_, features);
    fit(v_, features);
    for (int t = 0; t < tokens; ++t) {
        const float* row = qkv_act_.data() + static_cast<std::size_t>(t) * 3 * static_cast<std::size_t>(width);
        std::copy(row, row + width, q_.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(width));
        std::copy(row + width, row + 2 * width, k_.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(width));
        std::copy(row + 2 * width, row + 3 * width, v_.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(width));
    }
    apply_rope(q_.data(), tokens, heads, head_dim, rope_base, RopeDirection::Forward);
    apply_rope(k_.data(), tokens, heads, head_dim, rope_base, RopeDirection::Forward);

    fit(mixed_, features);
    fit(prob_, heads * tokens * tokens);
    attention_mix(q_.data(), k_.data(), v_.data(), mixed_.data(), prob_.data(), tokens, heads, head_dim);

    fit(proj_out_, features);
    fit(alpha_proj_, tokens);
    proj.forward(mixed_.data(), proj_out_.data(), alpha_proj_.data(), tokens);

    fit(resid_, features);
    for (int i = 0; i < features; ++i) {
        resid_[static_cast<std::size_t>(i)] = in_[static_cast<std::size_t>(i)] + proj_out_[static_cast<std::size_t>(i)];
    }

    fit(ffn_normed_, features);
    fit(inv_ffn_, tokens);
    ffn_norm.forward(resid_.data(), ffn_normed_.data(), inv_ffn_.data(), tokens);

    const int hidden_n = tokens * ffn_width;
    fit(up_act_, hidden_n);
    fit(alpha_up_, tokens);
    up.forward(ffn_normed_.data(), up_act_.data(), alpha_up_.data(), tokens);

    fit(hidden_, hidden_n);
    relu2(up_act_.data(), hidden_.data(), hidden_n);

    fit(down_out_, features);
    fit(alpha_down_, tokens);
    down.forward(hidden_.data(), down_out_.data(), alpha_down_.data(), tokens);

    fit(out_, features);
    for (int i = 0; i < features; ++i) {
        out_[static_cast<std::size_t>(i)] = resid_[static_cast<std::size_t>(i)] + down_out_[static_cast<std::size_t>(i)];
    }
}

void Block::backward(const float* d_out, float* d_in, int tokens) {
    require(d_out && d_in, "block: null gradient");
    const int features = tokens * width;
    const int hidden_n = tokens * ffn_width;

    std::vector<float> d_hidden(static_cast<std::size_t>(hidden_n), 0.f);
    std::vector<float> d_up(static_cast<std::size_t>(hidden_n), 0.f);
    std::vector<float> d_ffn_normed(static_cast<std::size_t>(features), 0.f);
    std::vector<float> d_resid_ffn(static_cast<std::size_t>(features), 0.f);
    down.backward(hidden_.data(), alpha_down_.data(), d_out, d_hidden.data(), tokens);
    relu2_backward(up_act_.data(), d_hidden.data(), d_up.data(), hidden_n);
    up.backward(ffn_normed_.data(), alpha_up_.data(), d_up.data(), d_ffn_normed.data(), tokens);
    ffn_norm.backward(resid_.data(), inv_ffn_.data(), d_ffn_normed.data(), d_resid_ffn.data(), tokens);

    std::vector<float> d_resid(static_cast<std::size_t>(features), 0.f);
    for (int i = 0; i < features; ++i) {
        d_resid[static_cast<std::size_t>(i)] = d_out[i] + d_resid_ffn[static_cast<std::size_t>(i)];
    }

    std::vector<float> d_mixed(static_cast<std::size_t>(features), 0.f);
    proj.backward(mixed_.data(), alpha_proj_.data(), d_resid.data(), d_mixed.data(), tokens);

    std::vector<float> dq(static_cast<std::size_t>(features), 0.f);
    std::vector<float> dk(static_cast<std::size_t>(features), 0.f);
    std::vector<float> dv(static_cast<std::size_t>(features), 0.f);
    attention_mix_backward(q_.data(), k_.data(), v_.data(), prob_.data(), d_mixed.data(),
                           dq.data(), dk.data(), dv.data(), tokens, heads, head_dim);
    apply_rope(dq.data(), tokens, heads, head_dim, rope_base, RopeDirection::Inverse);
    apply_rope(dk.data(), tokens, heads, head_dim, rope_base, RopeDirection::Inverse);

    std::vector<float> d_qkv(static_cast<std::size_t>(tokens) * 3 * static_cast<std::size_t>(width), 0.f);
    for (int t = 0; t < tokens; ++t) {
        float* row = d_qkv.data() + static_cast<std::size_t>(t) * 3 * static_cast<std::size_t>(width);
        const auto* qrow = dq.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(width);
        const auto* krow = dk.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(width);
        const auto* vrow = dv.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(width);
        std::copy(qrow, qrow + width, row);
        std::copy(krow, krow + width, row + width);
        std::copy(vrow, vrow + width, row + 2 * width);
    }

    std::vector<float> d_normed(static_cast<std::size_t>(features), 0.f);
    qkv.backward(normed_.data(), alpha_qkv_.data(), d_qkv.data(), d_normed.data(), tokens);
    std::vector<float> d_from_attn(static_cast<std::size_t>(features), 0.f);
    attn_norm.backward(in_.data(), inv_attn_.data(), d_normed.data(), d_from_attn.data(), tokens);

    for (int i = 0; i < features; ++i) {
        d_in[i] += d_resid[static_cast<std::size_t>(i)] + d_from_attn[static_cast<std::size_t>(i)];
    }
}

Transformer::Transformer(Config config, std::uint32_t seed) : cfg(std::move(config)) {
    require(cfg.vocab > 0, "config: vocab must be positive");
    require(cfg.d_model > 0, "config: d_model must be positive");
    require(cfg.n_layers >= 0, "config: n_layers must be non-negative");
    require(cfg.n_heads > 0 && cfg.d_model % cfg.n_heads == 0, "config: d_model must be divisible by n_heads");
    require((cfg.d_model / cfg.n_heads) % 2 == 0, "config: head dim must be even");
    require(cfg.d_ff > 0, "config: d_ff must be positive");
    require(cfg.max_seq >= 2, "config: max_seq must be at least 2");
    require(cfg.rms_eps > 0.f, "config: rms_eps must be positive");
    require(cfg.rope_base > 0.f, "config: rope_base must be positive");

    std::mt19937 rng(seed);
    token_embed.resize(static_cast<std::size_t>(cfg.vocab) * static_cast<std::size_t>(cfg.d_model));
    const float limit = std::sqrt(3.f / static_cast<float>(cfg.d_model));
    std::uniform_real_distribution<float> dist(-limit, limit);
    for (float& value : token_embed.value) {
        value = dist(rng);
    }

    blocks.resize(static_cast<std::size_t>(cfg.n_layers));
    for (Block& block : blocks) {
        block.configure(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.rms_eps, cfg.rope_base, rng);
    }
    final_norm.configure(cfg.d_model, cfg.rms_eps);
    head.configure(cfg.d_model, cfg.vocab, rng);
}

void Transformer::forward(std::span<const int> tokens) {
    const int tokens_n = static_cast<int>(tokens.size());
    require(tokens_n >= 1, "forward: empty token span");
    require(tokens_n <= cfg.max_seq, "forward: context exceeds max_seq");
    const int width = cfg.d_model;
    const int vocab = cfg.vocab;
    context_ = tokens_n;

    fit(hidden_, tokens_n * width);
    for (int t = 0; t < tokens_n; ++t) {
        const int id = tokens[static_cast<std::size_t>(t)];
        require(id >= 0 && id < vocab, "forward: token id out of range");
        const float* src = token_embed.value.data() + static_cast<std::size_t>(id) * static_cast<std::size_t>(width);
        std::copy(src, src + width, hidden_.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(width));
    }

    const float* hidden = hidden_.data();
    for (Block& block : blocks) {
        block.forward(hidden, tokens_n);
        hidden = block.output();
    }

    fit(final_in_, tokens_n * width);
    std::copy(hidden, hidden + static_cast<std::size_t>(tokens_n) * static_cast<std::size_t>(width), final_in_.begin());
    fit(final_normed_, tokens_n * width);
    fit(inv_final_, tokens_n);
    final_norm.forward(final_in_.data(), final_normed_.data(), inv_final_.data(), tokens_n);

    fit(head_alpha_, tokens_n);
    fit(logits_, tokens_n * vocab);
    head.forward(final_normed_.data(), logits_.data(), head_alpha_.data(), tokens_n);
}

float Transformer::backward(std::span<const int> tokens) {
    const int tokens_n = static_cast<int>(tokens.size());
    require(tokens_n == context_, "backward: token span does not match the last forward");
    require(tokens_n >= 2, "backward: need at least two tokens");
    const int width = cfg.d_model;
    const int vocab = cfg.vocab;

    std::vector<float> d_logits(static_cast<std::size_t>(tokens_n) * static_cast<std::size_t>(vocab), 0.f);
    const float loss = cross_entropy(logits_.data(), tokens.data(), tokens_n, vocab, d_logits.data());

    std::vector<float> d_normed(static_cast<std::size_t>(tokens_n) * static_cast<std::size_t>(width), 0.f);
    head.backward(final_normed_.data(), head_alpha_.data(), d_logits.data(), d_normed.data(), tokens_n);

    std::vector<float> d_hidden(static_cast<std::size_t>(tokens_n) * static_cast<std::size_t>(width), 0.f);
    final_norm.backward(final_in_.data(), inv_final_.data(), d_normed.data(), d_hidden.data(), tokens_n);

    for (int layer = static_cast<int>(blocks.size()) - 1; layer >= 0; --layer) {
        std::vector<float> d_in(static_cast<std::size_t>(tokens_n) * static_cast<std::size_t>(width), 0.f);
        blocks[static_cast<std::size_t>(layer)].backward(d_hidden.data(), d_in.data(), tokens_n);
        d_hidden.swap(d_in);
    }

    for (int t = 0; t < tokens_n; ++t) {
        const int id = tokens[static_cast<std::size_t>(t)];
        float* grad = token_embed.grad.data() + static_cast<std::size_t>(id) * static_cast<std::size_t>(width);
        const float* dx = d_hidden.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(width);
        for (int i = 0; i < width; ++i) {
            grad[i] += dx[i];
        }
    }
    return loss;
}

float Transformer::accuracy(std::span<const int> tokens) const {
    require(static_cast<int>(tokens.size()) == context_, "accuracy: token span does not match the last forward");
    require(context_ >= 2, "accuracy: need at least two tokens");
    const int predictions = context_ - 1;
    int correct = 0;
    for (int t = 0; t < predictions; ++t) {
        const float* row = logits_.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(cfg.vocab);
        if (argmax(row, cfg.vocab) == tokens[static_cast<std::size_t>(t + 1)]) {
            ++correct;
        }
    }
    return static_cast<float>(correct) / static_cast<float>(predictions);
}

void Transformer::zero_grad() {
    for_each_tensor([](Tensor& tensor) { tensor.zero_grad(); });
}

void Transformer::adamw(const TrainConfig& opt) {
    if (opt.use_adam) {
        require(opt.beta1 > 0.f && opt.beta1 < 1.f, "adam: beta1 must lie in (0, 1)");
        require(opt.beta2 > 0.f && opt.beta2 < 1.f, "adam: beta2 must lie in (0, 1)");
        require(opt.adam_eps > 0.f, "adam: eps must be positive");
    }
    if (opt.grad_clip > 0.f) {
        double sum = 0.0;
        for_each_tensor([&](const Tensor& tensor) { sum += tensor.grad_norm_sq(); });
        const double norm = std::sqrt(sum);
        if (norm > static_cast<double>(opt.grad_clip)) {
            const float scale = static_cast<float>(static_cast<double>(opt.grad_clip) / norm);
            for_each_tensor([&](Tensor& tensor) {
                for (float& grad : tensor.grad) {
                    grad *= scale;
                }
            });
        }
    }
    b1_pow_ *= opt.beta1;
    b2_pow_ *= opt.beta2;
    const float b1_correction = 1.f - b1_pow_;
    const float b2_correction = 1.f - b2_pow_;
    for_each_tensor([&](Tensor& tensor) { adamw_update(tensor, opt, b1_correction, b2_correction); });
}

StepResult Transformer::train_step(std::span<const int> tokens, const TrainConfig& opt) {
    zero_grad();
    forward(tokens);
    StepResult result;
    result.loss = backward(tokens);
    result.accuracy = accuracy(tokens);
    if (!std::isfinite(result.loss)) {
        throw std::runtime_error("non-finite loss");
    }
    adamw(opt);
    return result;
}

int Transformer::greedy_next(std::span<const int> tokens) {
    forward(tokens);
    const int last = static_cast<int>(tokens.size()) - 1;
    const float* row = logits_.data() + static_cast<std::size_t>(last) * static_cast<std::size_t>(cfg.vocab);
    return argmax(row, cfg.vocab);
}

std::vector<int> Transformer::greedy(std::span<const int> prefix, int total_len) {
    require(!prefix.empty(), "greedy: empty prefix");
    require(total_len >= static_cast<int>(prefix.size()), "greedy: total_len shorter than the prefix");
    require(total_len <= cfg.max_seq, "greedy: total_len exceeds max_seq");
    std::vector<int> out(prefix.begin(), prefix.end());
    while (static_cast<int>(out.size()) < total_len) {
        out.push_back(greedy_next(out));
    }
    return out;
}

void Transformer::save(const std::string& path) const {
    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("checkpoint: cannot open for write");
    }
    const char magic[8] = {'L', 'D', 'S', 'T', 'O', 'N', 'E', '1'};
    out.write(magic, 8);
    write_i32(out, 1);
    write_i32(out, cfg.vocab);
    write_i32(out, cfg.d_model);
    write_i32(out, cfg.n_layers);
    write_i32(out, cfg.n_heads);
    write_i32(out, cfg.d_ff);
    write_i32(out, cfg.max_seq);
    write_f32(out, cfg.rms_eps);
    write_f32(out, cfg.rope_base);
    for_each_tensor([&](const Tensor& tensor) {
        write_i32(out, static_cast<int>(tensor.value.size()));
        for (float value : tensor.value) {
            write_f32(out, value);
        }
    });
    if (!out) {
        throw std::runtime_error("checkpoint: write failed");
    }
}

void Transformer::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::in);
    if (!in) {
        throw std::runtime_error("checkpoint: cannot open for read");
    }
    char magic[8] = {};
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "LDSTONE1", 8) != 0) {
        throw std::runtime_error("checkpoint: bad magic");
    }
    const int version = static_cast<int>(read_u32(in));
    if (version != 1) {
        throw std::runtime_error("checkpoint: unsupported version");
    }
    auto expect_i = [&](int want, const char* name) {
        const int got = static_cast<int>(read_u32(in));
        if (got != want) {
            throw std::runtime_error(std::string("checkpoint: config mismatch: ") + name);
        }
    };
    expect_i(cfg.vocab, "vocab");
    expect_i(cfg.d_model, "d_model");
    expect_i(cfg.n_layers, "n_layers");
    expect_i(cfg.n_heads, "n_heads");
    expect_i(cfg.d_ff, "d_ff");
    expect_i(cfg.max_seq, "max_seq");
    const float eps = read_f32(in);
    const float base = read_f32(in);
    if (std::bit_cast<std::uint32_t>(eps) != std::bit_cast<std::uint32_t>(cfg.rms_eps) ||
        std::bit_cast<std::uint32_t>(base) != std::bit_cast<std::uint32_t>(cfg.rope_base)) {
        throw std::runtime_error("checkpoint: config mismatch: constants");
    }
    for_each_tensor([&](Tensor& tensor) {
        const int n = static_cast<int>(read_u32(in));
        if (n != static_cast<int>(tensor.value.size())) {
            throw std::runtime_error("checkpoint: tensor size mismatch");
        }
        for (float& value : tensor.value) {
            value = read_f32(in);
        }
        std::fill(tensor.grad.begin(), tensor.grad.end(), 0.f);
        std::fill(tensor.m.begin(), tensor.m.end(), 0.f);
        std::fill(tensor.v.begin(), tensor.v.end(), 0.f);
    });
    b1_pow_ = 1.f;
    b2_pow_ = 1.f;
    for (Block& block : blocks) {
        block.qkv.pack();
        block.proj.pack();
        block.up.pack();
        block.down.pack();
    }
    head.pack();
}

}  // namespace lodestone
