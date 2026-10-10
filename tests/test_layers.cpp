#include "harness.hpp"
#include "lodestone/lodestone.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

float finite_difference(float& slot, auto&& function, float eps = 1e-3f) {
    const float original = slot;
    slot = original + eps;
    const float plus = function();
    slot = original - eps;
    const float minus = function();
    slot = original;
    return (plus - minus) / (2.f * eps);
}

void expect_close(float got, float want, float tol, const char* label) {
    if (!near(got, want, tol)) {
        std::fprintf(stderr, "%s got %.8g want %.8g\n", label, got, want);
        EXPECT(false);
    }
}

}  // namespace

int main() {
    // Anchor from the README. Signs, scales, and the integer dot are all exact in f32.
    {
        lodestone::BitLinear layer;
        std::mt19937 rng(1);
        layer.configure(4, 1, rng);
        layer.weight.value = {0.5f, -0.25f, 0.f, -1.f};
        const std::vector<float> x = {1.f, -2.f, 0.f, 3.f};
        std::vector<float> y(1);
        std::vector<float> alpha(1);
        layer.forward(x.data(), y.data(), alpha.data(), 1);
        EXPECT(y[0] == 1.3125f);
        EXPECT(near(alpha[0], 1.5f, 0.f));
        EXPECT(near(layer.scale[0], 0.4375f, 0.f));

        std::vector<float> dy = {1.f};
        std::vector<float> dx(4, 0.f);
        layer.weight.zero_grad();
        layer.backward(x.data(), alpha.data(), dy.data(), dx.data(), 1);
        const float coeff = 1.f * 1.5f * 0.4375f;
        const float sx[4] = {1.f, -1.f, 1.f, 1.f};
        const float sw[4] = {1.f, -1.f, 1.f, -1.f};
        for (int i = 0; i < 4; ++i) {
            expect_close(layer.weight.grad[static_cast<std::size_t>(i)], coeff * sx[i], 1e-6f, "gw");
            expect_close(dx[static_cast<std::size_t>(i)], coeff * sw[i], 1e-6f, "gx");
        }
    }

    // Batch and multi-row indexing against an independent naive loop.
    {
        constexpr int kIn = 5;
        constexpr int kOut = 3;
        constexpr int kBatch = 2;
        lodestone::BitLinear layer;
        std::mt19937 rng(2);
        layer.configure(kIn, kOut, rng);
        std::vector<float> x = {
            0.2f, -1.f, 0.f, 3.f, -0.5f,
            -2.f, 0.4f, 1.f, -0.3f, 0.f,
        };
        std::vector<float> y(kBatch * kOut);
        std::vector<float> alpha(kBatch);
        layer.forward(x.data(), y.data(), alpha.data(), kBatch);

        auto pole = [](float v) { return v >= 0.f ? 1.f : -1.f; };
        for (int row = 0; row < kBatch; ++row) {
            double abs_sum = 0.0;
            for (int i = 0; i < kIn; ++i) {
                abs_sum += std::fabs(x[static_cast<std::size_t>(row) * kIn + i]);
            }
            const float want_alpha = static_cast<float>(abs_sum / kIn);
            expect_close(alpha[static_cast<std::size_t>(row)], want_alpha, 1e-6f, "alpha");
            for (int n = 0; n < kOut; ++n) {
                int dot = 0;
                double wabs = 0.0;
                for (int i = 0; i < kIn; ++i) {
                    const float w = layer.weight.value[static_cast<std::size_t>(n) * kIn + i];
                    wabs += std::fabs(w);
                    dot += static_cast<int>(pole(x[static_cast<std::size_t>(row) * kIn + i]) * pole(w));
                }
                const float scale = static_cast<float>(wabs / kIn);
                const float want = want_alpha * scale * static_cast<float>(dot);
                expect_close(y[static_cast<std::size_t>(row) * kOut + n], want, 1e-5f, "y");
            }
        }
    }

    {
        lodestone::RmsNorm norm;
        norm.configure(8, 1e-5f);
        for (int i = 0; i < 8; ++i) {
            norm.weight.value[static_cast<std::size_t>(i)] = 0.5f + 0.1f * static_cast<float>(i);
        }
        std::vector<float> x = {0.5f, -1.2f, 0.3f, 0.8f, -0.4f, 1.1f, -0.7f, 0.2f};
        std::vector<float> y(8);
        std::vector<float> inv(1);
        auto loss = [&]() {
            norm.forward(x.data(), y.data(), inv.data(), 1);
            double sum = 0.0;
            for (float value : y) {
                sum += value;
            }
            return static_cast<float>(sum);
        };
        const float base = loss();
        EXPECT(std::isfinite(base));
        std::vector<float> dy(8, 1.f);
        std::vector<float> dx(8, 0.f);
        norm.weight.zero_grad();
        norm.backward(x.data(), inv.data(), dy.data(), dx.data(), 1);
        for (int i = 0; i < 8; ++i) {
            const float numeric_x = finite_difference(x[static_cast<std::size_t>(i)], loss);
            expect_close(dx[static_cast<std::size_t>(i)], numeric_x, 2e-3f, "rmsnorm dx");
            const float numeric_g = finite_difference(norm.weight.value[static_cast<std::size_t>(i)], loss);
            expect_close(norm.weight.grad[static_cast<std::size_t>(i)], numeric_g, 2e-3f, "rmsnorm dg");
        }
    }

    {
        const std::vector<float> x = {1.f, -2.f, 0.5f, 3.f, -0.25f};
        std::vector<float> y(x.size());
        lodestone::relu2(x.data(), y.data(), static_cast<int>(x.size()));
        EXPECT(y[0] == 1.f);
        EXPECT(y[1] == 0.f);
        EXPECT(near(y[2], 0.25f, 1e-6f));
        std::vector<float> dy(x.size(), 1.f);
        std::vector<float> dx(x.size(), 0.f);
        lodestone::relu2_backward(x.data(), dy.data(), dx.data(), static_cast<int>(x.size()));
        EXPECT(dx[0] == 2.f);
        EXPECT(dx[1] == 0.f);
        EXPECT(near(dx[2], 1.f, 1e-6f));
    }

    {
        constexpr int kTokens = 4;
        constexpr int kHeads = 2;
        constexpr int kHeadDim = 4;
        constexpr int kWidth = kHeads * kHeadDim;
        constexpr float kBase = 10000.f;
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        std::vector<float> q0(kTokens * kWidth);
        std::vector<float> k0(kTokens * kWidth);
        std::vector<float> v0(kTokens * kWidth);
        for (float& value : q0) value = dist(rng);
        for (float& value : k0) value = dist(rng);
        for (float& value : v0) value = dist(rng);

        std::vector<float> q = q0;
        std::vector<float> k = k0;
        std::vector<float> v = v0;
        lodestone::apply_rope(q.data(), kTokens, kHeads, kHeadDim, kBase, lodestone::RopeDirection::Forward);
        lodestone::apply_rope(k.data(), kTokens, kHeads, kHeadDim, kBase, lodestone::RopeDirection::Forward);
        std::vector<float> roundtrip = q;
        lodestone::apply_rope(roundtrip.data(), kTokens, kHeads, kHeadDim, kBase, lodestone::RopeDirection::Inverse);
        for (std::size_t i = 0; i < q0.size(); ++i) {
            expect_close(roundtrip[i], q0[i], 1e-5f, "rope inverse");
        }

        auto loss_of = [&](const std::vector<float>& qq, const std::vector<float>& kk, const std::vector<float>& vv) {
            std::vector<float> q1 = qq;
            std::vector<float> k1 = kk;
            std::vector<float> out(qq.size());
            std::vector<float> prob(static_cast<std::size_t>(kHeads) * kTokens * kTokens);
            lodestone::apply_rope(q1.data(), kTokens, kHeads, kHeadDim, kBase, lodestone::RopeDirection::Forward);
            lodestone::apply_rope(k1.data(), kTokens, kHeads, kHeadDim, kBase, lodestone::RopeDirection::Forward);
            lodestone::attention_mix(q1.data(), k1.data(), vv.data(), out.data(), prob.data(),
                                     kTokens, kHeads, kHeadDim);
            double sum = 0.0;
            for (float value : out) {
                sum += value;
            }
            return static_cast<float>(sum);
        };

        std::vector<float> out(q.size());
        std::vector<float> prob(static_cast<std::size_t>(kHeads) * kTokens * kTokens);
        lodestone::attention_mix(q.data(), k.data(), v.data(), out.data(), prob.data(), kTokens, kHeads, kHeadDim);
        std::vector<float> dy(out.size(), 1.f);
        std::vector<float> dq(out.size(), 0.f);
        std::vector<float> dk(out.size(), 0.f);
        std::vector<float> dv(out.size(), 0.f);
        lodestone::attention_mix_backward(q.data(), k.data(), v.data(), prob.data(), dy.data(),
                                          dq.data(), dk.data(), dv.data(), kTokens, kHeads, kHeadDim);
        lodestone::apply_rope(dq.data(), kTokens, kHeads, kHeadDim, kBase, lodestone::RopeDirection::Inverse);
        lodestone::apply_rope(dk.data(), kTokens, kHeads, kHeadDim, kBase, lodestone::RopeDirection::Inverse);

        const int probes[] = {0, 3, 7, kWidth + 1};
        for (int index : probes) {
            const float numeric = finite_difference(q0[static_cast<std::size_t>(index)], [&]() {
                return loss_of(q0, k0, v0);
            });
            expect_close(dq[static_cast<std::size_t>(index)], numeric, 2e-3f, "attn dq");
        }
        const float numeric_k = finite_difference(k0[5], [&]() { return loss_of(q0, k0, v0); });
        expect_close(dk[5], numeric_k, 2e-3f, "attn dk");
        const float numeric_v = finite_difference(v0[2], [&]() { return loss_of(q0, k0, v0); });
        expect_close(dv[2], numeric_v, 2e-3f, "attn dv");
    }

    {
        constexpr int kTokens = 3;
        constexpr int kVocab = 4;
        std::vector<float> logits = {
            0.f, 0.f, 0.f, 0.f,
            1.f, -1.f, 0.5f, 0.25f,
            0.f, 0.f, 0.f, 0.f,
        };
        const int tokens[] = {0, 1, 2};
        const float zeros[] = {0.f, 0.f};
        const int pair[] = {0, 1};
        const float loss_uniform = lodestone::cross_entropy(zeros, pair, 2, 2, nullptr);
        EXPECT(near(loss_uniform, std::log(2.f), 1e-6f));

        std::vector<float> d_logits(logits.size(), 0.f);
        const float loss = lodestone::cross_entropy(logits.data(), tokens, kTokens, kVocab, d_logits.data());
        EXPECT(std::isfinite(loss));
        for (int i = 0; i < kTokens * kVocab; ++i) {
            if (i >= (kTokens - 1) * kVocab) {
                EXPECT(d_logits[static_cast<std::size_t>(i)] == 0.f);
                continue;
            }
            const float numeric = finite_difference(logits[static_cast<std::size_t>(i)], [&]() {
                return lodestone::cross_entropy(logits.data(), tokens, kTokens, kVocab, nullptr);
            }, 1e-3f);
            expect_close(d_logits[static_cast<std::size_t>(i)], numeric, 2e-3f, "ce");
        }
    }

    {
        lodestone::Config cfg;
        cfg.vocab = 5;
        cfg.d_model = 8;
        cfg.n_layers = 0;
        cfg.n_heads = 2;
        cfg.d_ff = 8;
        cfg.max_seq = 8;
        lodestone::Transformer model(cfg, 3);
        const std::vector<int> tokens = {0, 1, 2, 1};
        model.zero_grad();
        model.forward(tokens);
        const std::vector<float> saved_logits(model.logits().begin(), model.logits().end());
        const float loss = model.backward(tokens);
        EXPECT(std::isfinite(loss));
        const std::vector<float> g_embed = model.token_embed.grad;
        const std::vector<float> g_norm = model.final_norm.weight.grad;
        const std::vector<float> g_head = model.head.weight.grad;

        model.zero_grad();
        const int tokens_n = static_cast<int>(tokens.size());
        const int width = cfg.d_model;
        const int vocab = cfg.vocab;
        std::vector<float> embedded(static_cast<std::size_t>(tokens_n) * width);
        for (int t = 0; t < tokens_n; ++t) {
            const float* src = model.token_embed.value.data() + static_cast<std::size_t>(tokens[static_cast<std::size_t>(t)]) * width;
            std::copy(src, src + width, embedded.data() + static_cast<std::size_t>(t) * width);
        }
        std::vector<float> normed(embedded.size());
        std::vector<float> inv(static_cast<std::size_t>(tokens_n));
        model.final_norm.forward(embedded.data(), normed.data(), inv.data(), tokens_n);
        std::vector<float> alpha(static_cast<std::size_t>(tokens_n));
        std::vector<float> logits(static_cast<std::size_t>(tokens_n) * vocab);
        model.head.forward(normed.data(), logits.data(), alpha.data(), tokens_n);
        EXPECT(logits == saved_logits);

        std::vector<float> d_logits(logits.size(), 0.f);
        const float manual_loss = lodestone::cross_entropy(logits.data(), tokens.data(), tokens_n, vocab, d_logits.data());
        EXPECT(near(manual_loss, loss, 1e-6f));
        std::vector<float> d_normed(embedded.size(), 0.f);
        model.head.backward(normed.data(), alpha.data(), d_logits.data(), d_normed.data(), tokens_n);
        std::vector<float> d_embed_rows(embedded.size(), 0.f);
        model.final_norm.backward(embedded.data(), inv.data(), d_normed.data(), d_embed_rows.data(), tokens_n);
        std::vector<float> g_embed_manual(model.token_embed.grad.size(), 0.f);
        for (int t = 0; t < tokens_n; ++t) {
            float* dst = g_embed_manual.data() + static_cast<std::size_t>(tokens[static_cast<std::size_t>(t)]) * width;
            const float* src = d_embed_rows.data() + static_cast<std::size_t>(t) * width;
            for (int i = 0; i < width; ++i) {
                dst[i] += src[i];
            }
        }
        for (std::size_t i = 0; i < g_embed.size(); ++i) {
            expect_close(g_embed_manual[i], g_embed[i], 1e-5f, "embed grad");
        }
        for (std::size_t i = 0; i < g_head.size(); ++i) {
            expect_close(model.head.weight.grad[i], g_head[i], 1e-5f, "head grad");
        }
        for (std::size_t i = 0; i < g_norm.size(); ++i) {
            expect_close(model.final_norm.weight.grad[i], g_norm[i], 1e-5f, "norm grad");
        }
    }

    {
        lodestone::Config cfg;
        cfg.vocab = 8;
        cfg.d_model = 16;
        cfg.n_layers = 1;
        cfg.n_heads = 2;
        cfg.d_ff = 32;
        cfg.max_seq = 8;
        lodestone::Transformer model(cfg, 5);
        const std::vector<int> tokens = {1, 2, 3, 2, 1};
        const std::vector<float> before = model.blocks[0].qkv.weight.value;
        lodestone::TrainConfig opt;
        opt.lr = 0.05f;
        const lodestone::StepResult step = model.train_step(tokens, opt);
        EXPECT(std::isfinite(step.loss));
        EXPECT(step.loss > 0.f);
        EXPECT(step.loss < 20.f);
        bool changed = false;
        for (std::size_t i = 0; i < before.size(); ++i) {
            if (model.blocks[0].qkv.weight.value[i] != before[i]) {
                changed = true;
            }
        }
        EXPECT(changed);

        const std::string path = "/tmp/lodestone-roundtrip.ldst";
        model.save(path);
        lodestone::Transformer loaded(cfg, 99);
        loaded.load(path);
        loaded.forward(tokens);
        model.forward(tokens);
        EXPECT(std::vector<float>(loaded.logits().begin(), loaded.logits().end()) ==
               std::vector<float>(model.logits().begin(), model.logits().end()));

        {
            std::ofstream bad(path, std::ios::binary | std::ios::trunc);
            bad << "BADMAGIC";
        }
        bool threw = false;
        try {
            loaded.load(path);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        EXPECT(threw);

        bool rejected = false;
        try {
            model.forward(std::vector<int>{99, 0});
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        EXPECT(rejected);
    }

    if (lodestone_failures() != 0) {
        std::fprintf(stderr, "%d failure(s)\n", lodestone_failures());
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
