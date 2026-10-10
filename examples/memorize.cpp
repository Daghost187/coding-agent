#include "memorize.hpp"

#include "lodestone/lodestone.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace lodestone {
namespace {

constexpr std::string_view kCorpus = "the lodestone keeps its poles";

struct Codec {
    std::string alphabet;

    explicit Codec(std::string_view text) {
        for (char ch : text) {
            if (alphabet.find(ch) == std::string::npos) {
                alphabet.push_back(ch);
            }
        }
    }

    [[nodiscard]] std::vector<int> encode(std::string_view text) const {
        std::vector<int> ids;
        ids.reserve(text.size());
        for (char ch : text) {
            const auto found = alphabet.find(ch);
            if (found == std::string::npos) {
                throw std::invalid_argument("codec: character is not in the alphabet");
            }
            ids.push_back(static_cast<int>(found));
        }
        return ids;
    }

    [[nodiscard]] std::string decode(const std::vector<int>& ids) const {
        std::string text;
        text.reserve(ids.size());
        for (int id : ids) {
            if (id < 0 || id >= static_cast<int>(alphabet.size())) {
                text.push_back('?');
            } else {
                text.push_back(alphabet[static_cast<std::size_t>(id)]);
            }
        }
        return text;
    }
};

void log_step(std::ostream& log, int step, float loss, float accuracy) {
    log << "step " << std::setw(4) << step
        << "  loss " << std::fixed << std::setprecision(4) << loss
        << "  acc " << std::setprecision(3) << accuracy << '\n';
}

}  // namespace

MemorizeReport run_memorize(const MemorizeConfig& config, std::ostream* log) {
    if (config.steps < 0) {
        throw std::invalid_argument("memorize: steps must be non-negative");
    }

    const Codec codec(kCorpus);
    const std::vector<int> ids = codec.encode(kCorpus);
    const int tokens = static_cast<int>(ids.size());

    Config model_cfg;
    model_cfg.vocab = static_cast<int>(codec.alphabet.size());
    model_cfg.d_model = 64;
    model_cfg.n_layers = 1;
    model_cfg.n_heads = 4;
    model_cfg.d_ff = 256;
    model_cfg.max_seq = 64;

    TrainConfig opt;
    opt.lr = config.lr;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.adam_eps = 1e-8f;
    opt.weight_decay = config.weight_decay;
    opt.grad_clip = config.grad_clip;
    opt.use_adam = config.use_adam;

    Transformer model(model_cfg, config.seed);
    MemorizeReport report;
    report.corpus = std::string(kCorpus);
    report.isa = isa_name(active_isa());
    report.params = model.parameter_count();

    model.forward(ids);
    report.initial_loss = cross_entropy(model.logits().data(), ids.data(), tokens, model_cfg.vocab, nullptr);
    const float initial_accuracy = model.accuracy(ids);

    if (log != nullptr) {
        *log << "lodestone  isa " << report.isa
             << "  params " << report.params
             << "  vocab " << model_cfg.vocab
             << "  corpus \"" << report.corpus << "\"\n";
        if (config.log_every > 0) {
            log_step(*log, 0, report.initial_loss, initial_accuracy);
        }
    }

    for (int step = 1; step <= config.steps; ++step) {
        const StepResult result = model.train_step(ids, opt);
        if (log != nullptr && config.log_every > 0 && step % config.log_every == 0) {
            log_step(*log, step, result.loss, result.accuracy);
        }
        // Adam keeps stepping after the signs are right and will eventually
        // flip them. Stop at the first greedy reproduction of the sentence.
        if (step % 25 == 0 || step == config.steps) {
            model.forward(ids);
            const std::vector<int> probe_prefix{ids.front()};
            if (codec.decode(model.greedy(probe_prefix, tokens)) == kCorpus) {
                if (log != nullptr) {
                    *log << "memorized at step " << step << '\n';
                }
                break;
            }
        }
    }

    model.forward(ids);
    report.final_loss = cross_entropy(model.logits().data(), ids.data(), tokens, model_cfg.vocab, nullptr);
    report.final_accuracy = model.accuracy(ids);
    const std::vector<int> prefix{ids.front()};
    report.generated = codec.decode(model.greedy(prefix, tokens));

    if (log != nullptr) {
        *log << "final loss " << std::fixed << std::setprecision(4) << report.final_loss
             << "  acc " << std::setprecision(3) << report.final_accuracy << '\n';
        *log << "generated: " << report.generated << '\n';
    }
    return report;
}

}  // namespace lodestone
