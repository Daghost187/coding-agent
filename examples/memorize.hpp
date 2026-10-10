#pragma once

#include <iosfwd>
#include <string>

namespace lodestone {

struct MemorizeConfig {
    int steps = 800;
    unsigned seed = 1;
    int log_every = 50;
    float lr = 0.001f;
    float weight_decay = 0.f;
    float grad_clip = 1.f;
    bool use_adam = true;
};

struct MemorizeReport {
    float initial_loss = 0.f;
    float final_loss = 0.f;
    float final_accuracy = 0.f;
    std::string corpus;
    std::string generated;
    std::string isa;
    std::size_t params = 0;
};

// Fit a 1-bit transformer to one short sentence and greedily decode it.
// The same entry point backs the `lodestone` binary and the memorization test.
MemorizeReport run_memorize(const MemorizeConfig& config, std::ostream* log);

}  // namespace lodestone
