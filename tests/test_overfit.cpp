#include "harness.hpp"
#include "memorize.hpp"

#include <iostream>
#include <sstream>

int main() {
    lodestone::MemorizeConfig config;
    config.steps = 800;
    config.seed = 1;
    config.log_every = 100;
    config.lr = 0.001f;
    config.weight_decay = 0.f;
    config.use_adam = true;
    std::ostringstream log;
    const lodestone::MemorizeReport report = lodestone::run_memorize(config, &log);

    EXPECT(report.final_loss < report.initial_loss * 0.25f);
    EXPECT(report.final_accuracy >= 0.99f);
    EXPECT(report.generated == report.corpus);
    if (lodestone_failures() != 0) {
        std::cerr << log.str();
        std::fprintf(stderr, "%d failure(s)\n", lodestone_failures());
        return 1;
    }
    std::printf("ok loss %.4f -> %.4f  acc %.3f\n", report.initial_loss, report.final_loss, report.final_accuracy);
    return 0;
}
