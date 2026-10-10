#include "memorize.hpp"

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    lodestone::MemorizeConfig config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto value = [&](const char* flag) {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << flag << '\n';
                std::exit(2);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--steps") {
            config.steps = std::stoi(value("--steps"));
        } else if (arg == "--seed") {
            config.seed = static_cast<unsigned>(std::stoul(value("--seed")));
        } else if (arg == "--log-every") {
            config.log_every = std::stoi(value("--log-every"));
        } else if (arg == "--help") {
            std::cout << "usage: lodestone [--steps N] [--seed N] [--log-every N]\n"
                      << "Train a 1-bit transformer to memorize one sentence, then decode it.\n";
            return 0;
        } else {
            std::cerr << "unknown argument " << arg << '\n';
            return 2;
        }
    }

    const lodestone::MemorizeReport report = lodestone::run_memorize(config, &std::cout);
    return report.generated == report.corpus ? 0 : 1;
}
