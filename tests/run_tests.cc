#include "test.h"
#include "test_rng.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <streambuf>
#include <vector>

// We need to define all test functions here
std::vector<Test> register_camera_models_test();
std::vector<Test> register_hybrid_ransac_test();
std::vector<Test> register_ransac_test();
std::vector<Test> register_optim_absolute_test();
std::vector<Test> register_optim_relative_test();
std::vector<Test> register_optim_fundamental_test();
std::vector<Test> register_optim_gen_absolute_test();
std::vector<Test> register_optim_gen_relative_test();
std::vector<Test> register_optim_homography_test();
std::vector<Test> register_optim_monodepth_relpose_test();
std::vector<Test> register_recalibrator_test();

namespace {

constexpr unsigned int kDefaultSeed = 1;

struct RunnerOptions {
    unsigned int seed = kDefaultSeed;
    size_t stress_repetitions = 1;
    std::vector<std::string> filter;
};

class ScopedStreamCapture {
  public:
    ScopedStreamCapture()
        : cout_buf(std::cout.rdbuf(cout_stream.rdbuf())), cerr_buf(std::cerr.rdbuf(cerr_stream.rdbuf())) {}

    ScopedStreamCapture(const ScopedStreamCapture &) = delete;
    ScopedStreamCapture &operator=(const ScopedStreamCapture &) = delete;

    ~ScopedStreamCapture() {
        std::cout.rdbuf(cout_buf);
        std::cerr.rdbuf(cerr_buf);
    }

    std::string str() const { return cout_stream.str() + cerr_stream.str(); }

  private:
    std::ostringstream cout_stream;
    std::ostringstream cerr_stream;
    std::streambuf *cout_buf;
    std::streambuf *cerr_buf;
};

std::string repetition_suffix(size_t repetition, unsigned int seed) {
    return " [stress=" + std::to_string(repetition + 1) + ", seed=" + std::to_string(seed) + "]";
}

} // namespace

bool filter_test(const std::string &name, const std::vector<std::string> &filter) {
    if (filter.size() == 0)
        return true;
    for (const std::string &f : filter) {
        if (name.find(f) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::pair<int, int> run_tests_impl(const std::vector<Test> &tests, const std::string &name, const RunnerOptions &opt,
                                   std::vector<std::string> &failed_tests) {
    std::vector<Test> filtered_tests;
    for (const Test &test : tests) {
        if (filter_test(test.second, opt.filter)) {
            filtered_tests.push_back(test);
        }
    }

    int passed = 0;
    int num_tests = filtered_tests.size();

    if (num_tests > 0) {
        std::cout << "\nRunning tests from " << name << std::endl;
        for (const Test &test : filtered_tests) {
            bool passed_all = true;
            size_t failed_repetition = 0;
            std::string failed_log;

            for (size_t repetition = 0; repetition < opt.stress_repetitions; ++repetition) {
                test_rng::set_test_context(test.second, opt.seed, repetition);
                std::srand(test_rng::global_rand_seed());
                bool passed_repetition = false;
                std::string captured_log;
                {
                    ScopedStreamCapture capture;
                    passed_repetition = (test.first)();
                    captured_log = capture.str();
                }
                if (!passed_repetition) {
                    passed_all = false;
                    failed_repetition = repetition;
                    failed_log = std::move(captured_log);
                    break;
                }
            }

            if (passed_all) {
                std::cout << test.second + "\033[1m\033[32m PASSED!\033[0m\n";
                passed++;
            } else {
                std::cout << test.second + "\033[1m\033[31m FAILED!\033[0m\n";
                std::cout << "  stress=" << (failed_repetition + 1) << ", seed=" << opt.seed << "\n";
                if (!failed_log.empty()) {
                    std::cout << failed_log;
                    if (failed_log.back() != '\n') {
                        std::cout << "\n";
                    }
                }
                failed_tests.push_back(test.second + repetition_suffix(failed_repetition, opt.seed));
            }
        }
        std::cout << "Done! Passed " << passed << "/" << num_tests << " tests.\n";
    }
    return std::make_pair(passed, num_tests);
}

#define RUN_TESTS(NAME)                                                                                                \
    do {                                                                                                               \
        std::pair<int, int> ret = run_tests_impl(register_##NAME(), #NAME, opt, failed_tests);                         \
        passed += ret.first;                                                                                           \
        num_tests += ret.second;                                                                                       \
    } while (0);

bool is_uint(const std::string &str) {
    try {
        unsigned long long val = std::stoull(str);
        return val <= std::numeric_limits<unsigned int>::max();
    } catch (...) {
        return false; // Conversion failed (e.g., overflow)
    }
}

RunnerOptions parse_options(int argc, char *argv[]) {
    RunnerOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg(argv[i]);
        if (arg == "--seed" && i + 1 < argc && is_uint(argv[i + 1])) {
            opt.seed = std::stoull(argv[++i]);
        } else if (arg.rfind("--seed=", 0) == 0 && is_uint(arg.substr(7))) {
            opt.seed = std::stoull(arg.substr(7));
        } else if (arg == "--stress" && i + 1 < argc && is_uint(argv[i + 1])) {
            opt.stress_repetitions = std::max<size_t>(1, std::stoull(argv[++i]));
        } else if (arg.rfind("--stress=", 0) == 0 && is_uint(arg.substr(9))) {
            opt.stress_repetitions = std::max<size_t>(1, std::stoull(arg.substr(9)));
        } else if (is_uint(arg)) {
            opt.seed = std::stoull(arg);
        } else {
            opt.filter.push_back(arg);
        }
    }
    return opt;
}

int main(int argc, char *argv[]) {
    const RunnerOptions opt = parse_options(argc, argv);
    std::vector<std::string> failed_tests;
    std::cout << "Running tests... (seed = " << opt.seed << ", stress = " << opt.stress_repetitions << ")\n\n";
    int passed = 0, num_tests = 0;

    RUN_TESTS(camera_models_test);
    RUN_TESTS(hybrid_ransac_test);
    RUN_TESTS(ransac_test);
    RUN_TESTS(optim_absolute_test);
    RUN_TESTS(optim_relative_test);
    RUN_TESTS(optim_fundamental_test);
    RUN_TESTS(optim_gen_absolute_test);
    RUN_TESTS(optim_gen_relative_test);
    RUN_TESTS(optim_homography_test);
    RUN_TESTS(optim_monodepth_relpose_test);
    RUN_TESTS(recalibrator_test);

    std::cout << "Test suite finished (" << passed << " / " << num_tests << " passed, seed = " << opt.seed
              << ", stress = " << opt.stress_repetitions << ")\n\n";
    if (failed_tests.size() > 0) {
        std::cout << "Failed tests:\n";
        for (const std::string &test_name : failed_tests) {
            std::cout << " " << test_name << "\033[1m\033[31m FAILED!\033[0m\n";
        }
    }

    return failed_tests.empty() ? 0 : 1;
}
