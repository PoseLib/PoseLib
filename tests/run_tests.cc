#include "test.h"

#include <iostream>
#include <vector>

// We need to define all test functions here
std::vector<Test> register_camera_models_test();
std::vector<Test> register_optim_absolute_test();
std::vector<Test> register_optim_relative_test();
std::vector<Test> register_optim_fundamental_test();
std::vector<Test> register_optim_gen_absolute_test();
std::vector<Test> register_optim_gen_relative_test();
std::vector<Test> register_optim_homography_test();
std::vector<Test> register_recalibrator_test();

bool filter_test(const std::string &name, const std::vector<std::string> filter) {
    if (filter.size() == 0)
        return true;
    for (const std::string &f : filter) {
        if (name.find(f) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::pair<int, int> run_tests_impl(const std::vector<Test> &tests, const std::string &name,
                                   const std::vector<std::string> &filter, std::vector<std::string> &failed_tests) {
    std::vector<Test> filtered_tests;
    for (const Test &test : tests) {
        if (filter_test(test.second, filter)) {
            filtered_tests.push_back(test);
        }
    }

    int passed = 0;
    int num_tests = filtered_tests.size();

    if (num_tests > 0) {
        std::cout << "\nRunning tests from " << name << std::endl;
        for (const Test &test : filtered_tests) {
            if ((test.first)()) {
                std::cout << test.second + "\033[1m\033[32m PASSED!\033[0m\n";
                passed++;
            } else {
                std::cout << test.second + "\033[1m\033[31m FAILED!\033[0m\n";
                failed_tests.push_back(test.second);
            }
        }
        std::cout << "Done! Passed " << passed << "/" << num_tests << " tests.\n";
    }
    return std::make_pair(passed, num_tests);
}

#define RUN_TESTS(NAME)                                                                                                \
    do {                                                                                                               \
        std::pair<int, int> ret = run_tests_impl(register_##NAME(), #NAME, filter, failed_tests);                      \
        passed += ret.first;                                                                                           \
        num_tests += ret.second;                                                                                       \
    } while (0);

int main(int argc, char *argv[]) {
    std::vector<std::string> filter;
    for (int i = 1; i < argc; ++i) {
        filter.push_back(std::string(argv[i]));
    }

    std::vector<std::string> failed_tests;
    unsigned int seed = (unsigned int)time(0);
    srand(seed);
    std::cout << "Running tests... (seed = " << seed << ")\n\n";
    int passed = 0, num_tests = 0;

    RUN_TESTS(camera_models_test);
    RUN_TESTS(optim_absolute_test);
    RUN_TESTS(optim_relative_test);
    RUN_TESTS(optim_fundamental_test);
    RUN_TESTS(optim_gen_absolute_test);
    RUN_TESTS(optim_gen_relative_test);
    RUN_TESTS(optim_homography_test);
    RUN_TESTS(recalibrator_test);

    std::cout << "Test suite finished (" << passed << " / " << num_tests << " passed, seed = " << seed << ")\n\n";
    if (failed_tests.size() > 0) {
        std::cout << "Failed tests:\n";
        for (const std::string &test_name : failed_tests) {
            std::cout << " " << test_name << "\033[1m\033[31m FAILED!\033[0m\n";
        }
    }
}