#include <iostream>
#include <vector>
#include "test.h"

// We need to define all test functions here
std::vector<Test> register_camera_models_test();
std::vector<Test> register_optim_absolute_test();
std::vector<Test> register_optim_relative_test();
std::vector<Test> register_optim_fundamental_test();





std::pair<int,int> run_tests_impl(const std::vector<Test> &tests, const std::string &name) {
    std::cout << "\nRunning tests from " << name << std::endl;

    int passed = 0;
    int num_tests = tests.size();

    for(const Test test : tests) {
        if((test.first)()) {
            std::cout << test.second + "\033[1m\033[32m PASSED!\033[0m\n";
            passed++;
        } else {
             std::cout << test.second + "\033[1m\033[31m FAILED!\033[0m\n";
        }
    }

    std::cout << "Done! Passed " << passed << "/" << num_tests << " tests.\n";
    return std::make_pair(passed, num_tests);
}
#define RUN_TESTS(NAME) do { std::pair<int,int> ret = run_tests_impl(register_##NAME(), #NAME); passed += ret.first; num_tests += ret.second; } while(0);


int main() {
	
	unsigned int seed = (unsigned int)time(0);		
	srand(seed);
	std::cout << "Running tests... (seed = " << seed << ")\n\n";
    int passed = 0, num_tests = 0;
    
    RUN_TESTS(camera_models_test);
    RUN_TESTS(optim_absolute_test);
    RUN_TESTS(optim_relative_test);
    RUN_TESTS(optim_fundamental_test);

    std::cout << "Test suite finished (" << passed << " / " << num_tests << " passed, seed = "<< seed << ")\n\n";
}