#ifndef POSELIB_TEST_RNG_H_
#define POSELIB_TEST_RNG_H_

// Deterministic, reproducible RNG for test fixtures.
//
// Design goals:
//  - Every random draw in a test is seeded from a stable hash of (suite_seed,
//    test_name, case_name, case_index, repetition), so tests are reproducible
//    given the same --seed value and never interfere with each other.
//  - The suite_seed is the only value the user needs to reproduce a failure:
//    run with --seed=<suite_seed> [--stress=<repetition>].
//
// Usage pattern:
//   test_rng::Rng rng = test_rng::make_rng("my_fixture_name");
//   double x = rng.uniform(0.0, 1.0);
//
// NOTE: TestContext is a process-global singleton. The test runner sets it
// before each test via set_test_context(). Tests are run sequentially; the
// design is NOT thread-safe.

#include <Eigen/Dense>
#include <cassert>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>

namespace test_rng {

// Global context set by the test runner before each test function is called.
struct TestContext {
    uint64_t suite_seed = 1;
    size_t repetition = 0;
    std::string test_name;
};

inline TestContext &context() {
    static TestContext ctx;
    return ctx;
}

// splitmix64 finalizer — used to avalanche bits when combining seeds.
inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// FNV-1a hash over a UTF-8 string. Produces a stable, portable 64-bit value.
inline uint64_t stable_string_hash(const std::string &text) {
    uint64_t hash = 14695981039346656037ULL;
    for (unsigned char c : text) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t combine_seed(uint64_t lhs, uint64_t rhs) { return splitmix64(lhs ^ splitmix64(rhs)); }

// Called by the test runner once per (test, repetition) before invoking the test function.
inline void set_test_context(const std::string &test_name, uint64_t suite_seed, size_t repetition) {
    TestContext &ctx = context();
    ctx.suite_seed = suite_seed;
    ctx.repetition = repetition;
    ctx.test_name = test_name;
}

// Derive a deterministic 64-bit seed for a named sub-case within the current test.
// case_name and case_index let different fixtures or loop iterations within the
// same test function get independent (non-correlated) RNG streams.
inline uint64_t case_seed(const std::string &case_name = "", size_t case_index = 0) {
    const TestContext &ctx = context();
    uint64_t seed = combine_seed(ctx.suite_seed, ctx.repetition + 1);
    seed = combine_seed(seed, stable_string_hash(ctx.test_name));
    seed = combine_seed(seed, stable_string_hash(case_name));
    seed = combine_seed(seed, case_index + 1);
    return seed;
}

// Returns a 32-bit seed suitable for std::srand(), derived from the current test
// context. Used by the runner to seed the global C RNG so that any remaining
// Eigen::setRandom() calls are also deterministic.
inline unsigned int global_rand_seed() { return static_cast<unsigned int>(case_seed("__global__") & 0xffffffffULL); }

// Returns a human-readable identifier for a sub-case, e.g. "test_foo/bar#2".
// Used in failure messages so it's clear which loop iteration failed.
inline std::string case_id(const std::string &case_name = "", size_t case_index = 0) {
    std::ostringstream ss;
    ss << context().test_name;
    if (!case_name.empty()) {
        ss << "/" << case_name;
    }
    ss << "#" << case_index;
    return ss.str();
}

// Thin wrapper around mt19937_64 with convenience sampling methods.
class Rng {
  public:
    explicit Rng(uint64_t seed) : engine_(seed) {}

    double uniform(double lower, double upper) {
        std::uniform_real_distribution<double> dist(lower, upper);
        return dist(engine_);
    }

    int uniform_int(int lower, int upper) {
        std::uniform_int_distribution<int> dist(lower, upper);
        return dist(engine_);
    }

    size_t uniform_index(size_t upper_exclusive) {
        assert(upper_exclusive > 0);
        std::uniform_int_distribution<size_t> dist(0, upper_exclusive - 1);
        return dist(engine_);
    }

  private:
    std::mt19937_64 engine_;
};

// Convenience: create an Rng seeded from the current test context + case_name/index.
inline Rng make_rng(const std::string &case_name = "", size_t case_index = 0) {
    return Rng(case_seed(case_name, case_index));
}

// Sample a 2D vector with each component drawn uniformly from [-scale, scale].
inline Eigen::Vector2d symmetric_vec2(Rng &rng, double scale = 1.0) {
    return Eigen::Vector2d(rng.uniform(-scale, scale), rng.uniform(-scale, scale));
}

// Sample a 3D vector with each component drawn uniformly from [-scale, scale].
inline Eigen::Vector3d symmetric_vec3(Rng &rng, double scale = 1.0) {
    return Eigen::Vector3d(rng.uniform(-scale, scale), rng.uniform(-scale, scale), rng.uniform(-scale, scale));
}

} // namespace test_rng

#endif
