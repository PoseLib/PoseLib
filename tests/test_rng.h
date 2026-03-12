#ifndef POSELIB_TEST_RNG_H_
#define POSELIB_TEST_RNG_H_

#include <Eigen/Dense>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>

namespace test_rng {

struct TestContext {
    uint64_t suite_seed = 1;
    size_t repetition = 0;
    std::string test_name;
};

inline TestContext &context() {
    static TestContext ctx;
    return ctx;
}

inline uint64_t splitmix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

inline uint64_t stable_string_hash(const std::string &text) {
    uint64_t hash = 1469598103934665603ULL;
    for (unsigned char c : text) {
        hash ^= static_cast<uint64_t>(c);
        hash *= 1099511628211ULL;
    }
    return hash;
}

inline uint64_t combine_seed(uint64_t lhs, uint64_t rhs) { return splitmix64(lhs ^ splitmix64(rhs)); }

inline void set_test_context(const std::string &test_name, uint64_t suite_seed, size_t repetition) {
    TestContext &ctx = context();
    ctx.suite_seed = suite_seed;
    ctx.repetition = repetition;
    ctx.test_name = test_name;
}

inline uint64_t case_seed(const std::string &case_name = "", size_t case_index = 0) {
    const TestContext &ctx = context();
    uint64_t seed = combine_seed(ctx.suite_seed, ctx.repetition + 1);
    seed = combine_seed(seed, stable_string_hash(ctx.test_name));
    seed = combine_seed(seed, stable_string_hash(case_name));
    seed = combine_seed(seed, case_index + 1);
    return seed;
}

inline unsigned int global_rand_seed() { return static_cast<unsigned int>(case_seed("__global__") & 0xffffffffULL); }

inline unsigned int fixed_seed(const std::string &label, size_t case_index = 0) {
    uint64_t seed = combine_seed(stable_string_hash(label), case_index + 1);
    return static_cast<unsigned int>(seed & 0xffffffffULL);
}

inline std::string case_id(const std::string &case_name = "", size_t case_index = 0) {
    std::ostringstream ss;
    ss << context().test_name;
    if (!case_name.empty()) {
        ss << "/" << case_name;
    }
    ss << "#" << case_index;
    return ss.str();
}

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
        std::uniform_int_distribution<size_t> dist(0, upper_exclusive - 1);
        return dist(engine_);
    }

  private:
    std::mt19937_64 engine_;
};

inline Rng make_rng(const std::string &case_name = "", size_t case_index = 0) {
    return Rng(case_seed(case_name, case_index));
}

inline Eigen::Vector2d symmetric_vec2(Rng &rng, double scale = 1.0) {
    return Eigen::Vector2d(rng.uniform(-scale, scale), rng.uniform(-scale, scale));
}

inline Eigen::Vector3d symmetric_vec3(Rng &rng, double scale = 1.0) {
    return Eigen::Vector3d(rng.uniform(-scale, scale), rng.uniform(-scale, scale), rng.uniform(-scale, scale));
}

} // namespace test_rng

#endif
