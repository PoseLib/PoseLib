#include "test.h"

#include <PoseLib/robust/ransac_impl.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace poselib;

namespace test::ransac {

struct MockEstimator {
    MockEstimator(size_t num_data_, size_t sample_sz_, size_t inlier_count_)
        : sample_sz(sample_sz_), num_data(num_data_), inlier_count(inlier_count_) {}

    void generate_models(std::vector<int> *models) const { models->push_back(0); }

    double score_model(const int &, size_t *model_inlier_count) const {
        *model_inlier_count = inlier_count;
        return 0.0;
    }

    void refine_model(int *) const {}

    size_t sample_sz;
    size_t num_data;
    size_t inlier_count;
};

size_t expected_iterations_from_dynamic_limit(size_t dynamic_max_iter, const RansacOptions &opt) {
    const size_t stop_after = std::max(opt.min_iterations, dynamic_max_iter);
    if (stop_after >= opt.max_iterations) {
        return opt.max_iterations;
    }
    return stop_after + 1;
}

bool test_all_inlier_sample_probability_matches_hypergeometric() {
    const size_t num_inliers = 5;
    const size_t num_data = 10;
    const size_t sample_sz = 5;

    const double expected_prob = (5.0 / 10.0) * (4.0 / 9.0) * (3.0 / 8.0) * (2.0 / 7.0) * (1.0 / 6.0);
    const double actual_prob = detail::all_inlier_sample_probability(num_inliers, num_data, sample_sz);
    const double approximate_prob =
        std::pow(static_cast<double>(num_inliers) / static_cast<double>(num_data), sample_sz);

    REQUIRE_SMALL(actual_prob - expected_prob, 1e-12);
    REQUIRE(std::abs(actual_prob - approximate_prob) > 1e-3);
    return true;
}

bool test_dynamic_iterations_use_exact_probability() {
    MockEstimator estimator(10, 5, 5);
    RansacOptions opt;
    opt.min_iterations = 0;
    opt.max_iterations = 1000;
    opt.dyn_num_trials_mult = 1.0;
    opt.success_prob = 0.5;

    const double prob_all_inliers = (5.0 / 10.0) * (4.0 / 9.0) * (3.0 / 8.0) * (2.0 / 7.0) * (1.0 / 6.0);
    const size_t expected_dynamic_max_iter = static_cast<size_t>(
        std::ceil(std::log(1.0 - opt.success_prob) / std::log(1.0 - prob_all_inliers) * opt.dyn_num_trials_mult));

    REQUIRE_EQ(detail::compute_dynamic_max_iter(estimator.inlier_count, estimator.num_data, estimator.sample_sz,
                                                std::log(1.0 - opt.success_prob), opt.dyn_num_trials_mult,
                                                opt.min_iterations, opt.max_iterations),
               expected_dynamic_max_iter);

    int best_model = -1;
    const RansacStats stats = poselib::ransac<MockEstimator, int>(estimator, opt, &best_model);

    REQUIRE_EQ(stats.num_inliers, estimator.inlier_count);
    REQUIRE_EQ(stats.iterations, expected_iterations_from_dynamic_limit(expected_dynamic_max_iter, opt));
    return true;
}

bool test_dynamic_iterations_stay_at_max_when_not_enough_inliers() {
    MockEstimator estimator(10, 5, 4);
    RansacOptions opt;
    opt.min_iterations = 0;
    opt.max_iterations = 20;
    opt.dyn_num_trials_mult = 1.0;
    opt.success_prob = 0.5;

    REQUIRE_EQ(detail::all_inlier_sample_probability(estimator.inlier_count, estimator.num_data, estimator.sample_sz),
               0.0);
    REQUIRE_EQ(detail::compute_dynamic_max_iter(estimator.inlier_count, estimator.num_data, estimator.sample_sz,
                                                std::log(1.0 - opt.success_prob), opt.dyn_num_trials_mult,
                                                opt.min_iterations, opt.max_iterations),
               opt.max_iterations);

    int best_model = -1;
    const RansacStats stats = poselib::ransac<MockEstimator, int>(estimator, opt, &best_model);

    REQUIRE_EQ(stats.num_inliers, estimator.inlier_count);
    REQUIRE_EQ(stats.iterations, opt.max_iterations);
    return true;
}

bool test_dynamic_iterations_collapse_to_min_for_all_inliers() {
    MockEstimator estimator(8, 5, 8);
    RansacOptions opt;
    opt.min_iterations = 3;
    opt.max_iterations = 100;
    opt.dyn_num_trials_mult = 1.0;
    opt.success_prob = 0.5;

    REQUIRE_EQ(detail::all_inlier_sample_probability(estimator.inlier_count, estimator.num_data, estimator.sample_sz),
               1.0);
    REQUIRE_EQ(detail::compute_dynamic_max_iter(estimator.inlier_count, estimator.num_data, estimator.sample_sz,
                                                std::log(1.0 - opt.success_prob), opt.dyn_num_trials_mult,
                                                opt.min_iterations, opt.max_iterations),
               opt.min_iterations);

    int best_model = -1;
    const RansacStats stats = poselib::ransac<MockEstimator, int>(estimator, opt, &best_model);

    REQUIRE_EQ(stats.num_inliers, estimator.inlier_count);
    REQUIRE_EQ(stats.iterations, expected_iterations_from_dynamic_limit(opt.min_iterations, opt));
    return true;
}

} // namespace test::ransac

using namespace test::ransac;
std::vector<Test> register_ransac_test() {
    return {TEST(test_all_inlier_sample_probability_matches_hypergeometric),
            TEST(test_dynamic_iterations_use_exact_probability),
            TEST(test_dynamic_iterations_stay_at_max_when_not_enough_inliers),
            TEST(test_dynamic_iterations_collapse_to_min_for_all_inliers)};
}
