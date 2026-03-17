#include "test.h"

#include <PoseLib/robust/hybrid_ransac_impl.h>
#include <cmath>
#include <vector>

using namespace poselib;

namespace test::hybrid_ransac {

struct MockHybridEstimator {
    std::vector<size_t> num_data_values;
    std::vector<std::vector<size_t>> sample_sizes_values;
    std::vector<double> prior_probs;
    std::vector<std::vector<size_t>> inliers_per_solver;

    size_t num_data_types() const { return num_data_values.size(); }

    std::vector<size_t> num_data() const { return num_data_values; }

    size_t num_minimal_solvers() const { return sample_sizes_values.size(); }

    std::vector<std::vector<size_t>> min_sample_sizes() const { return sample_sizes_values; }

    std::vector<double> solver_probabilities() const { return prior_probs; }

    void generate_sample(size_t solver_idx, std::vector<std::vector<size_t>> *sample) const {
        sample->resize(num_data_values.size());
        for (size_t t = 0; t < num_data_values.size(); ++t) {
            (*sample)[t].resize(sample_sizes_values[solver_idx][t]);
            for (size_t i = 0; i < sample_sizes_values[solver_idx][t]; ++i) {
                (*sample)[t][i] = i;
            }
        }
    }

    void generate_models(const std::vector<std::vector<size_t>> &, size_t solver_idx, std::vector<int> *models) const {
        models->push_back(static_cast<int>(solver_idx));
    }

    double score_model(const int &model, std::vector<size_t> *inliers_per_type) const {
        *inliers_per_type = inliers_per_solver[model];
        return 0.0;
    }

    void refine_model(int *) const {}
};

bool test_hybrid_all_inlier_sample_probability_matches_hypergeometric() {
    const std::vector<size_t> num_inliers_per_type = {4, 3};
    const std::vector<size_t> num_data = {10, 8};
    const std::vector<size_t> sample_sizes = {2, 2};

    const double expected_prob = (4.0 / 10.0) * (3.0 / 9.0) * (3.0 / 8.0) * (2.0 / 7.0);
    const double actual_prob = detail::all_inlier_sample_probability(num_inliers_per_type, num_data, sample_sizes);
    const double approximate_prob = std::pow(4.0 / 10.0, 2.0) * std::pow(3.0 / 8.0, 2.0);

    REQUIRE_SMALL(actual_prob - expected_prob, 1e-12);
    REQUIRE(std::abs(actual_prob - approximate_prob) > 1e-3);
    return true;
}

bool test_hybrid_dynamic_iterations_use_exact_probability() {
    MockHybridEstimator estimator{{10, 8}, {{2, 2}}, {1.0}, {{4, 3}}};
    HybridRansacOptions opt;
    opt.min_iterations = 0;
    opt.max_iterations = 100;
    opt.dyn_num_trials_mult = 1.0;
    opt.success_prob = 0.5;

    const double prob_all_inliers = (4.0 / 10.0) * (3.0 / 9.0) * (3.0 / 8.0) * (2.0 / 7.0);
    const size_t expected_dynamic_max_iter = static_cast<size_t>(
        std::ceil(std::log(1.0 - opt.success_prob) / std::log(1.0 - prob_all_inliers) * opt.dyn_num_trials_mult));

    REQUIRE_EQ(detail::compute_dynamic_max_iter(estimator.inliers_per_solver[0], estimator.num_data_values,
                                                estimator.sample_sizes_values[0], std::log(1.0 - opt.success_prob),
                                                opt.dyn_num_trials_mult, opt.min_iterations, opt.max_iterations),
               expected_dynamic_max_iter);

    int best_model = -1;
    const HybridRansacStats stats = poselib::hybrid_ransac<MockHybridEstimator, int>(estimator, opt, &best_model);

    REQUIRE_EQ(stats.num_inliers_per_type[0], estimator.inliers_per_solver[0][0]);
    REQUIRE_EQ(stats.num_inliers_per_type[1], estimator.inliers_per_solver[0][1]);
    REQUIRE_EQ(stats.num_iterations_per_solver[0], expected_dynamic_max_iter);
    REQUIRE_EQ(stats.iterations, expected_dynamic_max_iter);
    return true;
}

bool test_hybrid_dynamic_iterations_stay_at_max_when_not_enough_inliers() {
    MockHybridEstimator estimator{{10, 8}, {{2, 2}}, {1.0}, {{1, 3}}};
    HybridRansacOptions opt;
    opt.min_iterations = 0;
    opt.max_iterations = 20;
    opt.dyn_num_trials_mult = 1.0;
    opt.success_prob = 0.5;

    REQUIRE_EQ(detail::all_inlier_sample_probability(estimator.inliers_per_solver[0], estimator.num_data_values,
                                                     estimator.sample_sizes_values[0]),
               0.0);
    REQUIRE_EQ(detail::compute_dynamic_max_iter(estimator.inliers_per_solver[0], estimator.num_data_values,
                                                estimator.sample_sizes_values[0], std::log(1.0 - opt.success_prob),
                                                opt.dyn_num_trials_mult, opt.min_iterations, opt.max_iterations),
               opt.max_iterations);

    int best_model = -1;
    const HybridRansacStats stats = poselib::hybrid_ransac<MockHybridEstimator, int>(estimator, opt, &best_model);

    REQUIRE_EQ(stats.num_iterations_per_solver[0], opt.max_iterations);
    REQUIRE_EQ(stats.iterations, opt.max_iterations);
    return true;
}

bool test_hybrid_dynamic_iterations_collapse_to_min_for_all_inliers() {
    MockHybridEstimator estimator{{10, 8}, {{2, 2}}, {1.0}, {{10, 8}}};
    HybridRansacOptions opt;
    opt.min_iterations = 3;
    opt.max_iterations = 100;
    opt.dyn_num_trials_mult = 1.0;
    opt.success_prob = 0.5;

    REQUIRE_EQ(detail::all_inlier_sample_probability(estimator.inliers_per_solver[0], estimator.num_data_values,
                                                     estimator.sample_sizes_values[0]),
               1.0);
    REQUIRE_EQ(detail::compute_dynamic_max_iter(estimator.inliers_per_solver[0], estimator.num_data_values,
                                                estimator.sample_sizes_values[0], std::log(1.0 - opt.success_prob),
                                                opt.dyn_num_trials_mult, opt.min_iterations, opt.max_iterations),
               opt.min_iterations);

    int best_model = -1;
    const HybridRansacStats stats = poselib::hybrid_ransac<MockHybridEstimator, int>(estimator, opt, &best_model);

    REQUIRE_EQ(stats.num_iterations_per_solver[0], opt.min_iterations);
    REQUIRE_EQ(stats.iterations, opt.min_iterations);
    return true;
}

bool test_hybrid_solver_selection_weights_use_exact_probability() {
    const std::vector<size_t> num_inliers_per_type = {4, 3};
    const std::vector<size_t> num_data = {10, 8};
    const std::vector<size_t> solver_a_sample_sizes = {2, 2};
    const std::vector<size_t> solver_b_sample_sizes = {4, 0};

    const double expected_weight_a = (4.0 / 10.0) * (3.0 / 9.0) * (3.0 / 8.0) * (2.0 / 7.0);
    const double expected_weight_b = (4.0 / 10.0) * (3.0 / 9.0) * (2.0 / 8.0) * (1.0 / 7.0);
    const double approximate_weight_a = std::pow(4.0 / 10.0, 2.0) * std::pow(3.0 / 8.0, 2.0);
    const double approximate_weight_b = std::pow(4.0 / 10.0, 4.0);

    const double actual_weight_a =
        detail::compute_solver_selection_weight(num_inliers_per_type, num_data, solver_a_sample_sizes, 1.0, 0, 0);
    const double actual_weight_b =
        detail::compute_solver_selection_weight(num_inliers_per_type, num_data, solver_b_sample_sizes, 1.0, 0, 0);

    REQUIRE_SMALL(actual_weight_a - expected_weight_a, 1e-12);
    REQUIRE_SMALL(actual_weight_b - expected_weight_b, 1e-12);
    REQUIRE(actual_weight_a > actual_weight_b);
    REQUIRE(approximate_weight_a < approximate_weight_b);
    return true;
}

} // namespace test::hybrid_ransac

using namespace test::hybrid_ransac;
std::vector<Test> register_hybrid_ransac_test() {
    return {TEST(test_hybrid_all_inlier_sample_probability_matches_hypergeometric),
            TEST(test_hybrid_dynamic_iterations_use_exact_probability),
            TEST(test_hybrid_dynamic_iterations_stay_at_max_when_not_enough_inliers),
            TEST(test_hybrid_dynamic_iterations_collapse_to_min_for_all_inliers),
            TEST(test_hybrid_solver_selection_weights_use_exact_probability)};
}
