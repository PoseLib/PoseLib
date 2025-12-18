// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Templated hybrid RANSAC implementation with multiple data types and solvers.
// Follows PoseLib ransac_impl.h structure, adapting only:
//   - Sampling: adaptive solver selection for multiple minimal solvers
//   - Stopping criteria: per-type inlier ratios for termination

#pragma once

#include <PoseLib/camera_pose.h>
#include <PoseLib/robust/base_hybrid_estimator.h>
#include <PoseLib/types.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace poselib {

struct HybridRansacState {
    size_t best_minimal_inlier_count = 0;
    double best_minimal_msac_score = std::numeric_limits<double>::max();
    std::vector<size_t> dynamic_max_iter; // per solver
    double log_prob_missing_model;
    std::mt19937 rng;
};

// Compute required iterations for a specific solver based on per-type inlier
// ratios (follows PoseLib's dynamic iteration update pattern)
inline size_t compute_dynamic_max_iter(const std::vector<double> &inlier_ratios,
                                       const std::vector<size_t> &sample_sizes, // for this solver
                                       double log_prob_missing, double dyn_num_trials_mult, size_t min_iterations,
                                       size_t max_iterations) {
    // Probability that all samples are inliers
    double prob_all_inliers = 1.0;
    for (size_t t = 0; t < inlier_ratios.size(); ++t) {
        if (sample_sizes[t] > 0) {
            prob_all_inliers *= std::pow(inlier_ratios[t], static_cast<double>(sample_sizes[t]));
        }
    }

    // Handle edge cases (matches PoseLib's ransac_impl.h)
    if (prob_all_inliers >= 0.9999) {
        return min_iterations;
    }
    if (prob_all_inliers <= 0.0001) {
        return max_iterations;
    }

    double prob_outlier = 1.0 - prob_all_inliers;
    size_t num_iters = static_cast<size_t>(std::ceil(log_prob_missing / std::log(prob_outlier) * dyn_num_trials_mult));
    return std::max(min_iterations, std::min(max_iterations, num_iters));
}

// Adaptive solver selection (Camposeco et al.)
// Returns solver index, or -1 if no valid solver
template <typename HybridSolver>
int select_solver(const HybridSolver &estimator, const std::vector<double> &prior_probs, const HybridRansacStats &stats,
                  size_t min_iterations, HybridRansacState &state) {
    const size_t num_solvers = estimator.num_minimal_solvers();
    const size_t num_types = estimator.num_data_types();
    const auto sample_sizes = estimator.min_sample_sizes();

    std::vector<double> probs(num_solvers, 0.0);
    double sum_probs = 0.0;

    // Check if we have valid inlier info (i.e., we've scored at least one model)
    bool have_inlier_info = stats.model_score < std::numeric_limits<double>::max();

    if (!have_inlier_info) {
        // No valid model yet, use prior probabilities
        for (size_t i = 0; i < num_solvers; ++i) {
            probs[i] = prior_probs[i];
            sum_probs += probs[i];
        }
    } else {
        // Adaptive selection based on inlier ratios
        for (size_t i = 0; i < num_solvers; ++i) {
            if (prior_probs[i] <= 0.0)
                continue;

            double num_iters = static_cast<double>(stats.num_iterations_per_solver[i]);
            if (num_iters > 0)
                num_iters -= 1.0;

            double prob_all_inliers = 1.0;
            for (size_t t = 0; t < num_types; ++t) {
                prob_all_inliers *= std::pow(stats.inlier_ratios[t], static_cast<double>(sample_sizes[i][t]));
            }

            if (num_iters < static_cast<double>(min_iterations)) {
                probs[i] = prob_all_inliers * prior_probs[i];
            } else {
                probs[i] = prob_all_inliers * std::pow(1.0 - prob_all_inliers, num_iters) * prior_probs[i];
            }
            sum_probs += probs[i];
        }
    }

    if (sum_probs <= 0.0)
        return -1;

    std::uniform_real_distribution<double> dist(0.0, sum_probs);
    double r = dist(state.rng);
    double cumulative = 0.0;
    for (size_t i = 0; i < num_solvers; ++i) {
        cumulative += probs[i];
        if (r <= cumulative)
            return static_cast<int>(i);
    }
    return static_cast<int>(num_solvers - 1);
}

// Score models and refine best one (follows PoseLib's score_models pattern)
template <typename HybridSolver, typename Model>
void score_models(HybridSolver &estimator, const std::vector<Model> &models, int solver_idx,
                  const HybridRansacOptions &opt, HybridRansacState &state, HybridRansacStats &stats,
                  Model *best_model) {
    const size_t num_types = estimator.num_data_types();
    const auto num_data = estimator.num_data();
    const auto sample_sizes = estimator.min_sample_sizes();
    const size_t num_solvers = estimator.num_minimal_solvers();

    // Find best model among candidates
    int best_model_ind = -1;
    size_t inlier_count = 0;

    for (size_t i = 0; i < models.size(); ++i) {
        double score_msac = estimator.score_model(models[i], &inlier_count);

        bool more_inliers = inlier_count > state.best_minimal_inlier_count;
        bool better_score = score_msac < state.best_minimal_msac_score;

        if (more_inliers || better_score) {
            if (more_inliers) {
                state.best_minimal_inlier_count = inlier_count;
            }
            if (better_score) {
                state.best_minimal_msac_score = score_msac;
            }
            best_model_ind = static_cast<int>(i);

            // Check if we should update best model already
            if (score_msac < stats.model_score) {
                stats.model_score = score_msac;
                *best_model = models[i];
                stats.num_inliers = inlier_count;
                stats.best_solver_type = solver_idx;
            }
        }
    }

    if (best_model_ind == -1)
        return;

    // Refinement
    Model refined_model = models[best_model_ind];
    estimator.refine_model(&refined_model);
    stats.refinements++;

    // Score refined model
    double refined_score = estimator.score_model(refined_model, &inlier_count);

    if (refined_score < stats.model_score) {
        stats.model_score = refined_score;
        stats.num_inliers = inlier_count;
        *best_model = refined_model;
        stats.best_solver_type = solver_idx;
    }

    // Update inlier ratios from estimator
    stats.inlier_ratios = estimator.inlier_ratios();

    // Update overall inlier ratio (matches PoseLib pattern)
    size_t total_data = 0;
    for (size_t t = 0; t < num_types; ++t) {
        total_data += num_data[t];
    }
    if (total_data > 0) {
        stats.inlier_ratio = static_cast<double>(stats.num_inliers) / static_cast<double>(total_data);
    }

    // Update dynamic max iterations per solver
    for (size_t s = 0; s < num_solvers; ++s) {
        state.dynamic_max_iter[s] =
            compute_dynamic_max_iter(stats.inlier_ratios, sample_sizes[s], state.log_prob_missing_model,
                                     opt.dyn_num_trials_mult, opt.min_iterations, opt.max_iterations);
    }
}

// Templated hybrid RANSAC implementation (follows PoseLib ransac_impl.h)
template <typename HybridSolver, typename Model = CameraPose>
HybridRansacStats hybrid_ransac(HybridSolver &estimator, const HybridRansacOptions &opt, Model *best_model) {
    // Static assertion to ensure estimator conforms to hybrid interface
    static_assert(is_hybrid_ransac_estimator_v<HybridSolver>,
                  "HybridSolver must inherit from BaseHybridRansacEstimator<ModelType>");

    HybridRansacStats stats;
    const size_t num_solvers = estimator.num_minimal_solvers();
    const size_t num_types = estimator.num_data_types();
    const auto num_data = estimator.num_data();
    const auto sample_sizes = estimator.min_sample_sizes();
    auto prior_probs = estimator.solver_probabilities();

    // Initialize statistics
    stats.num_iterations_per_solver.resize(num_solvers, 0);
    stats.inlier_ratios.resize(num_types, 0.0);
    stats.num_inliers_per_type.resize(num_types, 0);
    stats.inlier_indices.resize(num_types);
    stats.model_score = std::numeric_limits<double>::max();

    // Verify data availability and update priors
    for (size_t i = 0; i < num_solvers; ++i) {
        for (size_t j = 0; j < num_types; ++j) {
            if (sample_sizes[i][j] > num_data[j]) {
                prior_probs[i] = 0.0;
                break;
            }
        }
    }

    bool any_valid = false;
    for (size_t i = 0; i < num_solvers; ++i) {
        if (prior_probs[i] > 0.0) {
            any_valid = true;
            break;
        }
    }
    if (!any_valid)
        return stats;

    // Initialize state
    HybridRansacState state;
    state.rng.seed(opt.seed);
    state.log_prob_missing_model = std::log(1.0 - opt.success_prob);
    state.dynamic_max_iter.resize(num_solvers, opt.max_iterations);

    std::vector<std::vector<size_t>> sample;
    std::vector<Model> models;

    // Main RANSAC loop (follows PoseLib structure)
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; ++stats.iterations) {
        // Adaptive solver selection
        int solver_idx = select_solver(estimator, prior_probs, stats, opt.min_iterations, state);
        if (solver_idx < 0)
            break;

        // Check termination (per-solver dynamic max iter)
        if (stats.iterations > opt.min_iterations &&
            stats.num_iterations_per_solver[solver_idx] >= state.dynamic_max_iter[solver_idx]) {
            // Try to find another valid solver
            bool found_valid = false;
            for (size_t s = 0; s < num_solvers; ++s) {
                if (prior_probs[s] > 0.0 && stats.num_iterations_per_solver[s] < state.dynamic_max_iter[s]) {
                    found_valid = true;
                    break;
                }
            }
            if (!found_valid)
                break;
            continue; // Skip this solver, try again
        }

        stats.num_iterations_per_solver[solver_idx]++;

        // Generate sample and models
        estimator.generate_sample(solver_idx, &sample);
        estimator.generate_models(sample, solver_idx, &models);

        if (models.empty())
            continue;

        // Score models and refine best one
        score_models(estimator, models, solver_idx, opt, state, stats, best_model);
    }

    // Final refinement (matches PoseLib pattern)
    if (stats.model_score < std::numeric_limits<double>::max()) {
        Model refined_model = *best_model;
        estimator.refine_model(&refined_model);
        stats.refinements++;

        size_t inlier_count;
        double refined_score = estimator.score_model(refined_model, &inlier_count);

        if (refined_score < stats.model_score) {
            stats.num_inliers = inlier_count;
            *best_model = refined_model;
        }

        // Get final inlier info from estimator
        stats.inlier_ratios = estimator.inlier_ratios();
        stats.inlier_indices = estimator.inlier_indices();
        for (size_t t = 0; t < num_types; ++t) {
            stats.num_inliers_per_type[t] = stats.inlier_indices[t].size();
        }
    }

    return stats;
}

} // namespace poselib
