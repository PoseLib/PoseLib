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
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace poselib {

namespace detail {

inline double all_inlier_sample_probability(size_t num_inliers, size_t num_data, size_t sample_sz) {
    if (sample_sz == 0) {
        return 1.0;
    }
    if (num_inliers < sample_sz || num_data < sample_sz) {
        return 0.0;
    }

    double prob_all_inliers = 1.0;
    for (size_t i = 0; i < sample_sz; ++i) {
        prob_all_inliers *= static_cast<double>(num_inliers - i) / static_cast<double>(num_data - i);
    }
    return prob_all_inliers;
}

inline size_t compute_dynamic_max_iter(size_t num_inliers, size_t num_data, size_t sample_sz,
                                       double log_prob_missing_model, double dyn_num_trials_mult, size_t min_iterations,
                                       size_t max_iterations) {
    const double prob_all_inliers = all_inlier_sample_probability(num_inliers, num_data, sample_sz);
    if (prob_all_inliers >= 0.9999) {
        return min_iterations;
    }
    if (prob_all_inliers <= 0.0001) {
        return max_iterations;
    }

    const double prob_outlier = 1.0 - prob_all_inliers;
    const size_t num_iters =
        static_cast<size_t>(std::ceil(log_prob_missing_model / std::log(prob_outlier) * dyn_num_trials_mult));
    return std::max(min_iterations, std::min(max_iterations, num_iters));
}

} // namespace detail

// Example estimator for use with ransac():
//
//   class MyEstimator {
//     public:
//       // Required public members:
//       size_t sample_sz;  // Number of samples for minimal solver
//       size_t num_data;   // Total number of data points
//
//       // Required methods:
//
//       // Generates model hypotheses from a random minimal sample
//       void generate_models(std::vector<MyModel> *models);
//
//       // Computes MSAC score for model, returns score and inlier count
//       double score_model(const MyModel &model, size_t *inlier_count) const;
//
//       // Refines model using all inliers (e.g., bundle adjustment)
//       void refine_model(MyModel *model) const;
//   };
//
// See estimators/absolute_pose.h for a complete implementation.

struct RansacState {
    size_t best_minimal_inlier_count = 0;
    double best_minimal_msac_score = std::numeric_limits<double>::max();
    size_t dynamic_max_iter = 100000;
    double log_prob_missing_model = std::log(1.0 - 0.9999);
};

template <typename Solver, typename Model = CameraPose>
void score_models(const Solver &estimator, const std::vector<Model> &models, const RansacOptions &opt,
                  RansacState &state, RansacStats &stats, Model *best_model) {
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
            best_model_ind = i;

            // check if we should update best model already
            if (score_msac < stats.model_score) {
                stats.model_score = score_msac;
                *best_model = models[i];
                stats.num_inliers = inlier_count;
            }
        }
    }

    if (best_model_ind == -1)
        return;

    // Refinement
    Model refined_model = models[best_model_ind];
    estimator.refine_model(&refined_model);
    stats.refinements++;
    double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
    if (refined_msac_score < stats.model_score) {
        stats.model_score = refined_msac_score;
        stats.num_inliers = inlier_count;
        *best_model = refined_model;
    }

    // update number of iterations
    stats.inlier_ratio = static_cast<double>(stats.num_inliers) / static_cast<double>(estimator.num_data);
    state.dynamic_max_iter = detail::compute_dynamic_max_iter(
        stats.num_inliers, estimator.num_data, estimator.sample_sz, state.log_prob_missing_model,
        opt.dyn_num_trials_mult, opt.min_iterations, opt.max_iterations);
}

// Templated LO-RANSAC implementation (inspired by RansacLib from Torsten Sattler)
template <typename Solver, typename Model = CameraPose>
RansacStats ransac(Solver &estimator, const RansacOptions &opt, Model *best_model) {
    RansacStats stats;

    if (estimator.num_data < estimator.sample_sz) {
        return stats;
    }

    // Score/Inliers for best model found so far
    stats.num_inliers = 0;
    stats.model_score = std::numeric_limits<double>::max();
    // best inl/score for minimal model, used to decide when to LO
    RansacState state;
    state.dynamic_max_iter = opt.max_iterations;
    state.log_prob_missing_model = std::log(1.0 - opt.success_prob);

    // Score initial model if it was supplied
    if (opt.score_initial_model) {
        score_models(estimator, {*best_model}, opt, state, stats, best_model);
    }

    size_t inlier_count = 0;
    std::vector<Model> models;
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; stats.iterations++) {

        if (stats.iterations > opt.min_iterations && stats.iterations > state.dynamic_max_iter) {
            break;
        }
        models.clear();
        estimator.generate_models(&models);
        score_models(estimator, models, opt, state, stats, best_model);
    }

    // Final refinement
    Model refined_model = *best_model;
    estimator.refine_model(&refined_model);
    stats.refinements++;
    double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
    if (refined_msac_score < stats.model_score) {
        *best_model = refined_model;
        stats.num_inliers = inlier_count;
    }

    return stats;
}

} // namespace poselib
