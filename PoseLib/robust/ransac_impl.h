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

#ifndef POSELIB_RANSAC_IMPL_H_
#define POSELIB_RANSAC_IMPL_H_

#include "PoseLib/types.h"

#include <vector>

namespace poselib {

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
    size_t best_minimal_inlier_count = 0;
    double best_minimal_msac_score = std::numeric_limits<double>::max();

    const double log_prob_missing_model = std::log(1.0 - opt.success_prob);
    size_t inlier_count = 0;
    std::vector<Model> models;
    size_t dynamic_max_iter = opt.max_iterations;
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; stats.iterations++) {

        if (stats.iterations > opt.min_iterations && stats.iterations > dynamic_max_iter) {
            break;
        }
        models.clear();
        estimator.generate_models(&models);

        // Find best model among candidates
        int best_model_ind = -1;
        for (size_t i = 0; i < models.size(); ++i) {
            double score_msac = estimator.score_model(models[i], &inlier_count);
            bool more_inliers = inlier_count > best_minimal_inlier_count;
            bool better_score = score_msac < best_minimal_msac_score;

            if (more_inliers || better_score) {
                if (more_inliers) {
                    best_minimal_inlier_count = inlier_count;
                }
                if (better_score) {
                    best_minimal_msac_score = score_msac;
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
            continue;

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
        if (stats.inlier_ratio >= 0.9999) {
            // this is to avoid log(prob_outlier) = -inf below
            dynamic_max_iter = opt.min_iterations;
        } else if (stats.inlier_ratio <= 0.0001) {
            // this is to avoid log(prob_outlier) = 0 below
            dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(stats.inlier_ratio, estimator.sample_sz);
            dynamic_max_iter = std::ceil(log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }
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

#endif