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

#ifndef POSELIB_ROBUST_TYPES_H_
#define POSELIB_ROBUST_TYPES_H_

#include "alignment.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

struct RansacOptions {
    size_t max_iterations = 100000;
    size_t min_iterations = 1000;
    double dyn_num_trials_mult = 3.0;
    double success_prob = 0.9999;
    unsigned long seed = 0;
    // If we should use PROSAC sampling. Assumes data is sorted
    bool progressive_sampling = false;
    size_t max_prosac_iterations = 100000;
    // Whether to treat the input 'best_model' as an initial model and score it before running the main RANSAC loop
    bool score_initial_model = false;
};

struct RansacStats {
    size_t refinements = 0;
    size_t iterations = 0;
    size_t num_inliers = 0;
    double inlier_ratio = 0;
    double model_score = std::numeric_limits<double>::max();
};

struct BundleOptions {
    size_t max_iterations = 100;
    enum LossType {
        TRIVIAL,
        TRUNCATED,
        HUBER,
        CAUCHY,
        // This is the TR-IRLS scheme from Le and Zach, 3DV 2021
        TRUNCATED_LE_ZACH
    } loss_type = LossType::CAUCHY;
    double loss_scale = 1.0;
    double gradient_tol = 1e-12;
    double step_tol = 1e-8;
    double initial_lambda = 1e-3;
    double min_lambda = 1e-10;
    double max_lambda = 1e10;
    bool verbose = false;

    bool refine_focal_length = false;
    bool refine_extra_params = false;
    bool refine_principal_point = false;
};

struct BundleStats {
    size_t iterations = 0;
    double initial_cost;
    double cost;
    double lambda;
    size_t invalid_steps;
    double step_norm;
    double grad_norm;
};

// Options for robust estimators
struct AbsolutePoseOptions {
    RansacOptions ransac;
    BundleOptions bundle;

    double max_error = 12.0;
    // For problems with multiple types of residuals, we can have different max errors for each type
    // If not set, max_error is used for all residuals
    std::vector<double> max_errors = {};

    // Only applicable for pure PnP problems (central, 2D-3D points only)
    bool estimate_focal_length = false;
    bool estimate_extra_params = false;

    // Minimum (effective) field-of-view to accept when estimating focal length
    // in degrees. Effective means based on the image points supplied
    // and not on the actual image size.
    // Setting to 0 (or negative) disables checking.
    double min_fov = 5.0; // circa 500mm lens 35mm-equivalent
};

struct RelativePoseOptions {
    RansacOptions ransac;
    BundleOptions bundle;

    // Inlier threshold
    double max_error = 1.0;

    // TODO: refactor estimate_relative_pose to similarly to estimate_absolute_pose have a single entry point
    //bool estimate_focal_length = false;
    //bool estimate_extra_params = false;
    //bool shared_intrinsics = false;
    bool tangent_sampson = false;

    // Whether we should use real focal length checking: https://arxiv.org/abs/2311.16304
    // Assumes that principal points of both cameras are at origin.
    bool real_focal_check = false;
};

struct HomographyOptions {
    RansacOptions ransac;
    BundleOptions bundle;

    double max_error = 1.0;
};

typedef Eigen::Vector2d Point2D;
typedef Eigen::Vector3d Point3D;

// Used to store pairwise matches for generalized pose estimation
struct PairwiseMatches {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t cam_id1, cam_id2;
    std::vector<Point2D> x1, x2;
};

struct Line2D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line2D() {}
    Line2D(const Eigen::Vector2d &e1, const Eigen::Vector2d &e2) : x1(e1), x2(e2) {}
    Eigen::Vector2d x1, x2;
};
struct Line3D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line3D() {}
    Line3D(const Eigen::Vector3d &e1, const Eigen::Vector3d &e2) : X1(e1), X2(e2) {}
    Eigen::Vector3d X1, X2;
};

} // namespace poselib

#endif