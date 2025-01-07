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
#include "real_t.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

struct RansacOptions {
    size_t max_iterations = 100000;
    size_t min_iterations = 1000;
    real_t dyn_num_trials_mult = 3.0;
    real_t success_prob = 0.9999;
    real_t max_reproj_error = 12.0;  // used for 2D-3D matches
    real_t max_epipolar_error = 1.0; // used for 2D-2D matches
    unsigned long seed = 0;
    // If we should use PROSAC sampling. Assumes data is sorted
    bool progressive_sampling = false;
    size_t max_prosac_iterations = 100000;
    // Whether we should use real focal length checking: https://arxiv.org/abs/2311.16304
    // Assumes that principal points of both cameras are at origin.
    bool real_focal_check = false;
};

struct RansacStats {
    size_t refinements = 0;
    size_t iterations = 0;
    size_t num_inliers = 0;
    real_t inlier_ratio = 0;
    real_t model_score = std::numeric_limits<real_t>::max();
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
    real_t loss_scale = 1.0;
    real_t gradient_tol = 1e-10;
    real_t step_tol = 1e-8;
    real_t initial_lambda = 1e-3;
    real_t min_lambda = 1e-10;
    real_t max_lambda = 1e10;
    bool verbose = false;
};

struct BundleStats {
    size_t iterations = 0;
    real_t initial_cost;
    real_t cost;
    real_t lambda;
    size_t invalid_steps;
    real_t step_norm;
    real_t grad_norm;
};

typedef Eigen::Vector2_t Point2D;
typedef Eigen::Vector3_t Point3D;

// Used to store pairwise matches for generalized pose estimation
struct PairwiseMatches {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    size_t cam_id1, cam_id2;
    std::vector<Point2D> x1, x2;
};

struct Line2D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line2D() {}
    Line2D(const Eigen::Vector2_t &e1, const Eigen::Vector2_t &e2) : x1(e1), x2(e2) {}
    Eigen::Vector2_t x1, x2;
};
struct Line3D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line3D() {}
    Line3D(const Eigen::Vector3_t &e1, const Eigen::Vector3_t &e2) : X1(e1), X2(e2) {}
    Eigen::Vector3_t X1, X2;
};

} // namespace poselib

#endif