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
// Hybrid absolute pose estimator for point and line correspondences.

#pragma once

#include <PoseLib/camera_pose.h>
#include <PoseLib/types.h>
#include <random>
#include <vector>

namespace poselib {

// Hybrid estimator for absolute pose from points and lines.
//
// Supports 4 minimal solvers:
//   0: P3P (3 points)
//   1: P2P1LL (2 points + 1 line)
//   2: P1P2LL (1 point + 2 lines)
//   3: P3LL (3 lines)
//
class HybridPointLineAbsolutePoseEstimator {
  public:
    HybridPointLineAbsolutePoseEstimator(const HybridRansacOptions &opt, const std::vector<Point2D> &points2D,
                                         const std::vector<Point3D> &points3D, const std::vector<Line2D> &lines2D,
                                         const std::vector<Line3D> &lines3D);

    size_t num_data_types() const { return 2; }
    std::vector<size_t> num_data() const;
    size_t num_minimal_solvers() const { return 4; }
    std::vector<std::vector<size_t>> min_sample_sizes() const;
    std::vector<double> solver_probabilities() const;

    void generate_sample(size_t solver_idx, std::vector<std::vector<size_t>> *sample) const;
    void generate_models(const std::vector<std::vector<size_t>> &sample, size_t solver_idx,
                         std::vector<CameraPose> *models) const;
    double score_model(const CameraPose &pose, std::vector<size_t> *inliers_per_type) const;
    void refine_model(CameraPose *pose) const;

  private:
    void random_sample(size_t n, size_t k, std::vector<size_t> *sample) const;
    static unsigned long long combination(size_t n, size_t k);

    const HybridRansacOptions &opt_;
    const std::vector<Point2D> &points2D_;
    const std::vector<Point3D> &points3D_;
    const std::vector<Line2D> &lines2D_;
    const std::vector<Line3D> &lines3D_;

    mutable std::mt19937 rng_;

    // Pre-allocated buffers for minimal solvers
    mutable std::vector<Point3D> xs_, Xs_, ls_, Cs_, Vs_;

    // Cached inlier indices (updated by score_model, used by refine_model)
    mutable std::vector<std::vector<size_t>> cached_inlier_indices_;
};

} // namespace poselib
