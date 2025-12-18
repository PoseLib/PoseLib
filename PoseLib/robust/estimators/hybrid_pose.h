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

#ifndef POSELIB_ROBUST_ESTIMATORS_HYBRID_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_HYBRID_POSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"

namespace poselib {

class HybridPoseEstimator {
  public:
    HybridPoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D,
                        const std::vector<Point3D> &points3D, const std::vector<PairwiseMatches> &pairwise_matches,
                        const std::vector<CameraPose> &map_ext)
        : opt(ransac_opt), x(points2D), X(points3D), matches(pairwise_matches), map_poses(map_ext) {
        rng = opt.seed;
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
        num_data = points2D.size();
        for (const PairwiseMatches &m : matches) {
            num_data += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3;
    size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &map_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Point3D> xs, Xs;
    std::vector<size_t> sample;
};

} // namespace poselib

#endif