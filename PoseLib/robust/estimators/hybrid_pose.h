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
    HybridPoseEstimator(const HybridPoseOptions &opt, const std::vector<Point2D> &points2D,
                        const std::vector<Point3D> &points3D, const std::vector<PairwiseMatches> &pairwise_matches,
                        const std::vector<CameraPose> &map_ext)
        : opt(opt), x(points2D), X(points3D), matches(pairwise_matches), map_poses(map_ext), sampler(0, {}, opt.ransac) {
        xs.resize(sample_sz_p3p);
        Xs.resize(sample_sz_p3p);
        sample_p3p.resize(sample_sz_p3p);
        num_data_p3p = points2D.size();

        x1s.resize(sample_sz_5p1pt);
        x2s.resize(sample_sz_5p1pt);
        p1s.resize(sample_sz_5p1pt);
        p2s.resize(sample_sz_5p1pt);
        sample_5p1pt.resize(sample_sz_5p1pt);
        num_data_5p1pt_sum = 0;
        for (const PairwiseMatches &m : matches) {
            num_data_5p1pt.push_back(m.x1.size());
            num_data_5p1pt_sum += m.x1.size();
            // check if there are pairs with enough matches for 5p1pt
            if ((m.x1.size() >= 5) && (num_data_5p1pt_5p_check == 0)) {
                num_data_5p1pt_5p_check = 1;
            } else if (m.x1.size() >= 1) {
                num_data_5p1pt_1p_check = 1;
            }
        }

        if (num_data_5p1pt_1p_check == 0 || num_data_5p1pt_5p_check == 0) {
            // disable 5p1pt sampling in case p3p has enough data
            num_data_5p1pt.clear();
            // if neither p3p nor 5p1pt has enough data
            // --> fail the num_data > sample_sz check in ransac_impl.h
            if (num_data_p3p < sample_sz_p3p) {
                num_data = 0;
            }
        } else {
            num_data = num_data_p3p + num_data_5p1pt_sum;
        }

        sampler = HybridSampler(num_data_p3p, num_data_5p1pt, opt.ransac);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz_p3p = 3;
    size_t num_data_p3p;

    const size_t sample_sz_5p1pt = 6;
    std::vector<size_t> num_data_5p1pt;
    size_t num_data_5p1pt_sum;
    size_t num_data_5p1pt_5p_check = 0;
    size_t num_data_5p1pt_1p_check = 0;
    
    size_t sample_sz = 1; // dummy value used for check in ransac_impl.h
    size_t num_data;

  private:
    const HybridPoseOptions &opt;
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &map_poses;

    HybridSampler sampler;

    // pre-allocated vectors for sampling
    std::vector<Point3D> xs, Xs;
    std::vector<size_t> sample_p3p;
    std::vector<Eigen::Vector3d> x1s, x2s, p1s, p2s;
    std::vector<size_t> pairs_5p1pt; // indices of the two camera pairs
    std::vector<size_t> sample_5p1pt; // first 5 matches from the first pair, last match from the second pair
};

} // namespace poselib

#endif