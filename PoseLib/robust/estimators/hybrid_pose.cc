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

#include "hybrid_pose.h"

#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gp3p.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"

namespace poselib {

void HybridPoseEstimator::generate_models(std::vector<CameraPose> *models) {
    std::vector<CameraPose> models_p3p;
    std::vector<CameraPose> models_5p1pt;

    // sample data indices for both p3p and 5p1pt
    sampler.generate_sample(&sample_p3p, &pairs_5p1pt, &sample_5p1pt);

    // get p3p samples
    for (size_t k = 0; k < sample_sz_p3p; ++k) {
        xs[k] = x[sample_p3p[k]].homogeneous().normalized();
        Xs[k] = X[sample_p3p[k]];
    }
    
    // get 5p1pt samples
    // - 5 matches from first camera pair
    CameraPose pose1 = map_poses[matches[pairs_5p1pt[0]].cam_id1];
    CameraPose pose2;
    Eigen::Vector3d p1 = pose1.center();
    Eigen::Vector3d p2 = pose2.center();
    for (size_t k = 0; k < 5; ++k) {
        x1s[k] = pose1.derotate(matches[pairs_5p1pt[0]].x1[sample_5p1pt[k]].homogeneous().normalized());
        p1s[k] = p1;
        x2s[k] = pose2.derotate(matches[pairs_5p1pt[0]].x2[sample_5p1pt[k]].homogeneous().normalized());
        p2s[k] = p2;
    }

    // - 1 match from the second camera pair
    pose1 = map_poses[matches[pairs_5p1pt[1]].cam_id1];
    p1 = pose1.center();
    p2 = pose2.center();
    x1s[5] = pose1.derotate(matches[pairs_5p1pt[1]].x1[sample_5p1pt[5]].homogeneous().normalized());
    p1s[5] = p1;
    x2s[5] = pose2.derotate(matches[pairs_5p1pt[1]].x2[sample_5p1pt[5]].homogeneous().normalized());
    p2s[5] = p2;

    // run both solvers
    p3p(xs, Xs, &models_p3p);
    gen_relpose_5p1pt(p1s, x1s, p2s, x2s, &models_5p1pt);
    
    models->clear();
    models->shrink_to_fit();
    models->reserve(models_p3p.size() + models_5p1pt.size());
    models->insert(models->end(), models_p3p.begin(), models_p3p.end());
    models->insert(models->end(), models_5p1pt.begin(), models_5p1pt.end());
}

double HybridPoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    double th_pts, th_epi;
    th_pts = opt.max_errors[0] * opt.max_errors[0];
    th_epi = opt.max_errors[1] * opt.max_errors[1];

    // score the pose by sum of normalized reprojection and Sampson MSAC scores
    double score = compute_msac_score(pose, x, X, th_pts, inlier_count) / th_pts;

    for (const PairwiseMatches &m : matches) {
        const CameraPose &map_pose = map_poses[m.cam_id1];
        // Cameras are
        //  [map.R map.t]
        //  [R t]
        // Relative pose is [R * rig.R', t - R*rig.R'*rig.t]

        CameraPose rel_pose = pose;
        rel_pose.q = quat_multiply(rel_pose.q, quat_conj(map_pose.q));
        rel_pose.t -= rel_pose.rotate(map_pose.t);

        size_t inliers_2d2d = 0;
        score += compute_sampson_msac_score(rel_pose, m.x1, m.x2, th_epi, &inliers_2d2d) / th_epi;
        *inlier_count += inliers_2d2d;
    }

    return score;
}

void HybridPoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_errors[0];
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    refine_hybrid_pose(x, X, matches, map_poses, pose, bundle_opt, opt.max_errors[1]);
}

} // namespace poselib