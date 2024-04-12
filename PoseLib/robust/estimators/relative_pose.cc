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

#include "relative_pose.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "PoseLib/solvers/relpose_6pt_focal.h"
#include "PoseLib/solvers/relpose_7pt.h"

#include <iostream>

namespace poselib {

void RelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_5pt(x1s, x2s, models);
}

double RelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_sampson_msac_score(pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void RelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(*pose, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 5) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }
    refine_relpose(x1_inlier, x2_inlier, pose, bundle_opt);
}

void SharedFocalRelativePoseEstimator::generate_models(ImagePairVector *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_6pt_shared_focal(x1s, x2s, models);
}

double SharedFocalRelativePoseEstimator::score_model(const ImagePair &image_pair, size_t *inlier_count) const {
    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();
    // K_inv << 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0 / calib_pose.camera.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair.pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void SharedFocalRelativePoseEstimator::refine_model(ImagePair *image_pair) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    Eigen::Matrix3d K_inv;
    // K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, calib_pose->camera.focal();
    K_inv << 1.0 / image_pair->camera1.focal(), 0.0, 0.0, 0.0, 1.0 / image_pair->camera1.focal(), 0.0, 0.0, 0.0, 1.0;
    Eigen::Matrix3d E;
    essential_from_motion(image_pair->pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    // Find approximate inliers and bundle over these with a truncated loss
    std::vector<char> inliers;
    int num_inl = get_inliers(F, x1, x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    std::vector<Eigen::Vector2d> x1_inlier, x2_inlier;
    x1_inlier.reserve(num_inl);
    x2_inlier.reserve(num_inl);

    if (num_inl <= 6) {
        return;
    }

    for (size_t pt_k = 0; pt_k < x1.size(); ++pt_k) {
        if (inliers[pt_k]) {
            x1_inlier.push_back(x1[pt_k]);
            x2_inlier.push_back(x2[pt_k]);
        }
    }

    refine_shared_focal_relpose(x1_inlier, x2_inlier, image_pair, bundle_opt);
}

void GeneralizedRelativePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    // TODO replace by general 6pt solver?

    bool done = false;
    int pair0 = 0, pair1 = 1;
    while (!done) {
        pair0 = random_int(rng) % matches.size();
        if (matches[pair0].x1.size() < 5)
            continue;

        pair1 = random_int(rng) % matches.size();
        if (pair0 == pair1 || matches[pair1].x1.size() == 0)
            continue;

        done = true;
    }

    // Sample 5 points from the first camera pair
    CameraPose pose1 = rig1_poses[matches[pair0].cam_id1];
    CameraPose pose2 = rig2_poses[matches[pair0].cam_id2];
    Eigen::Vector3d p1 = pose1.center();
    Eigen::Vector3d p2 = pose2.center();
    draw_sample(5, matches[pair0].x1.size(), &sample, rng);
    for (size_t k = 0; k < 5; ++k) {
        x1s[k] = pose1.derotate(matches[pair0].x1[sample[k]].homogeneous().normalized());
        p1s[k] = p1;
        x2s[k] = pose2.derotate(matches[pair0].x2[sample[k]].homogeneous().normalized());
        p2s[k] = p2;
    }

    // Sample one point from the second camera pair
    pose1 = rig1_poses[matches[pair1].cam_id1];
    pose2 = rig2_poses[matches[pair1].cam_id2];
    p1 = pose1.center();
    p2 = pose2.center();
    size_t ind = random_int(rng) % matches[pair1].x1.size();
    x1s[5] = pose1.derotate(matches[pair1].x1[ind].homogeneous().normalized());
    p1s[5] = p1;
    x2s[5] = pose2.derotate(matches[pair1].x2[ind].homogeneous().normalized());
    p2s[5] = p2;

    gen_relpose_5p1pt(p1s, x1s, p2s, x2s, models);
}

double GeneralizedRelativePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    *inlier_count = 0;
    double cost = 0;
    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose.t);
        pose2.q = quat_multiply(pose2.q, pose.q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        size_t local_inlier_count = 0;
        cost += compute_sampson_msac_score(relpose, m.x1, m.x2, opt.max_epipolar_error * opt.max_epipolar_error,
                                           &local_inlier_count);
        *inlier_count += local_inlier_count;
    }

    return cost;
}

void GeneralizedRelativePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    std::vector<PairwiseMatches> inlier_matches;
    inlier_matches.resize(matches.size());

    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = rig1_poses[m.cam_id1];
        CameraPose pose2 = rig2_poses[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(pose->t);
        pose2.q = quat_multiply(pose2.q, pose->q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        // Compute inliers with a relaxed threshold
        std::vector<char> inliers;
        int num_inl = get_inliers(relpose, m.x1, m.x2, 5 * (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);

        inlier_matches[match_k].cam_id1 = m.cam_id1;
        inlier_matches[match_k].cam_id2 = m.cam_id2;
        inlier_matches[match_k].x1.reserve(num_inl);
        inlier_matches[match_k].x2.reserve(num_inl);

        for (size_t k = 0; k < m.x1.size(); ++k) {
            if (inliers[k]) {
                inlier_matches[match_k].x1.push_back(m.x1[k]);
                inlier_matches[match_k].x2.push_back(m.x2[k]);
            }
        }
    }

    refine_generalized_relpose(inlier_matches, rig1_poses, rig2_poses, pose, bundle_opt);
}

void FundamentalEstimator::generate_models(std::vector<Eigen::Matrix3d> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    relpose_7pt(x1s, x2s, models);

    if (opt.real_focal_check) {
        for (int i = models->size() - 1; i >= 0; i--) {
            if (!calculate_RFC((*models)[i]))
                models->erase(models->begin() + i);
        }
    }
}

double FundamentalEstimator::score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const {
    return compute_sampson_msac_score(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, inlier_count);
}

void FundamentalEstimator::refine_model(Eigen::Matrix3d *F) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_epipolar_error;
    bundle_opt.max_iterations = 25;

    refine_fundamental(x1, x2, F, bundle_opt);
}

} // namespace poselib
