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

#include "ransac.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/robust/estimators/absolute_pose.h"
#include "PoseLib/robust/estimators/homography.h"
#include "PoseLib/robust/estimators/hybrid_pose.h"
#include "PoseLib/robust/estimators/relative_pose.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/relpose_5pt.h"
#include "ransac_impl.h"

namespace poselib {

RansacStats ransac_pnp(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const RansacOptions &opt,
                       CameraPose *best_model, std::vector<char> *best_inliers) {

    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t.setZero();
    AbsolutePoseEstimator estimator(opt, x, X);
    RansacStats stats = ransac<AbsolutePoseEstimator>(estimator, opt, best_model);

    get_inliers(*best_model, x, X, opt.max_reproj_error * opt.max_reproj_error, best_inliers);

    return stats;
}

RansacStats ransac_gen_pnp(const std::vector<std::vector<Point2D>> &x, const std::vector<std::vector<Point3D>> &X,
                           const std::vector<CameraPose> &camera_ext, const RansacOptions &opt, CameraPose *best_model,
                           std::vector<std::vector<char>> *best_inliers) {
    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t.setZero();
    GeneralizedAbsolutePoseEstimator estimator(opt, x, X, camera_ext);
    RansacStats stats = ransac<GeneralizedAbsolutePoseEstimator>(estimator, opt, best_model);

    best_inliers->resize(camera_ext.size());
    for (size_t k = 0; k < camera_ext.size(); ++k) {
        CameraPose full_pose;
        full_pose.q = quat_multiply(camera_ext[k].q, best_model->q);
        full_pose.t = camera_ext[k].rotate(best_model->t) + camera_ext[k].t;
        get_inliers(full_pose, x[k], X[k], opt.max_reproj_error * opt.max_reproj_error, &(*best_inliers)[k]);
    }

    return stats;
}

RansacStats ransac_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                        const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                        const RansacOptions &opt, CameraPose *best_model, std::vector<char> *inliers_points,
                        std::vector<char> *inliers_lines) {

    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t.setZero();
    AbsolutePosePointLineEstimator estimator(opt, points2D, points3D, lines2D, lines3D);
    RansacStats stats = ransac<AbsolutePosePointLineEstimator>(estimator, opt, best_model);

    get_inliers(*best_model, points2D, points3D, opt.max_reproj_error * opt.max_reproj_error, inliers_points);
    get_inliers(*best_model, lines2D, lines3D, opt.max_epipolar_error * opt.max_epipolar_error, inliers_lines);

    return stats;
}

RansacStats ransac_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const RansacOptions &opt,
                           CameraPose *best_model, std::vector<char> *best_inliers) {
    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t.setZero();
    RelativePoseEstimator estimator(opt, x1, x2);
    RansacStats stats = ransac<RelativePoseEstimator>(estimator, opt, best_model);

    get_inliers(*best_model, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, best_inliers);

    return stats;
}

RansacStats ransac_shared_focal_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                        const RansacOptions &opt, ImagePair *best_model,
                                        std::vector<char> *best_inliers) {
    best_model->pose.q << 1.0, 0.0, 0.0, 0.0;
    best_model->pose.t.setZero();
    best_model->camera1 = Camera("SIMPLE_PINHOLE", std::vector<double>{1.0, 0.0, 0.0}, -1, -1);
    best_model->camera2 = best_model->camera1;
    SharedFocalRelativePoseEstimator estimator(opt, x1, x2);
    RansacStats stats = ransac<SharedFocalRelativePoseEstimator>(estimator, opt, best_model);

    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, best_model->camera1.focal();
    Eigen::Matrix3d E;
    essential_from_motion(best_model->pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    get_inliers(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, best_inliers);

    return stats;
}

RansacStats ransac_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const RansacOptions &opt,
                               Eigen::Matrix3d *best_model, std::vector<char> *best_inliers) {

    best_model->setIdentity();
    RansacStats stats;

    FundamentalEstimator estimator(opt, x1, x2);
    stats = ransac<FundamentalEstimator, Eigen::Matrix3d>(estimator, opt, best_model);
    get_inliers(*best_model, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, best_inliers);

    return stats;
}

RansacStats ransac_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const RansacOptions &opt,
                              Eigen::Matrix3d *best_model, std::vector<char> *best_inliers) {

    best_model->setIdentity();

    HomographyEstimator estimator(opt, x1, x2);
    RansacStats stats = ransac<HomographyEstimator, Eigen::Matrix3d>(estimator, opt, best_model);

    get_homography_inliers(*best_model, x1, x2, opt.max_reproj_error * opt.max_reproj_error, best_inliers);

    return stats;
}

RansacStats ransac_gen_relpose(const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
                               const std::vector<CameraPose> &camera2_ext, const RansacOptions &opt,
                               CameraPose *best_model, std::vector<std::vector<char>> *best_inliers) {
    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t.setZero();
    GeneralizedRelativePoseEstimator estimator(opt, matches, camera1_ext, camera2_ext);
    RansacStats stats = ransac<GeneralizedRelativePoseEstimator>(estimator, opt, best_model);

    best_inliers->resize(matches.size());
    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = camera1_ext[m.cam_id1];
        CameraPose pose2 = camera2_ext[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.rotate(best_model->t);
        pose2.q = quat_multiply(pose2.q, best_model->q);

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.q = quat_multiply(pose2.q, quat_conj(pose1.q));
        relpose.t = pose2.t - relpose.rotate(pose1.t);

        // Compute inliers
        std::vector<char> &inliers = (*best_inliers)[match_k];
        get_inliers(relpose, m.x1, m.x2, (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    }

    return stats;
}

RansacStats ransac_hybrid_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                               const std::vector<PairwiseMatches> &matches2D_2D, const std::vector<CameraPose> &map_ext,
                               const RansacOptions &opt, CameraPose *best_model, std::vector<char> *inliers_2D_3D,
                               std::vector<std::vector<char>> *inliers_2D_2D) {
    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t.setZero();
    HybridPoseEstimator estimator(opt, points2D, points3D, matches2D_2D, map_ext);
    RansacStats stats = ransac<HybridPoseEstimator>(estimator, opt, best_model);

    get_inliers(*best_model, points2D, points3D, opt.max_reproj_error * opt.max_reproj_error, inliers_2D_3D);

    inliers_2D_2D->resize(matches2D_2D.size());
    for (size_t match_k = 0; match_k < matches2D_2D.size(); ++match_k) {
        const PairwiseMatches &m = matches2D_2D[match_k];
        const CameraPose &map_pose = map_ext[m.cam_id1];
        // Cameras are
        //  [rig.R rig.t]
        //  [R t]
        // Relative pose is [R * rig.R' t - R*rig.R' * rig.t]

        CameraPose rel_pose = *best_model;
        // rel_pose.R = rel_pose.R * map_pose.R.transpose();
        // rel_pose.t -= rel_pose.R * map_pose.t;
        rel_pose.q = quat_multiply(rel_pose.q, quat_conj(map_pose.q));
        rel_pose.t -= rel_pose.rotate(map_pose.t);

        std::vector<char> &inliers = (*inliers_2D_2D)[match_k];
        get_inliers(rel_pose, m.x1, m.x2, (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    }

    return stats;
}

RansacStats ransac_1D_radial_pnp(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const RansacOptions &opt,
                                 CameraPose *best_model, std::vector<char> *best_inliers) {

    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t.setZero();
    Radial1DAbsolutePoseEstimator estimator(opt, x, X);
    RansacStats stats = ransac<Radial1DAbsolutePoseEstimator>(estimator, opt, best_model);

    get_inliers_1D_radial(*best_model, x, X, opt.max_reproj_error * opt.max_reproj_error, best_inliers);

    return stats;
}

} // namespace poselib
