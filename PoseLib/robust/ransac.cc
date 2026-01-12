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

RansacStats ransac_pnp(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const AbsolutePoseOptions &opt,
                       CameraPose *best_model, std::vector<char> *best_inliers) {

    if (!opt.ransac.score_initial_model) {
        best_model->q << 1.0, 0.0, 0.0, 0.0;
        best_model->t.setZero();
    }
    AbsolutePoseEstimator estimator(opt, x, X);
    RansacStats stats = ransac<AbsolutePoseEstimator>(estimator, opt.ransac, best_model);

    get_inliers(*best_model, x, X, opt.max_error * opt.max_error, best_inliers);

    return stats;
}

RansacStats ransac_pnpf(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const AbsolutePoseOptions &opt,
                        Image *best_model, std::vector<char> *best_inliers) {

    best_model->pose.q << 1.0, 0.0, 0.0, 0.0;
    best_model->pose.t.setZero();
    best_model->camera.model_id = CameraModelId::SIMPLE_PINHOLE;
    best_model->camera.width = 0;
    best_model->camera.height = 0;
    best_model->camera.params = {1.0, 0.0, 0.0};

    FocalAbsolutePoseEstimator estimator(opt, x, X);
    RansacStats stats = ransac<FocalAbsolutePoseEstimator>(estimator, opt.ransac, best_model);

    get_inliers(*best_model, x, X, opt.max_error * opt.max_error, best_inliers);

    return stats;
}

RansacStats ransac_pnpfr(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const AbsolutePoseOptions &opt,
                         Image *best_model, std::vector<char> *best_inliers) {

    best_model->pose.q << 1.0, 0.0, 0.0, 0.0;
    best_model->pose.t.setZero();
    best_model->camera.model_id = CameraModelId::SIMPLE_DIVISION;
    best_model->camera.width = 0;
    best_model->camera.height = 0;
    best_model->camera.params = {1.0, 0.0, 0.0};

    RDAbsolutePoseEstimator estimator(opt, x, X);
    RansacStats stats = ransac<RDAbsolutePoseEstimator>(estimator, opt.ransac, best_model);

    get_inliers(*best_model, x, X, opt.max_error * opt.max_error, best_inliers);

    return stats;
}

RansacStats ransac_gen_pnp(const std::vector<std::vector<Point2D>> &x, const std::vector<std::vector<Point3D>> &X,
                           const std::vector<CameraPose> &camera_ext, const AbsolutePoseOptions &opt,
                           CameraPose *best_model, std::vector<std::vector<char>> *best_inliers) {
    if (!opt.ransac.score_initial_model) {
        best_model->q << 1.0, 0.0, 0.0, 0.0;
        best_model->t.setZero();
    }
    GeneralizedAbsolutePoseEstimator estimator(opt, x, X, camera_ext);
    RansacStats stats = ransac<GeneralizedAbsolutePoseEstimator>(estimator, opt.ransac, best_model);

    best_inliers->resize(camera_ext.size());
    for (size_t k = 0; k < camera_ext.size(); ++k) {
        CameraPose full_pose;
        full_pose.q = quat_multiply(camera_ext[k].q, best_model->q);
        full_pose.t = camera_ext[k].rotate(best_model->t) + camera_ext[k].t;
        get_inliers(full_pose, x[k], X[k], opt.max_error * opt.max_error, &(*best_inliers)[k]);
    }

    return stats;
}

RansacStats ransac_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                        const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                        const AbsolutePoseOptions &opt, CameraPose *best_model, std::vector<char> *inliers_points,
                        std::vector<char> *inliers_lines) {

    if (!opt.ransac.score_initial_model) {
        best_model->q << 1.0, 0.0, 0.0, 0.0;
        best_model->t.setZero();
    }
    AbsolutePosePointLineEstimator estimator(opt, points2D, points3D, lines2D, lines3D);
    RansacStats stats = ransac<AbsolutePosePointLineEstimator>(estimator, opt.ransac, best_model);

    double th_pts, th_lines;
    if (opt.max_errors.size() != 2) {
        th_pts = th_lines = opt.max_error * opt.max_error;
    } else {
        th_pts = opt.max_errors[0] * opt.max_errors[0];
        th_lines = opt.max_errors[1] * opt.max_errors[1];
    }

    get_inliers(*best_model, points2D, points3D, th_pts, inliers_points);
    get_inliers(*best_model, lines2D, lines3D, th_lines, inliers_lines);

    return stats;
}

RansacStats ransac_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                           const RelativePoseOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers) {
    if (!opt.ransac.score_initial_model) {
        best_model->q << 1.0, 0.0, 0.0, 0.0;
        best_model->t.setZero();
    }
    RelativePoseEstimator estimator(opt, x1, x2);
    RansacStats stats = ransac<RelativePoseEstimator>(estimator, opt.ransac, best_model);

    get_inliers(*best_model, x1, x2, opt.max_error * opt.max_error, best_inliers);

    return stats;
}
RansacStats ransac_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const Camera &camera1,
                           const Camera &camera2, const RelativePoseOptions &opt, CameraPose *best_model,
                           std::vector<char> *best_inliers) {

    best_model->q << 1.0, 0.0, 0.0, 0.0;
    best_model->t.setZero();
    CameraRelativePoseEstimator estimator(opt, x1, x2, camera1, camera2);
    RansacStats stats = ransac<CameraRelativePoseEstimator>(estimator, opt.ransac, best_model);

    get_tangent_sampson_inliers(*best_model, estimator.d1, estimator.d2, estimator.M1, estimator.M2,
                                opt.max_error * opt.max_error, best_inliers);

    return stats;
}

RansacStats ransac_monodepth_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                     const std::vector<double> &d1, const std::vector<double> &d2,
                                     const RansacOptions &opt, MonoDepthTwoViewGeometry *best_model,
                                     std::vector<char> *best_inliers) {
    best_model->pose.q << 1.0, 0.0, 0.0, 0.0;
    best_model->pose.t.setZero();
    RelativePoseMonoDepthEstimator estimator(opt, x1, x2, d1, d2);
    RansacStats stats = ransac<RelativePoseMonoDepthEstimator>(estimator, opt, best_model);
    get_inliers(best_model->pose, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, best_inliers);
    return stats;
}

RansacStats ransac_shared_focal_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                        const RelativePoseOptions &opt, ImagePair *best_model,
                                        std::vector<char> *best_inliers) {
    if (!opt.ransac.score_initial_model) {
        best_model->pose.q << 1.0, 0.0, 0.0, 0.0;
        best_model->pose.t.setZero();
        best_model->camera1 = Camera(SimplePinholeCameraModel::model_id, std::vector<double>{1.0, 0.0, 0.0}, -1, -1);
        best_model->camera2 = best_model->camera1;
    }
    SharedFocalRelativePoseEstimator estimator(opt, x1, x2);
    RansacStats stats = ransac<SharedFocalRelativePoseEstimator>(estimator, opt.ransac, best_model);

    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, best_model->camera1.focal();
    Eigen::Matrix3d E;
    essential_from_motion(best_model->pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    get_inliers(F, x1, x2, opt.max_error * opt.max_error, best_inliers);

    return stats;
}

RansacStats ransac_shared_focal_monodepth_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                                  const std::vector<double> &d1, const std::vector<double> &d2,
                                                  const RansacOptions &opt, MonoDepthImagePair *best_model,
                                                  std::vector<char> *best_inliers) {
    best_model->geometry.pose.q << 1.0, 0.0, 0.0, 0.0;
    best_model->geometry.pose.t.setZero();
    best_model->camera1 = Camera(SimplePinholeCameraModel::model_id, std::vector<double>{1.0, 0.0, 0.0}, -1, -1);
    best_model->camera2 = best_model->camera1;
    SharedFocalMonodepthPoseEstimator estimator(opt, x1, x2, d1, d2);
    RansacStats stats = ransac<SharedFocalMonodepthPoseEstimator>(estimator, opt, best_model);

    Eigen::Matrix3d K_inv;
    K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, best_model->camera1.focal();
    Eigen::Matrix3d E;
    essential_from_motion(best_model->geometry.pose, &E);
    Eigen::Matrix3d F = K_inv * (E * K_inv);

    get_inliers(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, best_inliers);

    return stats;
}

RansacStats ransac_varying_focal_monodepth_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                                   const std::vector<double> &d1, const std::vector<double> &d2,
                                                   const RansacOptions &opt, MonoDepthImagePair *best_model,
                                                   std::vector<char> *best_inliers) {
    best_model->geometry.pose.q << 1.0, 0.0, 0.0, 0.0;
    best_model->geometry.pose.t.setZero();
    best_model->camera1 = Camera(SimplePinholeCameraModel::model_id, std::vector<double>{1.0, 0.0, 0.0}, -1, -1);
    best_model->camera2 = best_model->camera1;
    VaryingFocalMonodepthPoseEstimator estimator(opt, x1, x2, d1, d2);
    RansacStats stats = ransac<VaryingFocalMonodepthPoseEstimator>(estimator, opt, best_model);

    Eigen::DiagonalMatrix<double, 3> K1_inv(1.0, 1.0, best_model->camera1.focal()),
        K2_inv(1.0, 1.0, best_model->camera2.focal());
    Eigen::Matrix3d E;
    essential_from_motion(best_model->geometry.pose, &E);
    Eigen::Matrix3d F = K2_inv * (E * K1_inv);
    get_inliers(F, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, best_inliers);

    return stats;
}

RansacStats ransac_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                               const RelativePoseOptions &opt, Eigen::Matrix3d *best_model,
                               std::vector<char> *best_inliers) {

    if (!opt.ransac.score_initial_model) {
        best_model->setIdentity();
    }
    RansacStats stats;

    FundamentalEstimator estimator(opt, x1, x2);
    stats = ransac<FundamentalEstimator, Eigen::Matrix3d>(estimator, opt.ransac, best_model);
    get_inliers(*best_model, x1, x2, opt.max_error * opt.max_error, best_inliers);

    return stats;
}

RansacStats ransac_rd_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                  std::vector<double> &ks, const double min_k, const double max_k,
                                  const RelativePoseOptions &opt, ProjectiveImagePair *best_model,
                                  std::vector<char> *best_inliers) {

    best_model->F.setIdentity();
    best_model->camera1 = Camera("DIVISION", std::vector<double>{1.0, 1.0, 0.0, 0.0, 0.0}, -1, -1);
    best_model->camera2 = Camera("DIVISION", std::vector<double>{1.0, 1.0, 0.0, 0.0, 0.0}, -1, -1);
    RansacStats stats;

    RDFundamentalEstimator estimator(opt, x1, x2, ks, min_k, max_k);
    stats = ransac<RDFundamentalEstimator, ProjectiveImagePair>(estimator, opt.ransac, best_model);

    get_tangent_sampson_inliers(best_model->F, best_model->camera1, best_model->camera2, x1, x2,
                                opt.max_error * opt.max_error, best_inliers);
    return stats;
}

RansacStats ransac_shared_rd_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                         std::vector<double> &ks, const double min_k, const double max_k,
                                         const RelativePoseOptions &opt, ProjectiveImagePair *best_model,
                                         std::vector<char> *best_inliers) {

    best_model->F.setIdentity();
    best_model->camera1 = Camera("DIVISION", std::vector<double>{1.0, 1.0, 0.0, 0.0, 0.0}, -1, -1);
    best_model->camera2 = Camera("DIVISION", std::vector<double>{1.0, 1.0, 0.0, 0.0, 0.0}, -1, -1);
    RansacStats stats;

    SharedRDFundamentalEstimator estimator(opt, x1, x2, ks, min_k, max_k);
    stats = ransac<SharedRDFundamentalEstimator, ProjectiveImagePair>(estimator, opt.ransac, best_model);

    get_tangent_sampson_inliers(best_model->F, best_model->camera1, best_model->camera2, x1, x2,
                                opt.max_error * opt.max_error, best_inliers);
    return stats;
}

RansacStats ransac_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                              const HomographyOptions &opt, Eigen::Matrix3d *best_model,
                              std::vector<char> *best_inliers) {

    if (!opt.ransac.score_initial_model) {
        best_model->setIdentity();
    }

    HomographyEstimator estimator(opt, x1, x2);
    RansacStats stats = ransac<HomographyEstimator, Eigen::Matrix3d>(estimator, opt.ransac, best_model);

    get_homography_inliers(*best_model, x1, x2, opt.max_error * opt.max_error, best_inliers);

    return stats;
}

RansacStats ransac_gen_relpose(const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
                               const std::vector<CameraPose> &camera2_ext, const RelativePoseOptions &opt,
                               CameraPose *best_model, std::vector<std::vector<char>> *best_inliers) {
    if (!opt.ransac.score_initial_model) {
        best_model->q << 1.0, 0.0, 0.0, 0.0;
        best_model->t.setZero();
    }
    GeneralizedRelativePoseEstimator estimator(opt, matches, camera1_ext, camera2_ext);
    RansacStats stats = ransac<GeneralizedRelativePoseEstimator>(estimator, opt.ransac, best_model);

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
        get_inliers(relpose, m.x1, m.x2, (opt.max_error * opt.max_error), &inliers);
    }

    return stats;
}

RansacStats ransac_hybrid_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                               const std::vector<PairwiseMatches> &matches2D_2D, const std::vector<CameraPose> &map_ext,
                               const HybridPoseOptions &opt, CameraPose *best_model, std::vector<char> *inliers_2D_3D,
                               std::vector<std::vector<char>> *inliers_2D_2D) {
    if (!opt.ransac.score_initial_model) {
        best_model->q << 1.0, 0.0, 0.0, 0.0;
        best_model->t.setZero();
    }
    HybridPoseEstimator estimator(opt, points2D, points3D, matches2D_2D, map_ext);
    RansacStats stats = ransac<HybridPoseEstimator>(estimator, opt.ransac, best_model);

    double th_pts, th_epi;
    th_pts = opt.max_errors[0] * opt.max_errors[0];
    th_epi = opt.max_errors[1] * opt.max_errors[1];

    get_inliers(*best_model, points2D, points3D, th_pts, inliers_2D_3D);

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
        get_inliers(rel_pose, m.x1, m.x2, th_epi, &inliers);
    }

    return stats;
}

RansacStats ransac_1D_radial_pnp(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                                 const AbsolutePoseOptions &opt, CameraPose *best_model,
                                 std::vector<char> *best_inliers) {

    if (!opt.ransac.score_initial_model) {
        best_model->q << 1.0, 0.0, 0.0, 0.0;
        best_model->t.setZero();
    }
    Radial1DAbsolutePoseEstimator estimator(opt, x, X);
    RansacStats stats = ransac<Radial1DAbsolutePoseEstimator>(estimator, opt.ransac, best_model);

    get_inliers_1D_radial(*best_model, x, X, opt.max_error * opt.max_error, best_inliers);

    return stats;
}

} // namespace poselib
