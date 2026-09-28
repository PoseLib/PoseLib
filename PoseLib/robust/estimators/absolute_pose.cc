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

#include "absolute_pose.h"

#include "PoseLib/misc/constants.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gp3p.h"
#include "PoseLib/solvers/gp4ps.h"
#include "PoseLib/solvers/p1p2ll.h"
#include "PoseLib/solvers/p1p3llf.h"
#include "PoseLib/solvers/p2p1ll.h"
#include "PoseLib/solvers/p2p2llf.h"
#include "PoseLib/solvers/p35pf.h"
#include "PoseLib/solvers/p3ll.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/p3p1llf.h"
#include "PoseLib/solvers/p4llf.h"
#include "PoseLib/solvers/p4pf.h"
#include "PoseLib/solvers/p5lp_radial.h"
#include "PoseLib/solvers/p5pf.h"
#include "PoseLib/solvers/p5pfr.h"

namespace poselib {

void AbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    models->clear();
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]].homogeneous().normalized();
        Xs[k] = X[sample[k]];
    }
    p3p(xs, Xs, models);
}

double AbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_msac_score(pose, x, X, opt.max_error * opt.max_error, inlier_count);
}

void AbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    bundle_adjust(x, X, pose, bundle_opt);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Bearing-vector absolute pose estimator (for any central camera model)

void BearingAbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    models->clear();
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        // Bearings should already be unit length and
        // P3P is sensitive to non-unit inputs
        xs[k] = b[sample[k]];
        Xs[k] = X[sample[k]];
    }
    p3p(xs, Xs, models);
}

double BearingAbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_msac_score_bearing(pose, b, X, opt.max_error * opt.max_error, inlier_count);
}

void BearingAbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;

    bundle_adjust_bearing(b, X, pose, bundle_opt);
}

void FocalAbsolutePoseEstimator::generate_models(std::vector<Image> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]];
        Xs[k] = X[sample[k]];
    }

    std::vector<CameraPose> poses;
    std::vector<double> focals;

    if (minimal_solver == Solver::P4Pf) {
        p4pf(xs, Xs, &poses, &focals);
    } else if (minimal_solver == Solver::P35Pf) {
        p35pf(xs, Xs, &poses, &focals);
    } else { // if(minimal_solver == Solver::P5Pf) {
        p5pf(xs, Xs, &poses, &focals);
    }

    models->clear();
    for (size_t i = 0; i < poses.size(); ++i) {
        if (focals[i] < 0)
            continue;

        if (max_focal_length >= 0 && focals[i] > max_focal_length)
            continue;

        Camera camera;
        camera.model_id = CameraModelId::SIMPLE_PINHOLE;
        camera.width = 0;
        camera.height = 0;
        camera.params = {focals[i], 0.0, 0.0};

        Image image(poses[i], camera);

        if (refine_minimal_sample) {
            BundleOptions bundle_opt;
            bundle_opt.loss_type = BundleOptions::LossType::TRIVIAL;
            bundle_opt.max_iterations = 25;
            bundle_opt.refine_focal_length = true;
            bundle_opt.refine_principal_point = false;
            bundle_opt.refine_extra_params = false;
            bundle_adjust(xs, Xs, &image, bundle_opt);
        }

        if (filter_minimal_sample) {
            // check if all are inliers (since this is an overdetermined problem)
            size_t inlier_count = 0;
            compute_msac_score(image, xs, Xs, opt.max_error * opt.max_error, &inlier_count);
            if (inlier_count < 4) {
                continue;
            }
        }

        models->emplace_back(image);
    }
}

double FocalAbsolutePoseEstimator::score_model(const Image &image, size_t *inlier_count) const {
    if (image.camera.focal() < 0) {
        // Invalid focal length, skip this model
        return std::numeric_limits<double>::max();
    }
    double score = compute_msac_score(image, x, X, opt.max_error * opt.max_error, inlier_count);
    if (inlier_scoring) {
        // We do a combined MSAC score and inlier counting for model scoring. For some unknown reason this
        // seems slightly more robust? I have no idea...
        score += static_cast<double>(x.size() - *inlier_count) * opt.max_error * opt.max_error;
    }
    if (max_focal_length > 0 && image.camera.focal() > max_focal_length) {
        score = std::numeric_limits<double>::max();
    }
    return score;
}

void FocalAbsolutePoseEstimator::refine_model(Image *image) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;
    bundle_opt.refine_focal_length = true;
    bundle_opt.refine_principal_point = false;
    bundle_opt.refine_extra_params = false;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    bundle_adjust(x, X, image, bundle_opt);
}

double FocalAbsolutePoseEstimator::compute_max_focal_length(double min_fov) {
    if (min_fov <= 0) {
        return -1;
    }
    double max_coord = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        max_coord = std::max(max_coord, std::abs(x[i](0)));
        max_coord = std::max(max_coord, std::abs(x[i](1)));
    }
    // fov = 2 * arctan(max_coord / f)
    // max_coord / f = tan(fov / 2)
    // f = max_coord / tan(fov / 2)
    const double min_fov_radians = min_fov * M_PI / 180.0;
    return max_coord / std::tan(min_fov_radians / 2.0);
}

void RDAbsolutePoseEstimator::generate_models(std::vector<Image> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]];
        Xs[k] = X[sample[k]];
    }

    std::vector<CameraPose> poses;
    std::vector<double> focals;
    std::vector<double> dist;

    p5pfr(xs, Xs, &poses, &focals, &dist);

    models->clear();
    for (size_t i = 0; i < poses.size(); ++i) {
        if (focals[i] < 0)
            continue;

        Camera camera;
        camera.model_id = CameraModelId::SIMPLE_DIVISION;
        camera.width = 0;
        camera.height = 0;
        camera.params = {focals[i], 0.0, 0.0, dist[i]};

        Image image(poses[i], camera);

        if (filter_minimal_sample) {
            // check if all are inliers (since this is an overdetermined problem)
            size_t inlier_count = 0;
            compute_msac_score(image, xs, Xs, opt.max_error * opt.max_error, &inlier_count);
            if (inlier_count < 4) {
                continue;
            }
        }
        models->emplace_back(image);
    }
}

double RDAbsolutePoseEstimator::score_model(const Image &image, size_t *inlier_count) const {
    if (image.camera.focal() < 0) {
        // Invalid focal length, skip this model
        return std::numeric_limits<double>::max();
    }
    double score = compute_msac_score(image, x, X, opt.max_error * opt.max_error, inlier_count);
    if (inlier_scoring) {
        // We do a combined MSAC score and inlier counting for model scoring. For some unknown reason this
        // seems slightly more robust? I have no idea...
        score += static_cast<double>(x.size() - *inlier_count) * opt.max_error * opt.max_error;
    }
    return score;
}

void RDAbsolutePoseEstimator::refine_model(Image *image) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;
    bundle_opt.refine_focal_length = true;
    bundle_opt.refine_principal_point = false;
    bundle_opt.refine_extra_params = true;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    bundle_adjust(x, X, image, bundle_opt);
}

void GeneralizedAbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    models->clear();
    draw_sample(sample_sz, num_pts_camera, &sample, rng);

    for (size_t k = 0; k < sample_sz; ++k) {
        const size_t cam_k = sample[k].first;
        const size_t pt_k = sample[k].second;
        ps[k] = camera_centers[cam_k];
        xs[k] = rig_poses[cam_k].derotate(x[cam_k][pt_k].homogeneous().normalized());
        Xs[k] = X[cam_k][pt_k];
    }
    gp3p(ps, xs, Xs, models);
}

double GeneralizedAbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    const double sq_threshold = opt.max_error * opt.max_error;
    double score = 0;
    *inlier_count = 0;
    size_t cam_inlier_count;
    for (size_t k = 0; k < num_cams; ++k) {
        CameraPose full_pose;
        full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
        full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;

        score += compute_msac_score(full_pose, x[k], X[k], sq_threshold, &cam_inlier_count);
        *inlier_count += cam_inlier_count;
    }
    return score;
}

void GeneralizedAbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;
    generalized_bundle_adjust(x, X, rig_poses, pose, bundle_opt);
}

namespace {

// Sets up the rig bookkeeping shared by the two generalized absolute pose and scale
// estimators: the camera centers, their grouping by coinciding center, and the number of
// correspondences per camera. Returns the total number of correspondences, or zero if the
// scale is not observable at all.
//
// The scale is only observable from correspondences seen from at least two distinct rig
// centers: with a single center scale * p is absorbed by the translation and every scale
// explains the observations equally well. Reporting no data lets RANSAC return without a
// model rather than an arbitrary scale.
size_t setup_scale_estimator_rig(const std::vector<CameraPose> &camera_ext, const std::vector<size_t> &num_pts_camera,
                                 std::vector<Point3D> *camera_centers, std::vector<size_t> *center_group) {
    const size_t num_cams = num_pts_camera.size();
    camera_centers->resize(num_cams);
    for (size_t k = 0; k < num_cams; ++k) {
        (*camera_centers)[k] = camera_ext[k].center();
    }
    group_camera_centers(*camera_centers, center_group);

    size_t num_data = 0;
    size_t observed_group = 0;
    bool found = false;
    bool observable = false;
    for (size_t k = 0; k < num_cams; ++k) {
        num_data += num_pts_camera[k];
        if (num_pts_camera[k] == 0) {
            continue;
        }
        if (!found) {
            observed_group = (*center_group)[k];
            found = true;
        } else if ((*center_group)[k] != observed_group) {
            observable = true;
        }
    }
    return observable ? num_data : 0;
}

} // namespace

GeneralizedAbsolutePoseScaleEstimator::GeneralizedAbsolutePoseScaleEstimator(
    const AbsolutePoseOptions &opt, const std::vector<std::vector<Point2D>> &points2D,
    const std::vector<std::vector<Point3D>> &points3D, const std::vector<CameraPose> &camera_ext)
    : num_cams(points2D.size()), opt(opt), x(points2D), X(points3D), rig_poses(camera_ext) {
    rng = opt.ransac.seed;
    ps.resize(sample_sz);
    xs.resize(sample_sz);
    Xs.resize(sample_sz);
    sample.resize(sample_sz);

    num_pts_camera.resize(num_cams);
    for (size_t k = 0; k < num_cams; ++k) {
        num_pts_camera[k] = points2D[k].size();
    }
    num_data = setup_scale_estimator_rig(camera_ext, num_pts_camera, &camera_centers, &center_group);
}

void GeneralizedAbsolutePoseScaleEstimator::generate_models(std::vector<ScaledCameraPose> *models) {
    models->clear();
    // num_data is zero when the rig cannot constrain the scale, in which case no sample spans
    // two centers and gp4ps would return a pose with an arbitrary scale
    if (num_data < sample_sz) {
        return;
    }
    draw_sample_distinct_centers(sample_sz, num_pts_camera, center_group, &sample, rng);

    for (size_t k = 0; k < sample_sz; ++k) {
        const size_t cam_k = sample[k].first;
        const size_t pt_k = sample[k].second;
        ps[k] = camera_centers[cam_k];
        xs[k] = rig_poses[cam_k].derotate(x[cam_k][pt_k].homogeneous().normalized());
        Xs[k] = X[cam_k][pt_k];
    }
    gp4ps(ps, xs, Xs, &sample_poses, &sample_scales);

    for (size_t k = 0; k < sample_poses.size(); ++k) {
        // Only a positive scale maps the rig onto the 3D points
        if (sample_scales[k] <= 0.0) {
            continue;
        }
        models->emplace_back(sample_poses[k], sample_scales[k]);
    }
}

double GeneralizedAbsolutePoseScaleEstimator::score_model(const ScaledCameraPose &scaled_pose,
                                                          size_t *inlier_count) const {
    const double sq_threshold = opt.max_error * opt.max_error;
    double score = 0;
    *inlier_count = 0;
    size_t cam_inlier_count;
    for (size_t k = 0; k < num_cams; ++k) {
        score += compute_msac_score(scaled_pose.camera_pose(rig_poses[k]), x[k], X[k], sq_threshold, &cam_inlier_count);
        *inlier_count += cam_inlier_count;
    }
    return score;
}

void GeneralizedAbsolutePoseScaleEstimator::refine_model(ScaledCameraPose *scaled_pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;
    generalized_bundle_adjust(x, X, rig_poses, scaled_pose, bundle_opt);
}

BearingGeneralizedAbsolutePoseScaleEstimator::BearingGeneralizedAbsolutePoseScaleEstimator(
    const AbsolutePoseOptions &opt, const std::vector<std::vector<Point3D>> &bearings,
    const std::vector<std::vector<Point3D>> &points3D, const std::vector<CameraPose> &camera_ext)
    : num_cams(bearings.size()), opt(opt), b(bearings), X(points3D), rig_poses(camera_ext) {
    rng = opt.ransac.seed;
    ps.resize(sample_sz);
    xs.resize(sample_sz);
    Xs.resize(sample_sz);
    sample.resize(sample_sz);

    num_pts_camera.resize(num_cams);
    for (size_t k = 0; k < num_cams; ++k) {
        num_pts_camera[k] = bearings[k].size();
    }
    num_data = setup_scale_estimator_rig(camera_ext, num_pts_camera, &camera_centers, &center_group);
}

void BearingGeneralizedAbsolutePoseScaleEstimator::generate_models(std::vector<ScaledCameraPose> *models) {
    models->clear();
    // See GeneralizedAbsolutePoseScaleEstimator::generate_models
    if (num_data < sample_sz) {
        return;
    }
    draw_sample_distinct_centers(sample_sz, num_pts_camera, center_group, &sample, rng);

    for (size_t k = 0; k < sample_sz; ++k) {
        const size_t cam_k = sample[k].first;
        const size_t pt_k = sample[k].second;
        ps[k] = camera_centers[cam_k];
        xs[k] = rig_poses[cam_k].derotate(b[cam_k][pt_k].normalized());
        Xs[k] = X[cam_k][pt_k];
    }
    gp4ps(ps, xs, Xs, &sample_poses, &sample_scales);

    for (size_t k = 0; k < sample_poses.size(); ++k) {
        // Only a positive scale maps the rig onto the 3D points
        if (sample_scales[k] <= 0.0) {
            continue;
        }
        models->emplace_back(sample_poses[k], sample_scales[k]);
    }
}

double BearingGeneralizedAbsolutePoseScaleEstimator::score_model(const ScaledCameraPose &scaled_pose,
                                                                 size_t *inlier_count) const {
    const double sq_threshold = opt.max_error * opt.max_error;
    double score = 0;
    *inlier_count = 0;
    size_t cam_inlier_count;
    for (size_t k = 0; k < num_cams; ++k) {
        score += compute_msac_score_bearing(scaled_pose.camera_pose(rig_poses[k]), b[k], X[k], sq_threshold,
                                            &cam_inlier_count);
        *inlier_count += cam_inlier_count;
    }
    return score;
}

void BearingGeneralizedAbsolutePoseScaleEstimator::refine_model(ScaledCameraPose *scaled_pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;
    generalized_bundle_adjust_bearing(b, X, rig_poses, scaled_pose, bundle_opt);
}

void AbsolutePosePointLineEstimator::generate_models(std::vector<CameraPose> *models) {
    models->clear();
    draw_sample(sample_sz, num_data, &sample, rng);

    size_t pt_idx = 0;
    size_t line_idx = 0;
    for (size_t k = 0; k < sample_sz; ++k) {
        size_t idx = sample[k];
        if (idx < points2D.size()) {
            // we sampled a point correspondence
            xs[pt_idx] = points2D[idx].homogeneous();
            xs[pt_idx].normalize();
            Xs[pt_idx] = points3D[idx];
            pt_idx++;
        } else {
            // we sampled a line correspondence
            idx -= points2D.size();
            ls[line_idx] = lines2D[idx].x1.homogeneous().cross(lines2D[idx].x2.homogeneous());
            ls[line_idx].normalize();
            Cs[line_idx] = lines3D[idx].X1;
            Vs[line_idx] = lines3D[idx].X2 - lines3D[idx].X1;
            Vs[line_idx].normalize();
            line_idx++;
        }
    }

    if (pt_idx == 3 && line_idx == 0) {
        p3p(xs, Xs, models);
    } else if (pt_idx == 2 && line_idx == 1) {
        p2p1ll(xs, Xs, ls, Cs, Vs, models);
    } else if (pt_idx == 1 && line_idx == 2) {
        p1p2ll(xs, Xs, ls, Cs, Vs, models);
    } else if (pt_idx == 0 && line_idx == 3) {
        p3ll(ls, Cs, Vs, models);
    }
}

double AbsolutePosePointLineEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    size_t point_inliers, line_inliers;
    double th_pts, th_lines;
    if (opt.max_errors.size() != 2) {
        th_pts = th_lines = opt.max_error * opt.max_error;
    } else {
        th_pts = opt.max_errors[0] * opt.max_errors[0];
        th_lines = opt.max_errors[1] * opt.max_errors[1];
    }

    double score_pt = compute_msac_score(pose, points2D, points3D, th_pts, &point_inliers);
    double score_l = compute_msac_score(pose, lines2D, lines3D, th_lines, &line_inliers);
    *inlier_count = point_inliers + line_inliers;
    return score_pt + score_l;
}

void AbsolutePosePointLineEstimator::refine_model(CameraPose *pose) const {
    double th_pts, th_lines;
    if (opt.max_errors.size() != 2) {
        th_pts = th_lines = opt.max_error;
    } else {
        th_pts = opt.max_errors[0];
        th_lines = opt.max_errors[1];
    }

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = th_pts;
    bundle_opt.max_iterations = 25;

    BundleOptions line_bundle_opt;
    line_bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    line_bundle_opt.loss_scale = th_lines;

    bundle_adjust(points2D, points3D, lines2D, lines3D, pose, bundle_opt, line_bundle_opt, {}, {});
}

double FocalAbsolutePosePointLineEstimator::compute_max_focal_length(double min_fov) {
    if (min_fov <= 0) {
        return -1;
    }
    double max_coord = 0.0;
    for (size_t i = 0; i < points2D.size(); ++i) {
        max_coord = std::max(max_coord, std::abs(points2D[i](0)));
        max_coord = std::max(max_coord, std::abs(points2D[i](1)));
    }
    for (size_t i = 0; i < lines2D.size(); ++i) {
        max_coord = std::max(max_coord, std::abs(lines2D[i].x1(0)));
        max_coord = std::max(max_coord, std::abs(lines2D[i].x1(1)));
        max_coord = std::max(max_coord, std::abs(lines2D[i].x2(0)));
        max_coord = std::max(max_coord, std::abs(lines2D[i].x2(1)));
    }
    // fov = 2 * arctan(max_coord / f)  =>  f = max_coord / tan(fov / 2)
    const double min_fov_radians = min_fov * M_PI / 180.0;
    return max_coord / std::tan(min_fov_radians / 2.0);
}

void FocalAbsolutePosePointLineEstimator::generate_models(std::vector<Image> *models) {
    models->clear();
    draw_sample(sample_sz, num_data, &sample, rng);

    // Pooled index space: [0, points2D.size()) are points, the remainder are lines.
    // Count the sampled composition first, then size the solver buffers exactly (the focal
    // solvers read input sizes via .size()).
    size_t num_pts = 0;
    for (size_t k = 0; k < sample_sz; ++k) {
        if (sample[k] < points2D.size())
            num_pts++;
    }
    const size_t num_lines = sample_sz - num_pts;

    xs.resize(num_pts);
    Xs.resize(num_pts);
    ls.resize(num_lines);
    Cs.resize(num_lines);
    Vs.resize(num_lines);

    size_t pt_idx = 0, line_idx = 0;
    for (size_t k = 0; k < sample_sz; ++k) {
        size_t idx = sample[k];
        if (idx < points2D.size()) {
            // The focal point solvers take 2D pixel coordinates directly.
            xs[pt_idx] = points2D[idx];
            Xs[pt_idx] = points3D[idx];
            pt_idx++;
        } else {
            idx -= points2D.size();
            // Image line normal (pixel coordinates); uniform normalization keeps the line.
            ls[line_idx] = lines2D[idx].x1.homogeneous().cross(lines2D[idx].x2.homogeneous());
            ls[line_idx].normalize();
            Cs[line_idx] = lines3D[idx].X1;
            Vs[line_idx] = (lines3D[idx].X2 - lines3D[idx].X1).normalized();
            line_idx++;
        }
    }

    std::vector<CameraPose> poses;
    std::vector<double> focals;
    switch (num_pts) {
    case 4:
        p4pf(xs, Xs, &poses, &focals);
        break;
    case 3:
        p3p1llf(xs, Xs, ls, Cs, Vs, &poses, &focals);
        break;
    case 2:
        p2p2llf(xs, Xs, ls, Cs, Vs, &poses, &focals);
        break;
    case 1:
        p1p3llf(xs, Xs, ls, Cs, Vs, &poses, &focals);
        break;
    case 0:
        p4llf(ls, Cs, Vs, &poses, &focals);
        break;
    }

    models->reserve(poses.size());
    for (size_t i = 0; i < poses.size(); ++i) {
        if (focals[i] < 0)
            continue;
        if (max_focal_length >= 0 && focals[i] > max_focal_length)
            continue;

        Camera camera;
        camera.model_id = CameraModelId::SIMPLE_PINHOLE;
        camera.width = 0;
        camera.height = 0;
        camera.params = {focals[i], 0.0, 0.0};
        models->emplace_back(poses[i], camera);
    }
}

double FocalAbsolutePosePointLineEstimator::score_model(const Image &image, size_t *inlier_count) const {
    if (image.camera.focal() < 0 || (max_focal_length > 0 && image.camera.focal() > max_focal_length)) {
        *inlier_count = 0;
        return std::numeric_limits<double>::max();
    }

    double th_pts, th_lines;
    if (opt.max_errors.size() != 2) {
        th_pts = th_lines = opt.max_error * opt.max_error;
    } else {
        th_pts = opt.max_errors[0] * opt.max_errors[0];
        th_lines = opt.max_errors[1] * opt.max_errors[1];
    }

    size_t point_inliers = 0, line_inliers = 0;
    double score_pt = compute_msac_score(image, points2D, points3D, th_pts, &point_inliers);
    double score_l = compute_msac_score(image, lines2D, lines3D, th_lines, &line_inliers);
    *inlier_count = point_inliers + line_inliers;
    return score_pt + score_l;
}

void FocalAbsolutePosePointLineEstimator::refine_model(Image *image) const {
    double th_pts, th_lines;
    if (opt.max_errors.size() != 2) {
        th_pts = th_lines = opt.max_error;
    } else {
        th_pts = opt.max_errors[0];
        th_lines = opt.max_errors[1];
    }

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = th_pts;
    bundle_opt.max_iterations = 25;
    bundle_opt.refine_focal_length = true;
    bundle_opt.refine_principal_point = false;
    bundle_opt.refine_extra_params = false;

    BundleOptions line_bundle_opt = bundle_opt;
    line_bundle_opt.loss_scale = th_lines;

    bundle_adjust(points2D, points3D, lines2D, lines3D, image, bundle_opt, line_bundle_opt, {}, {});
}

void Radial1DAbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    models->clear();
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]].normalized();
        Xs[k] = X[sample[k]];
    }
    p5lp_radial(xs, Xs, models);
}

double Radial1DAbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_msac_score_1D_radial(pose, x, X, opt.max_error * opt.max_error, inlier_count);
}

void Radial1DAbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_error;
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    Camera camera(Radial1DCameraModel::model_id, {0.0, 0.0});
    bundle_adjust_1D_radial(x, X, pose, camera, bundle_opt);
}

} // namespace poselib