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

#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/gp3p.h"
#include "PoseLib/solvers/p1p2ll.h"
#include "PoseLib/solvers/p2p1ll.h"
#include "PoseLib/solvers/p35pf.h"
#include "PoseLib/solvers/p3ll.h"
#include "PoseLib/solvers/p3p.h"
#include "PoseLib/solvers/p4pf.h"
#include "PoseLib/solvers/p5lp_radial.h"
#include "PoseLib/solvers/p5pf.h"
#include "PoseLib/solvers/p5pfr.h"


namespace poselib {

void AbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
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
        camera.model_id = 0;
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
    double score = compute_msac_score(image, x, X, opt.max_error * opt.max_error, inlier_count);
    if (inlier_scoring) {
        // We do a combined MSAC score and inlier counting for model scoring. For some unknown reason this
        // seems slightly more robust? I have no idea...
        score += static_cast<double>(x.size() - *inlier_count) * opt.max_error * opt.max_error;
    }
    if (max_focal_length >= 0 && image.camera.focal() > max_focal_length) {
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

void AbsolutePosePointLineEstimator::generate_models(std::vector<CameraPose> *models) {
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
        th_pts = th_lines = opt.max_error * opt.max_error;
    } else {
        th_pts = opt.max_errors[0] * opt.max_errors[0];
        th_lines = opt.max_errors[1] * opt.max_errors[1];
    }

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = th_pts;
    bundle_opt.max_iterations = 25;

    BundleOptions line_bundle_opt;
    line_bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    line_bundle_opt.loss_scale = th_lines;

    Camera camera;
    camera.model_id = NullCameraModel::model_id;

    bundle_adjust(points2D, points3D, lines2D, lines3D, camera, pose, bundle_opt, line_bundle_opt, {}, {});
}

void Radial1DAbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
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
    Camera camera;
    bundle_adjust_1D_radial(x, X, pose, camera, bundle_opt);
}

} // namespace poselib