#include "absolute_pose.h"
#include <PoseLib/gp3p.h>
#include <PoseLib/p3ll.h>
#include <PoseLib/p1p2ll.h>
#include <PoseLib/p2p1ll.h>
#include <PoseLib/p3p.h>
#include <PoseLib/p5lp_radial.h>
#include <PoseLib/robust/bundle.h>

namespace pose_lib {

void AbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    draw_sample(sample_sz, num_data, &sample, rng);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]].homogeneous().normalized();
        Xs[k] = X[sample[k]];
    }
    p3p(xs, Xs, models);
}

double AbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_msac_score(pose, x, X, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
}

void AbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    bundle_adjust(x, X, pose, bundle_opt);
}

void GeneralizedAbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    draw_sample(sample_sz, num_pts_camera, &sample, rng);

    for (size_t k = 0; k < sample_sz; ++k) {
        const size_t cam_k = sample[k].first;
        const size_t pt_k = sample[k].second;
        ps[k] = camera_centers[cam_k];
        xs[k] = rig_poses[cam_k].R.transpose() * (x[cam_k][pt_k].homogeneous().normalized());
        Xs[k] = X[cam_k][pt_k];
    }
    gp3p(ps, xs, Xs, models);
}

double GeneralizedAbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    const double sq_threshold = opt.max_reproj_error * opt.max_reproj_error;
    double score = 0;
    *inlier_count = 0;
    size_t cam_inlier_count;
    for (size_t k = 0; k < num_cams; ++k) {
        CameraPose full_pose;
        full_pose.R = rig_poses[k].R * pose.R;
        full_pose.t = rig_poses[k].R * pose.t + rig_poses[k].t;

        score += compute_msac_score(full_pose, x[k], X[k], sq_threshold, &cam_inlier_count);
        *inlier_count += cam_inlier_count;
    }
    return score;
}

void GeneralizedAbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;
    generalized_bundle_adjust(x, X, rig_poses, pose, bundle_opt);
}

void AbsolutePosePointLineEstimator::generate_models(std::vector<CameraPose> *models) {
    draw_sample(sample_sz, num_data, &sample, rng);

    size_t pt_idx = 0;
    size_t line_idx = 0;
    for (size_t k = 0; k < sample_sz; ++k) {
        size_t idx = sample[k];
        if(idx < points2D.size()) {
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

    if(pt_idx == 3 && line_idx == 0) {
        p3p(xs,Xs,models);
    } else if(pt_idx == 2 && line_idx == 1) {
        p2p1ll(xs,Xs,ls,Cs,Vs,models);
    } else if(pt_idx == 1 && line_idx == 2) {
        p1p2ll(xs,Xs,ls,Cs,Vs,models);
    } else if(pt_idx == 0 && line_idx == 3) {
        p3ll(ls,Cs,Vs,models);
    }
}

double AbsolutePosePointLineEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    size_t point_inliers, line_inliers;
    double score_pt = compute_msac_score(pose, points2D, points3D, opt.max_reproj_error * opt.max_reproj_error, &point_inliers);
    double score_l = compute_msac_score(pose, lines2D, lines3D, opt.max_reproj_error * opt.max_reproj_error, &line_inliers);
    *inlier_count = point_inliers + line_inliers;
    return score_pt + score_l;
}

void AbsolutePosePointLineEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    bundle_adjust(points2D, points3D, lines2D, lines3D, pose, bundle_opt);
}

void Radial1DAbsolutePoseEstimator::generate_models(std::vector<CameraPose> *models) {
    draw_sample(sample_sz, num_data, &sample, rng);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]].normalized();
        Xs[k] = X[sample[k]];
    }
    p5lp_radial(xs, Xs, models);
}

double Radial1DAbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    return compute_msac_score_1D_radial(pose, x, X, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
}

void Radial1DAbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set

    bundle_adjust_1D_radial(x, X, pose, bundle_opt);
}

} // namespace pose_lib