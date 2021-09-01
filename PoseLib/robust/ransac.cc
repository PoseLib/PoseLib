#include "ransac.h"
#include "bundle.h"

#include "estimators/absolute_pose.h"
#include "estimators/hybrid_pose.h"
#include "estimators/relative_pose.h"

#include <PoseLib/gen_relpose_5p1pt.h>
#include <PoseLib/misc/essential.h>
#include <PoseLib/p3p.h>
#include <PoseLib/relpose_5pt.h>

namespace pose_lib {

// Templated LO-RANSAC implementation (inspired by RansacLib from Torsten Sattler)
template <typename Solver>
RansacStats ransac(Solver &estimator, const RansacOptions &opt, CameraPose *best_model) {
    RansacStats stats;

    if (estimator.num_data < estimator.sample_sz) {
        best_model->R.setIdentity();
        best_model->t.setZero();
        return stats;
    }

    // Score/Inliers for best model found so far
    stats.num_inliers = 0;
    stats.model_score = std::numeric_limits<double>::max();
    size_t best_minimal_inlier_count = 0; // best inl for minimal model, used to decide when to LO

    const double log_prob_missing_model = std::log(1.0 - opt.success_prob);
    size_t inlier_count = 0;
    std::vector<CameraPose> models;
    size_t dynamic_max_iter = opt.max_iterations;
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; stats.iterations++) {

        if (stats.iterations > opt.min_iterations && stats.iterations > dynamic_max_iter) {
            break;
        }
        models.clear();
        estimator.generate_models(&models);

        // Find best model among candidates
        int best_model_ind = -1;
        for (size_t i = 0; i < models.size(); ++i) {
            double score_msac = estimator.score_model(models[i], &inlier_count);

            if (best_minimal_inlier_count < inlier_count) {
                best_minimal_inlier_count = inlier_count;
                best_model_ind = i;

                // check if we should update best model already
                if (score_msac < stats.model_score) {
                    stats.model_score = score_msac;
                    *best_model = models[i];
                    stats.num_inliers = inlier_count;
                }
            }
        }

        if (best_model_ind == -1)
            continue;

        // Refinement
        CameraPose refined_model = models[best_model_ind];
        estimator.refine_model(&refined_model);
        stats.refinements++;
        double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
        if (refined_msac_score < stats.model_score) {
            stats.model_score = refined_msac_score;
            stats.num_inliers = inlier_count;
            *best_model = refined_model;
        }

        // update number of iterations
        stats.inlier_ratio = static_cast<double>(stats.num_inliers) / static_cast<double>(estimator.num_data);
        if (stats.inlier_ratio >= 0.9999) {
            // this is to avoid log(prob_outlier) = -inf below
            dynamic_max_iter = opt.min_iterations;
        } else if (stats.inlier_ratio <= 0.0001) {
            // this is to avoid log(prob_outlier) = 0 below
            dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(stats.inlier_ratio, estimator.sample_sz);
            dynamic_max_iter = std::ceil(log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }
    }

    // Final refinement
    CameraPose refined_model = *best_model;
    estimator.refine_model(&refined_model);
    stats.refinements++;
    double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
    if (refined_msac_score < stats.model_score) {
        *best_model = refined_model;
        stats.num_inliers = inlier_count;
    }

    return stats;
}

RansacStats ransac_pose(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const RansacOptions &opt,
                        CameraPose *best_model, std::vector<char> *best_inliers) {

    AbsolutePoseEstimator estimator(opt, x, X);
    RansacStats stats = ransac<AbsolutePoseEstimator>(estimator, opt, best_model);

    get_inliers(*best_model, x, X, opt.max_reproj_error * opt.max_reproj_error, best_inliers);

    return stats;
}

RansacStats ransac_gen_pose(const std::vector<std::vector<Eigen::Vector2d>> &x, const std::vector<std::vector<Eigen::Vector3d>> &X, const std::vector<CameraPose> &camera_ext, const RansacOptions &opt,
                            CameraPose *best_model, std::vector<std::vector<char>> *best_inliers) {

    GeneralizedAbsolutePoseEstimator estimator(opt, x, X, camera_ext);
    RansacStats stats = ransac<GeneralizedAbsolutePoseEstimator>(estimator, opt, best_model);

    best_inliers->resize(camera_ext.size());
    for (size_t k = 0; k < camera_ext.size(); ++k) {
        CameraPose full_pose;
        full_pose.R = camera_ext[k].R * best_model->R;
        full_pose.t = camera_ext[k].R * best_model->t + camera_ext[k].t;
        get_inliers(full_pose, x[k], X[k], opt.max_reproj_error * opt.max_reproj_error, &(*best_inliers)[k]);
    }

    return stats;
}

RansacStats ransac_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                           const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers) {

    RelativePoseEstimator estimator(opt, x1, x2);
    RansacStats stats = ransac<RelativePoseEstimator>(estimator, opt, best_model);

    get_inliers(*best_model, x1, x2, opt.max_epipolar_error * opt.max_epipolar_error, best_inliers);

    return stats;
}

RansacStats ransac_gen_relpose(const std::vector<PairwiseMatches> &matches,
                               const std::vector<CameraPose> &camera1_ext, const std::vector<CameraPose> &camera2_ext,
                               const RansacOptions &opt, CameraPose *best_model, std::vector<std::vector<char>> *best_inliers) {

    GeneralizedRelativePoseEstimator estimator(opt, matches, camera1_ext, camera2_ext);
    RansacStats stats = ransac<GeneralizedRelativePoseEstimator>(estimator, opt, best_model);

    best_inliers->resize(matches.size());
    for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
        const PairwiseMatches &m = matches[match_k];
        CameraPose pose1 = camera1_ext[m.cam_id1];
        CameraPose pose2 = camera1_ext[m.cam_id2];

        // Apply transform (transforming second rig into the first)
        pose2.t = pose2.t + pose2.R * best_model->t;
        pose2.R = pose2.R * best_model->R;

        // Now the relative poses should be consistent with the pairwise measurements
        CameraPose relpose;
        relpose.R = pose2.R * pose1.R.transpose();
        relpose.t = pose2.t - relpose.R * pose1.t;

        // Compute inliers
        std::vector<char> &inliers = (*best_inliers)[match_k];
        get_inliers(relpose, m.x1, m.x2, (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    }

    return stats;
}

RansacStats ransac_hybrid_pose(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                               const std::vector<PairwiseMatches> &matches2D_2D, const std::vector<CameraPose> &map_ext,
                               const RansacOptions &opt, CameraPose *best_model,
                               std::vector<char> *inliers_2D_3D, std::vector<std::vector<char>> *inliers_2D_2D) {

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
        rel_pose.R = rel_pose.R * map_pose.R.transpose();
        rel_pose.t -= rel_pose.R * map_pose.t;

        std::vector<char> &inliers = (*inliers_2D_2D)[match_k];
        get_inliers(rel_pose, m.x1, m.x2, (opt.max_epipolar_error * opt.max_epipolar_error), &inliers);
    }

    return stats;
}

} // namespace pose_lib