#include "hybrid_pose.h"
#include <PoseLib/gp3p.h>
#include <PoseLib/p3p.h>
#include <PoseLib/robust/bundle.h>

namespace pose_lib {

void HybridPoseEstimator::generate_models(std::vector<CameraPose> *models) {
    draw_sample(sample_sz, num_data, &sample, rng);
    for (size_t k = 0; k < sample_sz; ++k) {
        xs[k] = x[sample[k]].homogeneous().normalized();
        Xs[k] = X[sample[k]];
    }
    p3p(xs, Xs, models);
    // TODO: actual hybrid sampling (we have p2p2pl and 5+1 gen-relpose already implemented, should be enough)
}

double HybridPoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    double score = compute_msac_score(pose, x, X, opt.max_reproj_error * opt.max_reproj_error, inlier_count);

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
        score += compute_sampson_msac_score(rel_pose, m.x1, m.x2, opt.max_epipolar_error * opt.max_epipolar_error, &inliers_2d2d);
        *inlier_count += inliers_2d2d;
    }

    return score;
}

void HybridPoseEstimator::refine_model(CameraPose *pose) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    refine_hybrid_pose(x, X, matches, map_poses, pose, bundle_opt, opt.max_epipolar_error);
}

} // namespace pose_lib