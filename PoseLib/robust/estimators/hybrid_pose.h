#ifndef POSELIB_ROBUST_ESTIMATORS_HYBRID_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_HYBRID_POSE_H

#include "../../camera_pose.h"
#include "../../types.h"
#include "../sampling.h"
#include "../utils.h"

namespace pose_lib {

class HybridPoseEstimator {
  public:
    HybridPoseEstimator(const RansacOptions &ransac_opt,
                        const std::vector<Point2D> &points2D,
                        const std::vector<Point3D> &points3D,
                        const std::vector<PairwiseMatches> &pairwise_matches,
                        const std::vector<CameraPose> &map_ext)
        : opt(ransac_opt), x(points2D), X(points3D), matches(pairwise_matches), map_poses(map_ext) {
        rng = opt.seed;
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
        num_data = points2D.size();
        for (const PairwiseMatches &m : matches) {
            num_data += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3;
    size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &map_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Point3D> xs, Xs;
    std::vector<size_t> sample;
};

} // namespace pose_lib

#endif