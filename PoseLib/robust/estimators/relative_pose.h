#ifndef POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H

#include <PoseLib/robust/types.h>
#include <PoseLib/robust/utils.h>
#include <PoseLib/types.h>

namespace pose_lib {

class RelativePoseEstimator {
  public:
    RelativePoseEstimator(const RansacOptions &ransac_opt,
                          const std::vector<Eigen::Vector2d> &points2D_1,
                          const std::vector<Eigen::Vector2d> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Eigen::Vector2d> &x1;
    const std::vector<Eigen::Vector2d> &x2;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class GeneralizedRelativePoseEstimator {
  public:
    GeneralizedRelativePoseEstimator(const RansacOptions &ransac_opt,
                                     const std::vector<PairwiseMatches> &pairwise_matches,
                                     const std::vector<CameraPose> &camera1_ext,
                                     const std::vector<CameraPose> &camera2_ext)
        : opt(ransac_opt), matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        p1s.resize(sample_sz);
        p2s.resize(sample_sz);
        sample.resize(sample_sz);

        num_data = 0;
        for (const PairwiseMatches &m : matches) {
            num_data += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 6;
    size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s, p1s, p2s;
    std::vector<size_t> sample;
};

class FundamentalEstimator {
  public:
    FundamentalEstimator(const RansacOptions &ransac_opt,
                         const std::vector<Eigen::Vector2d> &points2D_1,
                         const std::vector<Eigen::Vector2d> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<Eigen::Matrix3d> *models);
    double score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const;
    void refine_model(Eigen::Matrix3d *F) const;

    const size_t sample_sz = 7;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Eigen::Vector2d> &x1;
    const std::vector<Eigen::Vector2d> &x2;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

} // namespace pose_lib

#endif