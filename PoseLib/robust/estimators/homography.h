#ifndef POSELIB_ROBUST_ESTIMATORS_HOMOGRAPHY_H
#define POSELIB_ROBUST_ESTIMATORS_HOMOGRAPHY_H

#include <PoseLib/robust/types.h>
#include <PoseLib/robust/utils.h>
#include <PoseLib/types.h>

namespace pose_lib {

class HomographyEstimator {
  public:
    HomographyEstimator(const RansacOptions &ransac_opt,
                          const std::vector<Point2D> &points2D_1,
                          const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<Eigen::Matrix3d> *models);
    double score_model(const Eigen::Matrix3d &H, size_t *inlier_count) const;
    void refine_model(Eigen::Matrix3d *H) const;

    const size_t sample_sz = 4;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

}

#endif