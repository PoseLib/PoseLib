#ifndef POSELIB_RANSAC_H_
#define POSELIB_RANSAC_H_

#include "../types.h"
#include <vector>

namespace pose_lib {

struct RansacOptions {
    size_t max_iterations = 100000;
    size_t min_iterations = 1000;
    double dyn_num_trials_mult = 3.0;
    double success_prob = 0.9999;
    double max_reproj_error = 12.0;
    unsigned long seed = 0;
};

int ransac_pose(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers);

int ransac_gen_pose(const std::vector<std::vector<Eigen::Vector2d>> &x,
                    const std::vector<std::vector<Eigen::Vector3d>> &X,
                    const std::vector<CameraPose> &camera_ext, const RansacOptions &opt,
                    CameraPose *best_model, std::vector<std::vector<char>> *best_inliers);

int ransac_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                   const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers);

} // namespace pose_lib

#endif