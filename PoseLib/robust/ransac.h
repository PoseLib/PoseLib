#ifndef POSELIB_RANSAC_H_
#define POSELIB_RANSAC_H_

#include <vector>
#include "types.h"

namespace pose_lib {

struct RansacOptions {
    size_t max_iterations = 100000;
    size_t min_iterations = 1000;
    double dyn_num_trials_mult = 3.0;
    double success_prob = 0.9999;
    double max_reproj_error = 12.0;    
    unsigned long seed = 0;
};

int ransac_pose(const std::vector<Eigen::Vector2d>& x, const std::vector<Eigen::Vector3d>& X,
               const RansacOptions& opt, CameraPose* best_model, std::vector<char>* best_inliers);

} // namespace pose_lib

#endif