
#ifndef POSELIB_RELPOSE_3PT_REL_DEPTH_H_
#define POSELIB_RELPOSE_3PT_REL_DEPTH_H_

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

int essential_3pt_relative_depth(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                            const std::vector<double> &sigma, std::vector<CameraPose> *rel_pose,
                            bool all_permutations = true);


}; // namespace poselib

#endif