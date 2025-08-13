#ifndef POSELIB_RELPOSE_MONODEPTH_4PT_VARYING_FOCAL_H_
#define POSELIB_RELPOSE_MONODEPTH_4PT_VARYING_FOCAL_H_

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {
void relpose_monodepth_4pt_varying_focal(const std::vector<Eigen::Vector3d> &x1h,
                                         const std::vector<Eigen::Vector3d> &x2h,
                                         const std::vector<double> &depth1, const std::vector<double> &depth2,
                                         std::vector<ImagePair> *models);
} // namespace poselib

#endif