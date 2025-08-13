#ifndef POSELIB_RELPOSE_MONODEPTH_3PT_SHARED_FOCAL_H_
#define POSELIB_RELPOSE_MONODEPTH_3PT_SHARED_FOCAL_H_

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {
// Estimates relative pose when depth estimates are provided for each point and cameras have shared unknown focal using:
// RePoseD: Efficient Relative Pose Estimation With Known Depth Information, Ding et al. (ICCV 2025)
// The norm of translation is set so that it provides the relative scale of the two depth estimates.
int relpose_monodepth_3pt_shared_focal(const std::vector<Eigen::Vector3d> &x1h,
                                        const std::vector<Eigen::Vector3d> &x2h,
                                        const std::vector<double> &d1, const std::vector<double> &d2,
                                        std::vector<ImagePair> *models);
} // namespace poselib

#endif