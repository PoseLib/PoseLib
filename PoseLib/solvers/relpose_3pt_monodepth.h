#ifndef POSELIB_RELPOSE_3PT_MONODEPTH_H
#define POSELIB_RELPOSE_3PT_MONODEPTH_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/univariate.h"
#include <Eigen/Dense>
#include <vector>

namespace poselib {
// Estimates relative pose when depth estimates are provided for each point using:
// RePoseD: Efficient Relative Pose Estimation With Known Depth Information, Ding et al. (ICCV 2025)
// The solver also estimates shift, but this is currently discarded.
// The norm of translation is set so that it provides the relative scale of the two depth estimates.
int relpose_3pt_monodepth(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                          const std::vector<double> &d1, const std::vector<double> &d2,
                          std::vector<CameraPose> *rel_pose);
}

#endif // POSELIB_RELPOSE_3PT_MONODEPTH_H
