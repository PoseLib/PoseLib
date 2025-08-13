#ifndef POSELIB_RELPOSE_3PT_MONODEPTH_H
#define POSELIB_RELPOSE_3PT_MONODEPTH_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/univariate.h"
#include <Eigen/Dense>
#include <vector>

namespace poselib {
int relpose_3pt_monodepth(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                          const std::vector<double> &d1, const std::vector<double> &d2,
                          std::vector<CameraPose> *rel_pose);
}

#endif // POSELIB_RELPOSE_3PT_MONODEPTH_H
