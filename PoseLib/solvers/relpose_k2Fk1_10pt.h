//
// Created by kocur on 15-May-24.
//

#ifndef POSELIB_RELPOSE_K2FK1_10PT_H
#define POSELIB_RELPOSE_K2FK1_10PT_H

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {
    int relpose_k2Fk1_10pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                           std::vector<FCamPair> *F_cam_pair);
}

#endif //POSELIB_RELPOSE_K2FK1_10PT_H
