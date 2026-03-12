//
// Created by kocur on 15-May-24.
//

#ifndef POSELIB_RELPOSE_KFK_9PT_H
#define POSELIB_RELPOSE_KFK_9PT_H

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Computes the fundamental matrix and k for division model from 9 point correspondences.
int relpose_kFk_9pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                    std::vector<ProjectiveImagePair> *F_cam);

}; // namespace poselib

#endif