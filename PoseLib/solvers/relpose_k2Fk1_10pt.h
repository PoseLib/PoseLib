//
// Created by kocur on 15-May-24.
//

#ifndef POSELIB_RELPOSE_K2FK1_10PT_H
#define POSELIB_RELPOSE_K2FK1_10PT_H

#include <Eigen/Dense>
#include <vector>

namespace poselib {

struct alignas(32) ProjectiveImagePairWithDivisionCamera {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
    double k1 = 0.0; // radial distortion parameter of camera 1
    double k2 = 0.0; // radial distortion parameter of camera 2
    ProjectiveImagePairWithDivisionCamera(const Eigen::Matrix3d &F, double k1, double k2) : F(F), k1(k1), k2(k2) {}
};

// Solver by Kukelova et al., ICCV 2015
int relpose_k2Fk1_10pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                       std::vector<ProjectiveImagePairWithDivisionCamera> *cam_pairs);

} // namespace poselib

#endif // POSELIB_RELPOSE_K2FK1_10PT_H
