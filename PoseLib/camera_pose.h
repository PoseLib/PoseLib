// Copyright (c) 2020, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_CAMERA_POSE_H_
#define POSELIB_CAMERA_POSE_H_

#include "PoseLib/misc/quaternion.h"
#include "alignment.h"
#include "misc/colmap_models.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

struct alignas(32) CameraPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Rotation is represented as a unit quaternion
    // with real part first, i.e. QW, QX, QY, QZ
    Eigen::Vector4d q;
    Eigen::Vector3d t;

    // Constructors (Defaults to identity camera)
    CameraPose() : q(1.0, 0.0, 0.0, 0.0), t(0.0, 0.0, 0.0) {}
    CameraPose(const Eigen::Vector4d &qq, const Eigen::Vector3d &tt) : q(qq), t(tt) {}
    CameraPose(const Eigen::Matrix3d &R, const Eigen::Vector3d &tt) : q(rotmat_to_quat(R)), t(tt) {}

    // Helper functions
    inline Eigen::Matrix3d R() const { return quat_to_rotmat(q); }
    inline Eigen::Matrix<double, 3, 4> Rt() const {
        Eigen::Matrix<double, 3, 4> tmp;
        tmp.block<3, 3>(0, 0) = quat_to_rotmat(q);
        tmp.col(3) = t;
        return tmp;
    }
    inline Eigen::Vector3d rotate(const Eigen::Vector3d &p) const { return quat_rotate(q, p); }
    inline Eigen::Vector3d derotate(const Eigen::Vector3d &p) const { return quat_rotate(quat_conj(q), p); }
    inline Eigen::Vector3d apply(const Eigen::Vector3d &p) const { return rotate(p) + t; }

    inline Eigen::Vector3d center() const { return -derotate(t); }
};

typedef std::vector<CameraPose> CameraPoseVector;

struct alignas(32) Image {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Struct simply holds information about camera and its pose
    CameraPose pose;
    Camera camera;

    // Constructors (Defaults to identity camera and pose)
    Image() : pose(CameraPose()), camera(Camera()) {}
    Image(CameraPose pose, Camera camera) : pose(std::move(pose)), camera(std::move(camera)) {}
};

typedef std::vector<Image> ImageVector;

struct alignas(32) ImagePair {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    // Struct simply holds information about two cameras and their relative pose
    CameraPose pose;
    Camera camera1;
    Camera camera2;

    // Constructors (Defaults to identity camera and poses)
    ImagePair() : pose(CameraPose()), camera1(Camera()), camera2(Camera()) {}
    ImagePair(CameraPose pose, Camera camera1, Camera camera2)
        : pose(std::move(pose)), camera1(std::move(camera1)), camera2(std::move(camera2)) {}
};

typedef std::vector<ImagePair> ImagePairVector;

} // namespace poselib

#endif
