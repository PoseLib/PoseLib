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

#pragma once

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Solves for camera pose and focal lengths (fx,fy) from 3 point + 1 line correspondences:
//   lambda * diag(1/fx, 1/fy, 1) * [xp; 1] = R * Xp + t  (points)
//   l^T * (R * (X + mu*V) + t) = 0                        (lines)
// Returns separate fx, fy. If filter_solutions is true, only returns solutions with positive focal length.
int p3p1llf(const std::vector<Eigen::Vector2d> &xp, const std::vector<Eigen::Vector3d> &Xp,
            const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
            const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output, std::vector<double> *output_fx,
            std::vector<double> *output_fy, bool filter_solutions = true);

// Wrapper that returns the average focal length instead.
// filter_solutions also removes instances where the aspect ratio (fx/fy) is far from 1
int p3p1llf(const std::vector<Eigen::Vector2d> &xp, const std::vector<Eigen::Vector3d> &Xp,
            const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
            const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output, std::vector<double> *output_focal,
            bool filter_solutions = true);

} // namespace poselib
