// Copyright (c) 2021, Viktor Larsson
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

// Solves for generalized relative pose from 6 correspondences
// The solver is created using Larsson et al. CVPR 2017
int gen_relpose_6pt(const std::vector<Eigen::Vector3d> &p1, const std::vector<Eigen::Vector3d> &x1,
                    const std::vector<Eigen::Vector3d> &p2, const std::vector<Eigen::Vector3d> &x2,
                    std::vector<CameraPose> *output);

// Solves the structure-less resection 3+3 problem from normalized bearing
// correspondences against two known reference cameras in a shared world frame.
// The first reference camera contributes 3 correspondences, and the second
// reference camera contributes 3 correspondences.
std::vector<CameraPose> structureless_resection_33(const std::vector<Eigen::Vector3d> &x_query1,
                                                   const std::vector<Eigen::Vector3d> &x_ref1,
                                                   const CameraPose &pose_ref1,
                                                   const std::vector<Eigen::Vector3d> &x_query2,
                                                   const std::vector<Eigen::Vector3d> &x_ref2,
                                                   const CameraPose &pose_ref2);

// Solves the structure-less resection problem from 6 normalized bearing
// correspondences against 6 known reference cameras in a shared world frame.
std::vector<CameraPose> structureless_resection_6pt(const std::vector<Eigen::Vector3d> &x_query,
                                                    const std::vector<Eigen::Vector3d> &x_ref,
                                                    const std::vector<CameraPose> &pose_ref);

}; // namespace poselib
