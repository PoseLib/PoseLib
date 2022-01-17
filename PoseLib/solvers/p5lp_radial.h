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

#ifndef POSELIB_P5LP_RADIAL_H_
#define POSELIB_P5LP_RADIAL_H_

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Solves for camera pose such that: l'*(R*X+t) = 0
// Assumes that all lines pass through the image center, i.e. l = [l1,l2,0]
// This is equivalent to the 1D Radial pose solver from
//   Kukelova et al., Real-Time Solution to the Absolute Pose Problem with Unknown Radial Distortion and Focal Length,
//   ICCV 2013
// Converting the 2D points to lines l = [-y,x,0]
// Note that this solver always returns tz = 0 since it is not observable from these constraints.
int p5lp_radial(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                std::vector<CameraPose> *output);

// Helper function using the 2D points. Corrects the sign of the camera using the first point correspondence.
int p5lp_radial(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                std::vector<CameraPose> *output);

} // namespace poselib

#endif