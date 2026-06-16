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

// Solves for camera pose and focal lengths (fx,fy) from 4 line correspondences:
//   l^T * (R * (X + mu*V) + t) = 0
// with unknown focal: l is defined in pixel coordinates with diag(fx,fy,1) calibration.
// Returns separate fx, fy. If filter_solutions is true, only returns solutions with positive focal length.
int p4llf(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
          const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output, std::vector<double> *output_fx,
          std::vector<double> *output_fy, bool filter_solutions = true);

// Wrapper that returns the average focal length instead of separate fx and fy.
// When filter_solutions is true, solutions are first filtered to have positive focal
// length and, if multiple valid solutions remain, the implementation selects a single
// solution based on the aspect ratio fx/fy. In that case, the returned pose and focal
// vectors correspond only to this selected solution, unlike in the unfiltered mode
// where all valid solutions are returned.
int p4llf(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
          const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output, std::vector<double> *output_focal,
          bool filter_solutions = true);

} // namespace poselib
