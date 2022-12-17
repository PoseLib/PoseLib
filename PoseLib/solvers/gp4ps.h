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

#ifndef POSELIB_GP4PS_H_
#define POSELIB_GP4PS_H_

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Solver the generalized absolute pose and scale problem.
// The solver automagically identifies the quasi-degenerate case where two 3D points coincides,
// and then either calls gp4ps_kukelova or gp4ps_camposeco.
// If you know that you never have duplicate observations (e.g. non-overlapping FoV) you can directly call
// gp4ps_kukelova
int gp4ps(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
          const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output, std::vector<double> *output_scales,
          bool filter_solutions = true);

// Solves for camera pose such that: scale*p+lambda*x = R*X+t
// Re-implementation of the gP4P solver from
//    Kukelova et al., Efficient Intersection of Three Quadrics and Applications in Computer Vision, CVPR 2016
// Note: this impl. assumes that x has been normalized and that the 3D points are distinct!
int gp4ps_kukelova(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                   const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output,
                   std::vector<double> *output_scales, bool filter_solutions = true);

// Solves for camera pose such that: scale*p+lambda*x = R*X+t
// Re-implementation of the gP4P solver from
//    Camposeco et al., Minimal solvers for generalized pose and scale estimation from two rays and one point, ECCV 2016
// Note: This solver assumes that the first two points correspond to the same 3D point!
// This is a minimal problem and it is not possible to filter solutions!
int gp4ps_camposeco(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                    const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output,
                    std::vector<double> *output_scales);

} // namespace poselib

#endif