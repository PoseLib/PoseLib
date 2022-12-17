// Copyright (c) 2020, Pierre Moulon
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

#ifndef POSELIB_RELPOSE_UPRIGHT_PLANAR_3PT_H_
#define POSELIB_RELPOSE_UPRIGHT_PLANAR_3PT_H_

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>

namespace poselib {

/**
 * Three-point algorithm for solving for the essential matrix from bearing
 * vector correspondences assuming upright images.
 * Implementation of [1] section 3.3. Linear 3-point Algorithm
 * Note: this is an approximate solver, not a minimal solver
 *
 * [1] "Fast and Reliable Minimal Relative Pose Estimation under Planar Motion"
 * Sunglok Choi, Jong-Hwan Kim, 2018
 *
 * [2] Street View Goes Indoors: Automatic Pose Estimation From Uncalibrated Unordered Spherical Panoramas.
 * Mohamed Aly and Jean-Yves Bouguet.
 * IEEE Workshop on Applications of Computer Vision (WACV), Colorado, January 2012.
 *
 * Comment [2] and [1] propose both a Direct Linear Method using 3 correspondences.
 * Note that they are using gravity axis and that [1] provides more details about the fact that
 * the linear formulation is not a minimal solver since it cannot enforce the
 * "Pythagorean" identity on sin^2(t) + cos^2(t) = 1
 *
 * Reimplementation from OpenMVG to PoseLib
 */
int relpose_upright_planar_3pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                               CameraPoseVector *output);

}; // namespace poselib

#endif
