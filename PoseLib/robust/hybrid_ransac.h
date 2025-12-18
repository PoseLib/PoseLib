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
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// High-level hybrid RANSAC functions for absolute pose estimation.

#pragma once

#include <PoseLib/camera_pose.h>
#include <PoseLib/types.h>
#include <vector>

namespace poselib {

// Hybrid RANSAC for absolute pose from points and lines (PnPL).
// Uses adaptive solver selection between P3P, P2P1LL, P1P2LL, P3LL.
//
// Arguments:
//   points2D, points3D: 2D-3D point correspondences
//   lines2D, lines3D: 2D-3D line correspondences (endpoint representation)
//   opt: Hybrid RANSAC options
//        - max_errors[0]: point reprojection error threshold in pixels
//        - max_errors[1]: line error threshold in pixels
//   pose: Output camera pose
//   inliers_points, inliers_lines: Output inlier masks
//
// Returns: HybridRansacStats with iteration counts and inlier info
HybridRansacStats hybrid_ransac_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                     const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                                     const HybridRansacOptions &opt, CameraPose *pose,
                                     std::vector<char> *inliers_points, std::vector<char> *inliers_lines);

} // namespace poselib
