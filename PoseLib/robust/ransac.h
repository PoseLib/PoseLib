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
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_RANSAC_H_
#define POSELIB_RANSAC_H_

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <vector>

namespace poselib {

// Absolute pose estimation
RansacStats ransac_pnp(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const RansacOptions &opt,
                       CameraPose *best_model, std::vector<char> *best_inliers);

RansacStats ransac_gen_pnp(const std::vector<std::vector<Point2D>> &x, const std::vector<std::vector<Point3D>> &X,
                           const std::vector<CameraPose> &camera_ext, const RansacOptions &opt, CameraPose *best_model,
                           std::vector<std::vector<char>> *best_inliers);

RansacStats ransac_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                        const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                        const RansacOptions &opt, CameraPose *best_model, std::vector<char> *inliers_points,
                        std::vector<char> *inliers_lines);

// Relative pose estimation
RansacStats ransac_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const RansacOptions &opt,
                           CameraPose *best_model, std::vector<char> *best_inliers);

RansacStats ransac_shared_focal_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                        const RansacOptions &opt, ImagePair *best_model,
                                        std::vector<char> *best_inliers);

RansacStats ransac_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const RansacOptions &opt,
                               Eigen::Matrix3d *best_model, std::vector<char> *best_inliers);

RansacStats ransac_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, const RansacOptions &opt,
                              Eigen::Matrix3d *best_model, std::vector<char> *best_inliers);

RansacStats ransac_gen_relpose(const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
                               const std::vector<CameraPose> &camera2_ext, const RansacOptions &opt,
                               CameraPose *best_model, std::vector<std::vector<char>> *best_inliers);

// Hybrid pose estimation (both 2D-2D and 2D-3D correspondences)
RansacStats ransac_hybrid_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                               const std::vector<PairwiseMatches> &matches2D_2D, const std::vector<CameraPose> &map_ext,
                               const RansacOptions &opt, CameraPose *best_model, std::vector<char> *inliers_2D_3D,
                               std::vector<std::vector<char>> *inliers_2D_2D);

// Absolute pose estimation with the 1D radial camera model
RansacStats ransac_1D_radial_pnp(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const RansacOptions &opt,
                                 CameraPose *best_model, std::vector<char> *best_inliers);

} // namespace poselib

#endif