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

#ifndef POSELIB_ROBUST_UTILS_H
#define POSELIB_ROBUST_UTILS_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Returns MSAC score of the reprojection error
double compute_msac_score(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                          double sq_threshold, size_t *inlier_count);
double compute_msac_score(const CameraPose &pose, const std::vector<Line2D> &lines2D,
                          const std::vector<Line3D> &lines3D, double sq_threshold, size_t *inlier_count);
// Returns MSAC score of the Sampson error
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Point2D> &x1,
                                  const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);
double compute_sampson_msac_score(const Eigen::Matrix3d &F, const std::vector<Point2D> &x1,
                                  const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);

// Returns MSAC score of transfer error for homography
double compute_homography_msac_score(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1,
                                     const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);

// Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
void get_inliers(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                 double sq_threshold, std::vector<char> *inliers);
void get_inliers(const CameraPose &pose, const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                 double sq_threshold, std::vector<char> *inliers);

// Compute inliers for relative pose estimation (using Sampson error)
int get_inliers(const CameraPose &pose, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                double sq_threshold, std::vector<char> *inliers);
int get_inliers(const Eigen::Matrix3d &E, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                double sq_threshold, std::vector<char> *inliers);

// inliers for homography
void get_homography_inliers(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                            double sq_threshold, std::vector<char> *inliers);

// Helpers for the 1D radial camera model
double compute_msac_score_1D_radial(const CameraPose &pose, const std::vector<Point2D> &x,
                                    const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count);
void get_inliers_1D_radial(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                           double sq_threshold, std::vector<char> *inliers);

// Normalize points by shifting/scaling coordinate systems.
double normalize_points(std::vector<Eigen::Vector2d> &x1, std::vector<Eigen::Vector2d> &x2, Eigen::Matrix3d &T1,
                        Eigen::Matrix3d &T2, bool normalize_scale, bool normalize_centroid, bool shared_scale);

// Calculate whether F would yield real focals, assumes both pp at [0, 0]
bool calculate_RFC(const Eigen::Matrix3d &F);

} // namespace poselib

#endif