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

#pragma once

#include "PoseLib/camera_pose.h"
#include "PoseLib/types.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Returns MSAC score of the reprojection error
double compute_msac_score(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                          double sq_threshold, size_t *inlier_count);
double compute_msac_score(const Image &image, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                          double sq_threshold, size_t *inlier_count);

double compute_msac_score(const CameraPose &pose, const std::vector<Line2D> &lines2D,
                          const std::vector<Line3D> &lines3D, double sq_threshold, size_t *inlier_count);
// MSAC score of the reprojection error on projected 3D points
double compute_msac_score(const CameraPose &pose, double focal, const std::vector<Point2D> &x,
                          const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count);

// Returns MSAC score of the Sampson error
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Point2D> &x1,
                                  const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);
double compute_sampson_msac_score(const Eigen::Matrix3d &F, const std::vector<Point2D> &x1,
                                  const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);

// Bearing-vector variants for central camera models (pinhole, spherical, fisheye, ...).
// The inputs are 3D unit (or near-unit) bearing vectors in camera space; for spherical
// cameras these preserve hemisphere information (sign(z)) that the 2D Point2D form loses.
// For pinhole bearings these are first-order equivalent to the 2D Point2D variants —
// they share the same minimum in the noise-free limit but differ by O(error^3).
//
// compute_msac_score_bearing (absolute pose): residual is the chord distance squared
// between the observed unit bearing and the predicted bearing normalize(R*X + t).
// Cheirality is enforced bearing-natively as b_pred . b_obs > 0 (the spherical
// replacement for the pinhole Z(2) > 0 check); back-hemisphere features remain valid
// as long as observed and predicted bearings agree on sign.
//
// sq_threshold is the squared chord-distance threshold in unit-bearing coordinates.
// Bearing-estimator entry points (estimate_absolute_pose_bearings) accept an angular
// threshold in radians and convert it internally to chord = 2*sin(angle/2).
double compute_msac_score_bearing(const CameraPose &pose, const std::vector<Point3D> &bearings,
                                  const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count);

// compute_sampson_msac_score_bearing (relative pose): unit-norm symmetric Sampson on
// the sphere
//   r = (b2^T E b1) / sqrt(|E b1|^2 + |E^T b2|^2)
// the asymptotic perpendicular angular distance to the epipolar great circles. For
// pinhole bearings this reduces to the standard 2D Sampson formula once bearings
// are made unit.
//
// Cheirality is enforced via the existing check_cheirality(pose, b1, b2) overload,
// which operates directly on unit bearings. sq_threshold is in squared residual units
// (sin^2(angle), approximately radians^2 in the small-error limit).
double compute_sampson_msac_score_bearing(const CameraPose &pose, const std::vector<Point3D> &bearings1,
                                          const std::vector<Point3D> &bearings2, double sq_threshold,
                                          size_t *inlier_count, bool check_cheirality_flag = true);

// Bearing-vector inlier selection for absolute pose (chord-distance threshold,
// bearing-native cheirality b_pred . b_obs > 0). Threshold is sq_threshold in
// chord-distance units.
void get_inliers_abs_bearing(const CameraPose &pose, const std::vector<Point3D> &bearings,
                             const std::vector<Point3D> &X, double sq_threshold, std::vector<char> *inliers);

// Bearing-vector inlier selection for relative pose (unit-norm symmetric Sampson,
// same form as compute_sampson_msac_score_bearing). Returns the number of inliers.
// Optional cheirality check via check_cheirality(pose, b1, b2).
int get_inliers_rel_bearing(const CameraPose &pose, const std::vector<Point3D> &bearings1,
                            const std::vector<Point3D> &bearings2, double sq_threshold, std::vector<char> *inliers,
                            bool check_cheirality_flag = true);

// Returns MSAC score for the Tangent Sampson error (Terekhov and Larsson, CVPR 2023)
double compute_tangent_sampson_msac_score(const Eigen::Matrix3d &F, const std::vector<Point2D> &x1,
                                          const std::vector<Point2D> &x2, const Camera &cam1, const Camera &cam2,
                                          double sq_threshold, size_t *inlier_count);
// Returns MSAC score for the Tangent Sampson error (Terekhov and Larsson, CVPR 2023)
// with pre-computed unprojections and unprojection jacobians
double compute_tangent_sampson_msac_score(const CameraPose &pose, const std::vector<Point3D> &d1,
                                          const std::vector<Point3D> &d2,
                                          const std::vector<Eigen::Matrix<double, 3, 2>> &M1,
                                          const std::vector<Eigen::Matrix<double, 3, 2>> &M2, double sq_threshold,
                                          size_t *inlier_count);

// Returns MSAC score of transfer error for homography
double compute_homography_msac_score(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1,
                                     const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);

// Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
void get_inliers(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                 double sq_threshold, std::vector<char> *inliers);
void get_inliers(const Image &image, const std::vector<Point2D> &x, const std::vector<Point3D> &X, double sq_threshold,
                 std::vector<char> *inliers);
void get_inliers(const CameraPose &pose, const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                 double sq_threshold, std::vector<char> *inliers);
// Compute inliers for relative pose with monodepth by using reprojection error
void get_inliers(const CameraPose &pose, double focal, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                 double sq_threshold, std::vector<char> *inliers);

// Compute inliers for relative pose estimation (using Sampson error)
int get_inliers(const CameraPose &pose, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                double sq_threshold, std::vector<char> *inliers);
int get_inliers(const Eigen::Matrix3d &E, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                double sq_threshold, std::vector<char> *inliers);

// Compute inliers for relative pose + distortion estimation using Tangent Sampson Error (Terekhov and Larsson, CVPR
// 2023)
int get_tangent_sampson_inliers(const Eigen::Matrix3d &F, const Camera &cam1, const Camera &cam2,
                                const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, double sq_threshold,
                                std::vector<char> *inliers);
int get_tangent_sampson_inliers(const CameraPose &pose, const std::vector<Point3D> &d1, const std::vector<Point3D> &d2,
                                const std::vector<Eigen::Matrix<double, 3, 2>> &M1,
                                const std::vector<Eigen::Matrix<double, 3, 2>> &M2, double sq_threshold,
                                std::vector<char> *inliers);

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