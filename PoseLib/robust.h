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

#ifndef POSELIB_ROBUST_H_
#define POSELIB_ROBUST_H_

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/camera_models.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/robust/ransac.h"
#include "PoseLib/types.h"

#include <vector>

namespace poselib {

// Estimates absolute pose using LO-RANSAC followed by non-linear refinement
// Threshold for reprojection error is set by RansacOptions.max_reproj_error
// If ransac_opt.estimate_focal, the camera in image.camera should contain correct principal point
RansacStats estimate_absolute_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                   AbsolutePoseOptions opt, Image *image, std::vector<char> *inliers);

// Estimates absolute pose from 3D unit bearing vectors (any central camera model:
// pinhole, spherical / equirectangular, fisheye, ...). Uses LO-RANSAC followed by
// non-linear refinement. Scoring is chord-distance squared on the unit sphere.
// Cheirality is enforced bearing-natively as b_pred . b_obs > 0 (the spherical
// replacement for the pinhole Z(2) > 0 check); back-hemisphere features remain
// valid as long as observed and predicted bearings agree on sign.
//
// opt.max_error is interpreted as an angular threshold in radians: the bearing
// estimator converts it internally to the chord-distance units used by the
// scorer. For pinhole bearings this path is first-order equivalent to
// estimate_absolute_pose(Point2D, ...) when bearings come from
// normalize((X/Z, Y/Z, 1)) — the chord and pixel-plane reprojection objectives
// share the same minimum in the noise-free limit but differ by O(error^3) and
// produce slightly different LM iterates. The bearing path is the
// geometrically correct formulation for non-pinhole central cameras.
RansacStats estimate_absolute_pose_bearings(const std::vector<Point3D> &bearings, const std::vector<Point3D> &points3D,
                                            const AbsolutePoseOptions &opt, CameraPose *pose,
                                            std::vector<char> *inliers);

// Estimates generalized absolute pose using LO-RANSAC followed by non-linear refinement
// Threshold for reprojection error is set by RansacOptions.max_reproj_error
RansacStats estimate_generalized_absolute_pose(const std::vector<std::vector<Point2D>> &points2D,
                                               const std::vector<std::vector<Point3D>> &points3D,
                                               const std::vector<CameraPose> &camera_ext,
                                               const std::vector<Camera> &cameras, const AbsolutePoseOptions &opt,
                                               CameraPose *pose, std::vector<std::vector<char>> *inliers);

// Estimates absolute pose using LO-RANSAC followed by non-linear refinement
// using both 2D-3D point and line matches
// Note that line segments are described by their endpoints
// Threshold for point reprojection error is set by RansacOptions.max_reproj_error
// and for lines the threshold is set by RansacOptions.max_epipolar_error
RansacStats estimate_absolute_pose_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                        const std::vector<Line2D> &line2D, const std::vector<Line3D> &line3D,
                                        const Camera &camera, const AbsolutePoseOptions &opt, CameraPose *pose,
                                        std::vector<char> *inliers_points, std::vector<char> *inliers_lines);

// Estimates relative pose using LO-RANSAC followed by non-linear refinement
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
RansacStats estimate_relative_pose(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                   const Camera &camera1, const Camera &camera2, const RelativePoseOptions &opt,
                                   CameraPose *relative_pose, std::vector<char> *inliers);

// Estimates relative pose from 3D unit bearing vectors (any central camera model:
// pinhole, spherical / equirectangular, fisheye, ...). Uses LO-RANSAC followed by
// non-linear refinement. Scoring is unit-norm symmetric Sampson on the sphere
//   r = (b2^T E b1) / sqrt(|E b1|^2 + |E^T b2|^2)
// the asymptotic perpendicular angular distance to the epipolar great circles.
// This reduces to the standard 2D Sampson for pinhole bearings (b.z = 1) once
// bearings are made unit, and generalizes naturally to any central camera.
//
// Cheirality is checked by default via check_cheirality(pose, b1, b2), which is
// bearing-native: it asserts that the midpoint-triangulation parameters along
// each bearing ray are positive (i.e. the 3D point lies in the observed direction
// along both bearings). This works for ANY unit bearing — including back-hemisphere
// spherical features where camera-space z < 0 — because the test is about ray
// direction, not a z-sign check. Without cheirality the four (R, ±t), (R', ±t)
// decompositions of the essential matrix produce identical Sampson scores, so
// RANSAC would pick whichever one the 5-point solver returned first; keeping it
// enabled disambiguates the decomposition robustly. Set check_cheirality=false
// only for intentional virtual-point reconstructions.
//
// opt.max_error is interpreted as an angular threshold in radians: the bearing
// estimator converts it internally to the residual unit (sin(angle), which
// equals the residual magnitude in the small-error limit). For pinhole bearings
// the bearing path is first-order equivalent to estimate_relative_pose after
// unprojection but differs by O(error^3) since the 2D Sampson uses non-unit
// homogeneous (x, y, 1).
RansacStats estimate_relative_pose_bearings(const std::vector<Point3D> &bearings_1,
                                            const std::vector<Point3D> &bearings_2, const RelativePoseOptions &opt,
                                            CameraPose *relative_pose, std::vector<char> *inliers,
                                            bool check_cheirality = true);

// Estimates relative geometry from using points and estimated depth using LO-RANSAC followed by non-linear refinement
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
// MonoDepth relative pose estimation with known calibration
// Uses hybrid scoring with reprojection and epipolar errors
RansacStats estimate_monodepth_relative_pose(const std::vector<Point2D> &points2D_1,
                                             const std::vector<Point2D> &points2D_2, const std::vector<double> &depth_1,
                                             const std::vector<double> &depth_2, const Camera &camera1,
                                             const Camera &camera2, const MonoDepthRelativePoseOptions &opt,
                                             MonoDepthTwoViewGeometry *geometry, std::vector<char> *inliers);

// Estimates relative pose with shared unknown focal length using LO-RANSAC followed by non-linear refinement
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
RansacStats estimate_shared_focal_relative_pose(const std::vector<Point2D> &points2D_1,
                                                const std::vector<Point2D> &points2D_2, const Point2D &pp,
                                                const RelativePoseOptions &opt, ImagePair *image_pair,
                                                std::vector<char> *inliers);

// Estimates relative pose with shared unknown focal length from point correspondences with estimated monodepth
// using LO-RANSAC followed by non-linear refinement. The points are assumed to be normalized such that pp = [0,0].
// Uses hybrid scoring with reprojection and epipolar errors
RansacStats estimate_shared_focal_monodepth_relative_pose(const std::vector<Point2D> &points2D_1,
                                                          const std::vector<Point2D> &points2D_2,
                                                          const std::vector<double> &depths_1,
                                                          const std::vector<double> &depths_2,
                                                          const MonoDepthRelativePoseOptions &opt,
                                                          MonoDepthImagePair *image_pair, std::vector<char> *inliers);

// Estimates relative pose with two different unknown focal lengths from point correspondences with estimated monodepth
// using LO-RANSAC followed by non-linear refinement. The points are assumed to be normalized such that pp = [0,0].
// Uses hybrid scoring with reprojection and epipolar errors
RansacStats estimate_varying_focal_monodepth_relative_pose(const std::vector<Point2D> &points2D_1,
                                                           const std::vector<Point2D> &points2D_2,
                                                           const std::vector<double> &depth_1,
                                                           const std::vector<double> &depth_2,
                                                           const MonoDepthRelativePoseOptions &opt,
                                                           MonoDepthImagePair *image_pair, std::vector<char> *inliers);

// Estimates a fundamental matrix using LO-RANSAC followed by non-linear refinement
// NOTE: USE estimate_relative_pose IF YOU KNOW THE INTRINSICS!!!
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
RansacStats estimate_fundamental(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                 const RelativePoseOptions &opt, Eigen::Matrix3d *F, std::vector<char> *inliers);

// Estimates a fundamental matrix with the radial distortion of two cameras followed by non-linear refinement
// Uses 10 pt algorithm (Kukelova et al., ICCV 2015) if ks is empty
// otherwise uses the sampling 7pt algorithm (Tzamos et al., ECCVW 2024)
RansacStats estimate_rd_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                    std::vector<double> ks, const RelativePoseOptions &opt,
                                    ProjectiveImagePair *F_cam_pair, std::vector<char> *inliers);

// Estimates a fundamental matrix with the radial distortion of two cameras with shared radial distortion parameter
// followed by non-linear refinement
// Uses 9 pt algorithm (Fitzgibbon, CVPR 2001) with modification from (Tzamos et al., ECCVW 2024) if ks is empty
// otherwise uses the sampling 7pt algorithm (Tzamos et al., ECCVW 2024)
RansacStats estimate_shared_rd_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                           std::vector<double> ks, const RelativePoseOptions &opt,
                                           ProjectiveImagePair *F_cam_pair, std::vector<char> *inliers);

// Estimates a homography matrix using LO-RANSAC followed by non-linear refinement
// Convention is x2 = H*x1
// Threshold for transfer error is set by RansacOptions.max_reproj_error
RansacStats estimate_homography(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                const HomographyOptions &opt, Eigen::Matrix3d *H, std::vector<char> *inliers);

// Estimates generalized relative pose using LO-RANSAC followed by non-linear refinement
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
RansacStats estimate_generalized_relative_pose(const std::vector<PairwiseMatches> &matches,
                                               const std::vector<CameraPose> &camera1_ext,
                                               const std::vector<Camera> &cameras1,
                                               const std::vector<CameraPose> &camera2_ext,
                                               const std::vector<Camera> &cameras2, const RelativePoseOptions &opt,
                                               CameraPose *relative_pose, std::vector<std::vector<char>> *inliers);

// Estimates camera pose from hybrid correspondences using LO-RANSAC followed by non-linear refinement
//  camera are the intrinsics for the query camera
//  (points2D, points3D) are the 2D-3D matches
//  (matches2D_2D, map_ext, map_cameras) are the 2D-2D matches to the map images with extrinsics/intrinsics
//     Note for matches2D_2D it is assumed that cam_ind1 indexes into map_cameras and map_ext, and cam_ind2 = 0
//     So that PairwiseMatches::x1 are the map image 2D points and PairwiseMatches::x2 are in the query camera
// TODO: Not fully implemented (only p3p sampling for now) and very untested!
RansacStats estimate_hybrid_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                 const std::vector<PairwiseMatches> &matches2D_2D, const Camera &camera,
                                 const std::vector<CameraPose> &map_ext, const std::vector<Camera> &map_cameras,
                                 const HybridPoseOptions &opt, CameraPose *pose, std::vector<char> *inliers_2D_3D,
                                 std::vector<std::vector<char>> *inliers_2D_2D);

// Estimates generalized camera pose from hybrid correspondences using LO-RANSAC followed by non-linear refinement
//  (points2D_1, points3D_1) are the 2D-3D matches where the 2D point is in the first rig and the 3D points are in the
//  second (points2D_2, points3D_2) are the 2D-3D matches where the 2D point is in the second rig and the 3D points are
//  in the first (matches2D_2D) are the 2D-2D matches between the generalized cameras camerasX, cameraX_ext  are the
//  intrinsics/extrinsics for each of the generalized cameras
// TODO: Not yet implemented.
RansacStats estimate_generalized_hybrid_pose(
    const std::vector<std::vector<Point2D>> &points2D_1, const std::vector<std::vector<Point3D>> &points3D_1,
    const std::vector<std::vector<Point2D>> &points2D_2, const std::vector<std::vector<Point3D>> &points3D_2,
    const std::vector<PairwiseMatches> &matches2D_2D, const std::vector<CameraPose> &camera1_ext,
    const std::vector<Camera> &cameras1, const std::vector<CameraPose> &camera2_ext,
    const std::vector<Camera> &cameras2, const AbsolutePoseOptions &opt, CameraPose *pose,
    std::vector<std::vector<char>> *inliers_1, std::vector<std::vector<char>> *inliers_2,
    std::vector<std::vector<char>> *inliers_2D_2D);

// Estimates the 1D absolute pose using LO-RANSAC followed by non-linear refinement
// Assumes that the image points are centered already
// Threshold for radial reprojection error is set by RansacOptions.max_reproj_error
RansacStats estimate_1D_radial_absolute_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                             const AbsolutePoseOptions &opt, CameraPose *pose,
                                             std::vector<char> *inliers);

} // namespace poselib

#endif