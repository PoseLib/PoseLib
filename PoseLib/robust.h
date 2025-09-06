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
#include "PoseLib/misc/colmap_models.h"
#include "PoseLib/robust/bundle.h"
#include "PoseLib/robust/ransac.h"
#include "PoseLib/types.h"

#include <vector>

namespace poselib {

// Estimates absolute pose using LO-RANSAC followed by non-linear refinement
// Threshold for reprojection error is set by RansacOptions.max_reproj_error
RansacStats estimate_absolute_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                   const Camera &camera, const RansacOptions &ransac_opt,
                                   const BundleOptions &bundle_opt, CameraPose *pose, std::vector<char> *inliers);

// Estimates generalized absolute pose using LO-RANSAC followed by non-linear refinement
// Threshold for reprojection error is set by RansacOptions.max_reproj_error
RansacStats estimate_generalized_absolute_pose(const std::vector<std::vector<Point2D>> &points2D,
                                               const std::vector<std::vector<Point3D>> &points3D,
                                               const std::vector<CameraPose> &camera_ext,
                                               const std::vector<Camera> &cameras, const RansacOptions &ransac_opt,
                                               const BundleOptions &bundle_opt, CameraPose *pose,
                                               std::vector<std::vector<char>> *inliers);

// Estimates absolute pose using LO-RANSAC followed by non-linear refinement
// using both 2D-3D point and line matches
// Note that line segments are described by their endpoints
// Threshold for point reprojection error is set by RansacOptions.max_reproj_error
// and for lines the threshold is set by RansacOptions.max_epipolar_error
RansacStats estimate_absolute_pose_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                        const std::vector<Line2D> &line2D, const std::vector<Line3D> &line3D,
                                        const Camera &camera, const RansacOptions &ransac_opt,
                                        const BundleOptions &bundle_opt, CameraPose *pose,
                                        std::vector<char> *inliers_points, std::vector<char> *inliers_lines);

// Estimates relative pose using LO-RANSAC followed by non-linear refinement
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
RansacStats estimate_relative_pose(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                   const Camera &camera1, const Camera &camera2, const RansacOptions &ransac_opt,
                                   const BundleOptions &bundle_opt, CameraPose *relative_pose,
                                   std::vector<char> *inliers);

// Estimates relative pose with shared unknown focal length using LO-RANSAC followed by non-linear refinement
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
RansacStats estimate_shared_focal_relative_pose(const std::vector<Point2D> &points2D_1,
                                                const std::vector<Point2D> &points2D_2, const Point2D &pp,
                                                const RansacOptions &ransac_opt, const BundleOptions &bundle_opt,
                                                ImagePair *image_pair, std::vector<char> *inliers);

// Estimates a fundamental matrix using LO-RANSAC followed by non-linear refinement
// NOTE: USE estimate_relative_pose IF YOU KNOW THE INTRINSICS!!!
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
RansacStats estimate_fundamental(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                 const RansacOptions &ransac_opt, const BundleOptions &bundle_opt, Eigen::Matrix3d *F,
                                 std::vector<char> *inliers);

// Estimates a homography matrix using LO-RANSAC followed by non-linear refinement
// Convention is x2 = H*x1
// Threshold for transfer error is set by RansacOptions.max_reproj_error
RansacStats estimate_homography(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                const RansacOptions &ransac_opt, const BundleOptions &bundle_opt, Eigen::Matrix3d *H,
                                std::vector<char> *inliers);

// Estimates generalized relative pose using LO-RANSAC followed by non-linear refinement
// Threshold for Sampson error is set by RansacOptions.max_epipolar_error
RansacStats estimate_generalized_relative_pose(const std::vector<PairwiseMatches> &matches,
                                               const std::vector<CameraPose> &camera1_ext,
                                               const std::vector<Camera> &cameras1,
                                               const std::vector<CameraPose> &camera2_ext,
                                               const std::vector<Camera> &cameras2, const RansacOptions &ransac_opt,
                                               const BundleOptions &bundle_opt, CameraPose *relative_pose,
                                               std::vector<std::vector<char>> *inliers);

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
                                 const RansacOptions &ransac_opt, const BundleOptions &bundle_opt, CameraPose *pose,
                                 std::vector<char> *inliers_2D_3D, std::vector<std::vector<char>> *inliers_2D_2D);

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
    const std::vector<Camera> &cameras2, const RansacOptions &ransac_opt, const BundleOptions &bundle_opt,
    CameraPose *pose, std::vector<std::vector<char>> *inliers_1, std::vector<std::vector<char>> *inliers_2,
    std::vector<std::vector<char>> *inliers_2D_2D);

// Estimates the 1D absolute pose using LO-RANSAC followed by non-linear refinement
// Assumes that the image points are centered already
// Threshold for radial reprojection error is set by RansacOptions.max_reproj_error
RansacStats estimate_1D_radial_absolute_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                             const RansacOptions &ransac_opt, const BundleOptions &bundle_opt,
                                             CameraPose *pose, std::vector<char> *inliers);

} // namespace poselib

#endif