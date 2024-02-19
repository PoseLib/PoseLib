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

#ifndef POSELIB_BUNDLE_H_
#define POSELIB_BUNDLE_H_

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/colmap_models.h"
#include "PoseLib/types.h"

#include <Eigen/Dense>

namespace poselib {

// Minimizes reprojection error. Assumes identity intrinsics (calibrated camera)
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                          const BundleOptions &opt = BundleOptions(),
                          const std::vector<double> &weights = std::vector<double>());

// Uses intrinsic calibration from Camera (see colmap_models.h)
// Slightly slower than bundle_adjust above
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const Camera &camera,
                          CameraPose *pose, const BundleOptions &opt = BundleOptions(),
                          const std::vector<double> &weights = std::vector<double>());

// opt_line is used to define the robust loss used for the line correspondences
BundleStats bundle_adjust(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                          const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, CameraPose *pose,
                          const BundleOptions &opt = BundleOptions(), const BundleOptions &opt_line = BundleOptions(),
                          const std::vector<double> &weights_pts = std::vector<double>(),
                          const std::vector<double> &weights_line = std::vector<double>());

/*
// Camera models for lines are currently not supported...
int bundle_adjust(const std::vector<Point2D> &points2D,
                  const std::vector<Point3D> &points3D,
                  const std::vector<Line2D> &lines2D,
                  const std::vector<Line3D> &lines3D,
                  const Camera &camera,
                  CameraPose *pose,
                  const BundleOptions &opt = BundleOptions(),
                  const std::vector<double> &weights = std::vector<double>());
*/

// Minimizes reprojection error. Assumes identity intrinsics (calibrated camera)
BundleStats
generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x, const std::vector<std::vector<Point3D>> &X,
                          const std::vector<CameraPose> &camera_ext, CameraPose *pose,
                          const BundleOptions &opt = BundleOptions(),
                          const std::vector<std::vector<double>> &weights = std::vector<std::vector<double>>());

// Uses intrinsic calibration from Camera (see colmap_models.h)
// Slightly slower than bundle_adjust above
BundleStats
generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x, const std::vector<std::vector<Point3D>> &X,
                          const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &cameras,
                          CameraPose *pose, const BundleOptions &opt = BundleOptions(),
                          const std::vector<std::vector<double>> &weights = std::vector<std::vector<double>>());

// Relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
BundleStats refine_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, CameraPose *pose,
                           const BundleOptions &opt = BundleOptions(),
                           const std::vector<double> &weights = std::vector<double>());

// Relative pose with single unknown focal refinement. Minimizes Sampson error error. Assumes identity intrinsics
// (calibrated camera)
BundleStats refine_shared_focal_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                        ImagePair *image_pair, const BundleOptions &opt = BundleOptions(),
                                        const std::vector<double> &weights = std::vector<double>());

// Fundamental matrix refinement. Minimizes Sampson error error.
BundleStats refine_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *F,
                               const BundleOptions &opt = BundleOptions(),
                               const std::vector<double> &weights = std::vector<double>());

// Homography matrix refinement.
BundleStats refine_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *H,
                              const BundleOptions &opt = BundleOptions(),
                              const std::vector<double> &weights = std::vector<double>());

// Generalized relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
BundleStats
refine_generalized_relpose(const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
                           const std::vector<CameraPose> &camera2_ext, CameraPose *pose,
                           const BundleOptions &opt = BundleOptions(),
                           const std::vector<std::vector<double>> &weights = std::vector<std::vector<double>>());

// If you use weights, then both weights_abs and weights_rel needs to be passed!
BundleStats
refine_hybrid_pose(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                   const std::vector<PairwiseMatches> &matches_2D_2D, const std::vector<CameraPose> &map_ext,
                   CameraPose *pose, const BundleOptions &opt = BundleOptions(), double loss_scale_epipolar = 1.0,
                   const std::vector<double> &weights_abs = std::vector<double>(),
                   const std::vector<std::vector<double>> &weights_rel = std::vector<std::vector<double>>());

// Minimizes the 1D radial reprojection error. Assumes that the image points are centered
BundleStats bundle_adjust_1D_radial(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                                    const BundleOptions &opt = BundleOptions(),
                                    const std::vector<double> &weights = std::vector<double>());

} // namespace poselib

#endif