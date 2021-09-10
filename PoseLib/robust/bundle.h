#ifndef POSELIB_BUNDLE_H_
#define POSELIB_BUNDLE_H_

#include "../types.h"
#include "colmap_models.h"
#include "types.h"
#include <Eigen/Dense>

namespace pose_lib {

// Minimizes reprojection error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int bundle_adjust(const std::vector<Eigen::Vector2d> &x,
                  const std::vector<Eigen::Vector3d> &X,
                  CameraPose *pose,
                  const BundleOptions &opt = BundleOptions(),
                  const std::vector<double> &weights = std::vector<double>());

// Uses intrinsic calibration from Camera (see colmap_models.h)
// Slightly slower than bundle_adjust above
int bundle_adjust(const std::vector<Eigen::Vector2d> &x,
                  const std::vector<Eigen::Vector3d> &X,
                  const Camera &camera,
                  CameraPose *pose,
                  const BundleOptions &opt = BundleOptions(),
                  const std::vector<double> &weights = std::vector<double>());

// Minimizes reprojection error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int generalized_bundle_adjust(const std::vector<std::vector<Eigen::Vector2d>> &x,
                              const std::vector<std::vector<Eigen::Vector3d>> &X,
                              const std::vector<CameraPose> &camera_ext,
                              CameraPose *pose,
                              const BundleOptions &opt = BundleOptions(),
                              const std::vector<std::vector<double>> &weights = std::vector<std::vector<double>>());

// Uses intrinsic calibration from Camera (see colmap_models.h)
// Slightly slower than bundle_adjust above
int generalized_bundle_adjust(const std::vector<std::vector<Eigen::Vector2d>> &x,
                              const std::vector<std::vector<Eigen::Vector3d>> &X,
                              const std::vector<CameraPose> &camera_ext,
                              const std::vector<Camera> &cameras,
                              CameraPose *pose,
                              const BundleOptions &opt = BundleOptions(),
                              const std::vector<std::vector<double>> &weights = std::vector<std::vector<double>>());

// Relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_relpose(const std::vector<Eigen::Vector2d> &x1,
                   const std::vector<Eigen::Vector2d> &x2,
                   CameraPose *pose,
                   const BundleOptions &opt = BundleOptions(),
                   const std::vector<double> &weights = std::vector<double>());

// Fundamental matrix refinement. Minimizes Sampson error error.
// Returns number of iterations.
int refine_fundamental(const std::vector<Eigen::Vector2d> &x1,
                       const std::vector<Eigen::Vector2d> &x2,
                       Eigen::Matrix3d *pose,
                       const BundleOptions &opt = BundleOptions(),
                       const std::vector<double> &weights = std::vector<double>());

// Generalized relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_generalized_relpose(const std::vector<PairwiseMatches> &matches,
                               const std::vector<CameraPose> &camera1_ext, const std::vector<CameraPose> &camera2_ext,
                               CameraPose *pose, const BundleOptions &opt = BundleOptions(),
                               const std::vector<std::vector<double>> &weights = std::vector<std::vector<double>>());

// If you use weights, then both weights_abs and weights_rel needs to be passed!
int refine_hybrid_pose(const std::vector<Eigen::Vector2d> &x,
                       const std::vector<Eigen::Vector3d> &X,
                       const std::vector<PairwiseMatches> &matches_2D_2D,
                       const std::vector<CameraPose> &map_ext,
                       CameraPose *pose, const BundleOptions &opt = BundleOptions(), double loss_scale_epipolar = 1.0,
                       const std::vector<double> &weights_abs = std::vector<double>(),
                       const std::vector<std::vector<double>> &weights_rel = std::vector<std::vector<double>>());

// Minimizes the 1D radial reprojection error. Assumes that the image points are centered
// Returns number of iterations.
int bundle_adjust_1D_radial(const std::vector<Eigen::Vector2d> &x,
                            const std::vector<Eigen::Vector3d> &X,
                            CameraPose *pose,
                            const BundleOptions &opt = BundleOptions(),
                            const std::vector<double> &weights = std::vector<double>());

} // namespace pose_lib

#endif