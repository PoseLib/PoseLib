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
                  const BundleOptions &opt = BundleOptions());

// Uses intrinsic calibration from Camera (see colmap_models.h)
// Slightly slower than bundle_adjust above
int bundle_adjust(const std::vector<Eigen::Vector2d> &x,
                  const std::vector<Eigen::Vector3d> &X,
                  const Camera &camera,
                  CameraPose *pose,
                  const BundleOptions &opt = BundleOptions());

// Minimizes reprojection error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int generalized_bundle_adjust(const std::vector<std::vector<Eigen::Vector2d>> &x,
                              const std::vector<std::vector<Eigen::Vector3d>> &X,
                              const std::vector<CameraPose> &camera_ext,
                              CameraPose *pose,
                              const BundleOptions &opt = BundleOptions());

// Uses intrinsic calibration from Camera (see colmap_models.h)
// Slightly slower than bundle_adjust above
int generalized_bundle_adjust(const std::vector<std::vector<Eigen::Vector2d>> &x,
                              const std::vector<std::vector<Eigen::Vector3d>> &X,
                              const std::vector<CameraPose> &camera_ext,
                              const std::vector<Camera> &cameras,
                              CameraPose *pose,
                              const BundleOptions &opt = BundleOptions());

// Relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_relpose(const std::vector<Eigen::Vector2d> &x1,
                   const std::vector<Eigen::Vector2d> &x2,
                   CameraPose *pose,
                   const BundleOptions &opt = BundleOptions());

// Generalized relative pose refinement. Minimizes Sampson error error. Assumes identity intrinsics (calibrated camera)
// Returns number of iterations.
int refine_generalized_relpose(const std::vector<PairwiseMatches> &matches,
                               const std::vector<CameraPose> &camera1_ext, const std::vector<CameraPose> &camera2_ext,
                               CameraPose *pose, const BundleOptions &opt = BundleOptions());

} // namespace pose_lib

#endif