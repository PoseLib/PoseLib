#ifndef POSELIB_ROBUST_H_
#define POSELIB_ROBUST_H_

#include "bundle.h"
#include "colmap_models.h"
#include "ransac.h"
#include "types.h"
#include <PoseLib/types.h>
#include <vector>

namespace pose_lib {

// Estimates absolute pose using LO-RANSAC followed by non-linear refinement
RansacStats estimate_absolute_pose(const std::vector<Eigen::Vector2d> &points2D,
                                   const std::vector<Eigen::Vector3d> &points3D,
                                   const Camera &camera, const RansacOptions &ransac_opt,
                                   const BundleOptions &bundle_opt,
                                   CameraPose *pose, std::vector<char> *inliers);

// Estimates generalized absolute pose using LO-RANSAC followed by non-linear refinement
RansacStats estimate_generalized_absolute_pose(
    const std::vector<std::vector<Eigen::Vector2d>> &points2D,
    const std::vector<std::vector<Eigen::Vector3d>> &points3D,
    const std::vector<CameraPose> &camera_ext,
    const std::vector<Camera> &cameras,
    const RansacOptions &ransac_opt,
    const BundleOptions &bundle_opt,
    CameraPose *pose, std::vector<std::vector<char>> *inliers);

// Estimates relative pose using LO-RANSAC followed by non-linear refinement
RansacStats estimate_relative_pose(
    const std::vector<Eigen::Vector2d> &points2D_1,
    const std::vector<Eigen::Vector2d> &points2D_2,
    const Camera &camera1, const Camera &camera2,
    const RansacOptions &ransac_opt, const BundleOptions &bundle_opt,
    CameraPose *relative_pose, std::vector<char> *inliers);

// Estimates generalized relative pose using LO-RANSAC followed by non-linear refinement
RansacStats estimate_generalized_relative_pose(
    const std::vector<PairwiseMatches> &matches,
    const std::vector<CameraPose> &camera1_ext,
    const std::vector<Camera> &cameras1,
    const std::vector<CameraPose> &camera2_ext,
    const std::vector<Camera> &cameras2,
    const RansacOptions &ransac_opt, const BundleOptions &bundle_opt,
    CameraPose *relative_pose, std::vector<std::vector<char>> *inliers);

} // namespace pose_lib

#endif