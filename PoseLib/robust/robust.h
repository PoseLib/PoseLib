#ifndef POSELIB_ROBUST_H_
#define POSELIB_ROBUST_H_

#include "bundle.h"
#include "colmap_models.h"
#include "ransac.h"
#include <PoseLib/types.h>
#include <vector>

namespace pose_lib {

int estimate_absolute_pose(const std::vector<Eigen::Vector2d> &points2D,
                           const std::vector<Eigen::Vector3d> &points3D,
                           const Camera &camera, const RansacOptions &ransac_opt,
                           const BundleOptions &bundle_opt,
                           CameraPose *pose, std::vector<char> *inliers);

int estimate_generalized_absolute_pose(
    const std::vector<std::vector<Eigen::Vector2d>> &points2D,
    const std::vector<std::vector<Eigen::Vector3d>> &points3D,
    const std::vector<CameraPose> &camera_ext,
    const std::vector<Camera> &cameras,
    const RansacOptions &ransac_opt,
    const BundleOptions &bundle_opt,
    CameraPose *pose, std::vector<std::vector<char>> *inliers);

} // namespace pose_lib

#endif