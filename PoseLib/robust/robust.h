#ifndef POSELIB_ROBUST_H_
#define POSELIB_ROBUST_H_

#include <vector>
#include "types.h"
#include "colmap_models.h"
#include "ransac.h"
#include "bundle.h"

namespace pose_lib {

int estimate_absolute_pose(const std::vector<Eigen::Vector2d>& points2D,
                           const std::vector<Eigen::Vector3d>& points3D,
                           const Camera &camera, const RansacOptions &ransac_opt,
                           const BundleOptions &bundle_opt,
                           CameraPose *pose, std::vector<char> *inliers);


} // namespace pose_lib

#endif