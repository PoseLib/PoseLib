#ifndef POSELIB_RANSAC_H_
#define POSELIB_RANSAC_H_

#include "../types.h"
#include "types.h"
#include <vector>

namespace pose_lib {

RansacStats ransac_pose(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X,
                        const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers);

RansacStats ransac_gen_pose(const std::vector<std::vector<Eigen::Vector2d>> &x,
                            const std::vector<std::vector<Eigen::Vector3d>> &X,
                            const std::vector<CameraPose> &camera_ext, const RansacOptions &opt,
                            CameraPose *best_model, std::vector<std::vector<char>> *best_inliers);

RansacStats ransac_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                           const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers);

RansacStats ransac_gen_relpose(const std::vector<PairwiseMatches> &matches,
                               const std::vector<CameraPose> &camera1_ext, const std::vector<CameraPose> &camera2_ext,
                               const RansacOptions &opt, CameraPose *best_model, std::vector<std::vector<char>> *best_inliers);

} // namespace pose_lib

#endif