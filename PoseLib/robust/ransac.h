#ifndef POSELIB_RANSAC_H_
#define POSELIB_RANSAC_H_

#include "../camera_pose.h"
#include "../types.h"

#include <vector>

namespace pose_lib {

// Absolute pose estimation
RansacStats ransac_pnp(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                        const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers);

RansacStats ransac_gen_pnp(const std::vector<std::vector<Point2D>> &x,
                            const std::vector<std::vector<Point3D>> &X,
                            const std::vector<CameraPose> &camera_ext, const RansacOptions &opt,
                            CameraPose *best_model, std::vector<std::vector<char>> *best_inliers);

RansacStats ransac_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                             const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                             const RansacOptions &opt, CameraPose *best_model,
                             std::vector<char> *inliers_points, std::vector<char> *inliers_lines);

// Relative pose estimation
RansacStats ransac_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                           const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers);

RansacStats ransac_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                               const RansacOptions &opt, Eigen::Matrix3d *best_model, std::vector<char> *best_inliers);

RansacStats ransac_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                               const RansacOptions &opt, Eigen::Matrix3d *best_model, std::vector<char> *best_inliers);


RansacStats ransac_gen_relpose(const std::vector<PairwiseMatches> &matches,
                               const std::vector<CameraPose> &camera1_ext, const std::vector<CameraPose> &camera2_ext,
                               const RansacOptions &opt, CameraPose *best_model, std::vector<std::vector<char>> *best_inliers);

// Hybrid pose estimation (both 2D-2D and 2D-3D correspondences)
RansacStats ransac_hybrid_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                               const std::vector<PairwiseMatches> &matches2D_2D, const std::vector<CameraPose> &map_ext,
                               const RansacOptions &opt, CameraPose *best_model,
                               std::vector<char> *inliers_2D_3D, std::vector<std::vector<char>> *inliers_2D_2D);

// Absolute pose estimation with the 1D radial camera model
RansacStats ransac_1D_radial_pnp(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                                  const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers);

} // namespace pose_lib

#endif