#ifndef POSELIB_ROBUST_UTILS_H
#define POSELIB_ROBUST_UTILS_H

#include "types.h"
#include <Eigen/Dense>
#include <PoseLib/types.h>
#include <vector>

namespace pose_lib {

// Returns MSAC score of the reprojection error
double compute_msac_score(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count);
double compute_msac_score(const CameraPose &pose, const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, double sq_threshold, size_t *inlier_count);
// Returns MSAC score of the Sampson error
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);
double compute_sampson_msac_score(const Eigen::Matrix3d &F, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);

// Returns MSAC score of transfer error for homography
double compute_homography_msac_score(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count);

// Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
void get_inliers(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X, double sq_threshold, std::vector<char> *inliers);
void get_inliers(const CameraPose &pose, const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, double sq_threshold, std::vector<char> *inliers);

// Compute inliers for relative pose estimation (using Sampson error)
int get_inliers(const CameraPose &pose, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, double sq_threshold, std::vector<char> *inliers);
int get_inliers(const Eigen::Matrix3d &E, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, double sq_threshold, std::vector<char> *inliers);

// inliers for homography
void get_homography_inliers(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, double sq_threshold, std::vector<char> *inliers);

// Helpers for the 1D radial camera model
double compute_msac_score_1D_radial(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count);
void get_inliers_1D_radial(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X, double sq_threshold, std::vector<char> *inliers);

// Normalize points by shifting/scaling coordinate systems.
double normalize_points(std::vector<Eigen::Vector2d> &x1, std::vector<Eigen::Vector2d> &x2,
                      Eigen::Matrix3d &T1, Eigen::Matrix3d &T2, bool normalize_scale, bool normalize_centroid, bool shared_scale);

} // namespace pose_lib

#endif