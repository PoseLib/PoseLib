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

typedef uint64_t RNG_t;
int random_int(RNG_t &state);

// Draws a random sample
void draw_sample(size_t sample_sz, size_t N, std::vector<size_t> *sample, RNG_t &rng);

// Sampling for multi-camera systems
void draw_sample(size_t sample_sz, const std::vector<size_t> &N, std::vector<std::pair<size_t, size_t>> *sample, RNG_t &rng);

} // namespace pose_lib

#endif