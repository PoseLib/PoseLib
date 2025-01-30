#pragma once

#include <Eigen/Dense>
#include <PoseLib/camera_pose.h>
#include <vector>

namespace poselib {

struct AbsolutePoseProblemInstance {
  public:
    // Ground truth camera pose
    CameraPose pose_gt;
    Real scale_gt = 1.0;
    Real focal_gt = 1.0;

    // Point-to-point correspondences
    std::vector<Vector3> x_point_;
    std::vector<Vector3> X_point_;

    // Point-to-line correspondences
    std::vector<Vector3> x_line_;
    std::vector<Vector3> X_line_;
    std::vector<Vector3> V_line_;

    // Line-to-point correspondences
    std::vector<Vector3> l_line_point_;
    std::vector<Vector3> X_line_point_;

    // Line-to-line correspondences
    std::vector<Vector3> l_line_line_;
    std::vector<Vector3> X_line_line_;
    std::vector<Vector3> V_line_line_;

    // For generalized cameras we have an offset in the camera coordinate system
    std::vector<Vector3> p_point_;
    std::vector<Vector3> p_line_;
    std::vector<Vector3> p_line_point_;
    std::vector<Vector3> p_line_line_;
};

struct RelativePoseProblemInstance {
  public:
    // Ground truth camera pose
    CameraPose pose_gt;
    Matrix3x3 H_gt; // for homography problems
    Real scale_gt = 1.0;
    Real focal_gt = 1.0;

    // Point-to-point correspondences
    std::vector<Vector3> p1_;
    std::vector<Vector3> x1_;

    std::vector<Vector3> p2_;
    std::vector<Vector3> x2_;
};

struct CalibPoseValidator {
    // Computes the distance to the ground truth pose
    static Real compute_pose_error(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, Real scale);
    static Real compute_pose_error(const RelativePoseProblemInstance &instance, const CameraPose &pose);
    static Real compute_pose_error(const RelativePoseProblemInstance &instance, const ImagePair &image_pair);
    // Checks if the solution is valid (i.e. is rotation matrix and satisfies projection constraints)
    static bool is_valid(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, Real scale, Real tol);
    static bool is_valid(const RelativePoseProblemInstance &instance, const CameraPose &pose, Real tol);
    static bool is_valid(const RelativePoseProblemInstance &instance, const ImagePair &image_pair, Real tol);
};

struct HomographyValidator {
    // Computes the distance to the ground truth pose
    static Real compute_pose_error(const RelativePoseProblemInstance &instance, const Matrix3x3 &H);
    // Checks if the solution is valid (i.e. is rotation matrix and satisfies projection constraints)
    static bool is_valid(const RelativePoseProblemInstance &instance, const Matrix3x3 &H, Real tol);
};

struct UnknownFocalValidator {
    // Computes the distance to the ground truth pose
    static Real compute_pose_error(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, Real focal);
    // Checks if the solution is valid (i.e. is rotation matrix and satisfies projection constraints)
    static bool is_valid(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, Real focal, Real tol);
};

struct RadialPoseValidator {
    // Computes the distance to the ground truth pose
    static Real compute_pose_error(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, Real scale);
    // Checks if the solution is valid (i.e. is rotation matrix and satisfies projection constraints)
    static bool is_valid(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, Real scale, Real tol);
};

struct ProblemOptions {
    Real min_depth_ = 0.1;
    Real max_depth_ = 10.0;
    Real camera_fov_ = 70.0;
    int n_point_point_ = 0;
    int n_point_line_ = 0;
    int n_line_point_ = 0;
    int n_line_line_ = 0;
    bool upright_ = false;
    bool planar_ = false;
    bool generalized_ = false;
    bool generalized_duplicate_obs_ = false;
    int generalized_first_cam_obs_ = 0; // how many of the points should from the first camera (relpose only)
    bool unknown_scale_ = false;
    bool unknown_focal_ = false;
    bool radial_lines_ = false;
    Real min_scale_ = 0.1;
    Real max_scale_ = 10.0;
    Real min_focal_ = 100.0;
    Real max_focal_ = 1000.0;
    std::string additional_name_ = "";
};

void set_random_pose(CameraPose &pose, bool upright, bool planar);

void generate_abspose_problems(int n_problems, std::vector<AbsolutePoseProblemInstance> *problem_instances,
                               const ProblemOptions &options);
void generate_relpose_problems(int n_problems, std::vector<RelativePoseProblemInstance> *problem_instances,
                               const ProblemOptions &options);
void generate_homography_problems(int n_problems, std::vector<RelativePoseProblemInstance> *problem_instances,
                                  const ProblemOptions &options);

}; // namespace poselib
