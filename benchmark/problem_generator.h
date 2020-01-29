#pragma once

#include <Eigen/Dense>
#include <vector>
#include <PoseLib/types.h>

namespace pose_lib {

    struct ProblemInstance {        
    public:
        // Computes the distance to the ground truth pose
        double compute_pose_error(const CameraPose &pose) const;
        // Checks if the solution is valid (i.e. is rotation matrix and satisfies projection constraints)
        bool is_valid(const CameraPose &pose, double tol) const;

        // Ground truth camera pose
        CameraPose pose_gt;

        // Point-to-point correspondences      
        std::vector<Eigen::Vector3d> x_point_;
        std::vector<Eigen::Vector3d> X_point_;         

        // Point-to-line correspondences
        std::vector<Eigen::Vector3d> x_line_;
        std::vector<Eigen::Vector3d> X_line_;        
        std::vector<Eigen::Vector3d> V_line_;        

        // For generalized cameras we have an offset in the camera coordinate system
        std::vector<Eigen::Vector3d> p_point_;       
        std::vector<Eigen::Vector3d> p_line_;              
    };

    struct ProblemOptions {
        double min_depth_ = 0.1;
        double max_depth_ = 10.0;
        double camera_fov_ = 70.0;
        int n_point_point_ = 3;
        int n_point_line_ = 0;
        bool upright_ = false;
        bool generalized_ = false;
    };

    void set_random_pose(CameraPose &pose, bool upright);    

    void generate_problems(int n_problems, std::vector<ProblemInstance> *problem_instances, const ProblemOptions &options);


};