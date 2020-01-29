#include <Eigen/Dense>
#include <vector>
#include "problem_generator.h"
#include <random>


namespace pose_lib {

static const double kPI = 3.14159265358979323846;


double ProblemInstance::compute_pose_error(const CameraPose &pose) const {
    return (pose_gt.R - pose.R).norm() + (pose_gt.t - pose.t).norm();
}

bool ProblemInstance::is_valid(const CameraPose &pose, double tol) const {
    if((pose.R.transpose()*pose.R - Eigen::Matrix3d::Identity()).norm() > tol)
        return false;

    // p + lambda*x = R*X + t
    for(int i = 0; i < x_point_.size(); ++i) {
        double err = 1.0 - std::abs(x_point_[i].dot( (pose.R * X_point_[i] + pose.t - p_point_[i]).normalized() ));
        if(err > tol)
            return false;
    }

    return true;
}



void set_random_pose(CameraPose &pose, bool upright) {
    if(upright) {
        Eigen::Vector2d r; r.setRandom().normalize();
        //pose.R << r(0), 0.0, r(1), 0.0, 1.0, 0.0, -r(1), 0.0, r(0); // y-gravity
        pose.R << r(0), r(1), 0.0, -r(1), r(0), 0.0, 0.0, 0.0, 1.0; // z-gravity
    } else {
        pose.R = Eigen::Quaternion<double>::UnitRandom();
    }
    pose.t.setRandom();
}

void generate_problems(int n_problems, std::vector<ProblemInstance> *problem_instances,
                       const ProblemOptions &options)
{
    problem_instances->clear();
    problem_instances->reserve(n_problems);

    double fov_scale = std::tan(options.camera_fov_ / 2.0 * kPI / 180.0);  

    std::default_random_engine random_engine;
    std::uniform_real_distribution<double> depth_gen(options.min_depth_, options.max_depth_);
    std::uniform_real_distribution<double> coord_gen(-fov_scale, fov_scale);
    std::normal_distribution<double> direction_gen(0.0, 1.0);
    std::normal_distribution<double> offset_gen(0.0, 1.0);

    for(int i = 0; i < n_problems; ++i) {
        ProblemInstance instance;
        set_random_pose(instance.pose_gt, options.upright_);

        instance.x_point_.reserve(options.n_point_point_);
        instance.X_point_.reserve(options.n_point_point_);
        for(int j = 0; j < options.n_point_point_; ++j) {

            Eigen::Vector3d p{0.0, 0.0, 0.0};
            Eigen::Vector3d x{coord_gen(random_engine), coord_gen(random_engine), 1.0};            
            x.normalize();
            Eigen::Vector3d X;
            
            if(options.generalized_) {
               p << offset_gen(random_engine), offset_gen(random_engine), offset_gen(random_engine);             
            }

            X = p + x * depth_gen(random_engine);
            
            X = instance.pose_gt.R.transpose() * (X - instance.pose_gt.t);

            instance.x_point_.push_back(x);
            instance.X_point_.push_back(X);
            instance.p_point_.push_back(p);
        }

        instance.x_line_.reserve(options.n_point_line_);
        instance.X_line_.reserve(options.n_point_line_);
        instance.V_line_.reserve(options.n_point_line_);
        for(int j = 0; j < options.n_point_line_; ++j) {
            Eigen::Vector3d p{0.0, 0.0, 0.0};
            Eigen::Vector3d x{coord_gen(random_engine), coord_gen(random_engine), 1.0};            
            x.normalize();
            Eigen::Vector3d X;
            
            if(options.generalized_) {
                p << offset_gen(random_engine), offset_gen(random_engine), offset_gen(random_engine);
            
            }
            X = p + x * depth_gen(random_engine);
            X = instance.pose_gt.R.transpose() * (X - instance.pose_gt.t);


            Eigen::Vector3d V{direction_gen(random_engine), direction_gen(random_engine), direction_gen(random_engine)};
            V.normalize();


            // Translate X such that X.dot(V) = 0
            X = X - V.dot(X) * V;

            instance.x_line_.push_back(x);
            instance.X_line_.push_back(X);
            instance.V_line_.push_back(V);
            instance.p_line_.push_back(p);
        }

        problem_instances->push_back(instance);
    }
}


};