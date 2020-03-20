#include "problem_generator.h"
#include <Eigen/Dense>
#include <random>
#include <vector>

namespace pose_lib {

static const double kPI = 3.14159265358979323846;

double CalibPoseValidator::compute_pose_error(const ProblemInstance &instance, const CameraPose &pose) {
  return (instance.pose_gt.R - pose.R).norm() + (instance.pose_gt.t - pose.t).norm() + std::abs(instance.pose_gt.alpha - pose.alpha);
}

bool CalibPoseValidator::is_valid(const ProblemInstance &instance, const CameraPose &pose, double tol) {
  if ((pose.R.transpose() * pose.R - Eigen::Matrix3d::Identity()).norm() > tol)
    return false;

  // Point to point correspondences
  // alpha * p + lambda*x = R*X + t
  for (int i = 0; i < instance.x_point_.size(); ++i) {
    double err = 1.0 - std::abs(instance.x_point_[i].dot((pose.R * instance.X_point_[i] + pose.t - pose.alpha * instance.p_point_[i]).normalized()));
    if (err > tol)
      return false;
  }

  // Point to Line correspondences
  // alpha * p + lambda * x = R*(X + mu*V) + t
  for (int i = 0; i < instance.x_line_.size(); ++i) {
    // lambda * x - mu * R*V = R*X + t - alpha * p
    // x.cross(R*V).dot(R*X+t-alpha.p) = 0
    Eigen::Vector3d X = pose.R * instance.X_line_[i] + pose.t - pose.alpha * instance.p_line_[i];
    double err = instance.x_line_[i].cross(pose.R * instance.V_line_[i]).normalized().dot(X);

    if (err > tol)
      return false;
  }

  // Line to point correspondences
  // l'*(R*X + t - alpha*p) = 0
  for (int i = 0; i < instance.l_line_point_.size(); ++i) {

    Eigen::Vector3d X = pose.R * instance.X_line_point_[i] + pose.t - pose.alpha * instance.p_line_point_[i];

    double err = std::abs(instance.l_line_point_[i].dot(X.normalized()));
    if (err > tol)
      return false;
  }

  // Line to line correspondences
  // l'*(R*(X + mu*V) + t - alpha*p) = 0
  for (int i = 0; i < instance.l_line_line_.size(); ++i) {

    Eigen::Vector3d X = pose.R * instance.X_line_line_[i] + pose.t - pose.alpha * instance.p_line_line_[i];
    Eigen::Vector3d V = pose.R * instance.V_line_line_[i];

    double err = std::abs(instance.l_line_line_[i].dot(X.normalized())) + std::abs(instance.l_line_line_[i].dot(V.normalized()));
    if (err > tol)
      return false;
  }

  return true;
}

double UnknownFocalValidator::compute_pose_error(const ProblemInstance &instance, const CameraPose &pose) {
  return (instance.pose_gt.R - pose.R).norm() + (instance.pose_gt.t - pose.t).norm() + std::abs(instance.pose_gt.alpha - pose.alpha);
}

bool UnknownFocalValidator::is_valid(const ProblemInstance &instance, const CameraPose &pose, double tol) {
  if ((pose.R.transpose() * pose.R - Eigen::Matrix3d::Identity()).norm() > tol)
    return false;

  if (pose.alpha < 0)
    return false;

  Eigen::Matrix3d Kinv;
  Kinv.setIdentity();
  Kinv(2, 2) = pose.alpha;
  // lambda*diag(1,1,alpha)*x = R*X + t
  for (int i = 0; i < instance.x_point_.size(); ++i) {
    double err = 1.0 - std::abs((Kinv * instance.x_point_[i]).normalized().dot((pose.R * instance.X_point_[i] + pose.t).normalized()));
    if (err > tol)
      return false;
  }

  return true;
}

void set_random_pose(CameraPose &pose, bool upright) {
  if (upright) {
    Eigen::Vector2d r;
    r.setRandom().normalize();
    //pose.R << r(0), 0.0, r(1), 0.0, 1.0, 0.0, -r(1), 0.0, r(0); // y-gravity
    pose.R << r(0), r(1), 0.0, -r(1), r(0), 0.0, 0.0, 0.0, 1.0; // z-gravity
  } else {
    pose.R = Eigen::Quaternion<double>::UnitRandom();
  }
  pose.t.setRandom();
}

void generate_problems(int n_problems, std::vector<ProblemInstance> *problem_instances,
                       const ProblemOptions &options) {
  problem_instances->clear();
  problem_instances->reserve(n_problems);

  double fov_scale = std::tan(options.camera_fov_ / 2.0 * kPI / 180.0);

  // Random generators
  std::default_random_engine random_engine;
  std::uniform_real_distribution<double> depth_gen(options.min_depth_, options.max_depth_);
  std::uniform_real_distribution<double> coord_gen(-fov_scale, fov_scale);
  std::uniform_real_distribution<double> scale_gen(options.min_scale_, options.max_scale_);
  std::uniform_real_distribution<double> focal_gen(options.min_focal_, options.max_focal_);
  std::normal_distribution<double> direction_gen(0.0, 1.0);
  std::normal_distribution<double> offset_gen(0.0, 1.0);

  for (int i = 0; i < n_problems; ++i) {
    ProblemInstance instance;
    set_random_pose(instance.pose_gt, options.upright_);

    if (options.unknown_scale_) {
      instance.pose_gt.alpha = scale_gen(random_engine);
    } else if (options.unknown_focal_) {
      instance.pose_gt.alpha = focal_gen(random_engine);
    }

    // Point to point correspondences
    instance.x_point_.reserve(options.n_point_point_);
    instance.X_point_.reserve(options.n_point_point_);
    instance.p_point_.reserve(options.n_point_point_);
    for (int j = 0; j < options.n_point_point_; ++j) {

      Eigen::Vector3d p{0.0, 0.0, 0.0};
      Eigen::Vector3d x{coord_gen(random_engine), coord_gen(random_engine), 1.0};
      x.normalize();
      Eigen::Vector3d X;

      if (options.generalized_) {
        p << offset_gen(random_engine), offset_gen(random_engine), offset_gen(random_engine);
      }

      X = instance.pose_gt.alpha * p + x * depth_gen(random_engine);

      X = instance.pose_gt.R.transpose() * (X - instance.pose_gt.t);

      if (options.unknown_focal_) {
        x.block<2, 1>(0, 0) *= instance.pose_gt.alpha;
        x.normalize();
      }

      instance.x_point_.push_back(x);
      instance.X_point_.push_back(X);
      instance.p_point_.push_back(p);
    }

    // Point to line correspondences
    instance.x_line_.reserve(options.n_point_line_);
    instance.X_line_.reserve(options.n_point_line_);
    instance.V_line_.reserve(options.n_point_line_);
    instance.p_line_.reserve(options.n_point_line_);
    for (int j = 0; j < options.n_point_line_; ++j) {
      Eigen::Vector3d p{0.0, 0.0, 0.0};
      Eigen::Vector3d x{coord_gen(random_engine), coord_gen(random_engine), 1.0};
      x.normalize();
      Eigen::Vector3d X;

      if (options.generalized_) {
        p << offset_gen(random_engine), offset_gen(random_engine), offset_gen(random_engine);
      }
      X = instance.pose_gt.alpha * p + x * depth_gen(random_engine);
      X = instance.pose_gt.R.transpose() * (X - instance.pose_gt.t);

      Eigen::Vector3d V{direction_gen(random_engine), direction_gen(random_engine), direction_gen(random_engine)};
      V.normalize();

      // Translate X such that X.dot(V) = 0
      X = X - V.dot(X) * V;

      if (options.unknown_focal_) {
        // TODO implement this.
      }

      instance.x_line_.push_back(x);
      instance.X_line_.push_back(X);
      instance.V_line_.push_back(V);
      instance.p_line_.push_back(p);
    }

    // Line to point correspondences
    instance.l_line_point_.reserve(options.n_line_point_);
    instance.X_line_point_.reserve(options.n_line_point_);
    instance.p_line_point_.reserve(options.n_line_point_);
    for (int j = 0; j < options.n_line_point_; ++j) {
      Eigen::Vector3d p{0.0, 0.0, 0.0};
      Eigen::Vector3d x{coord_gen(random_engine), coord_gen(random_engine), 1.0};
      x.normalize();
      Eigen::Vector3d X;

      if (options.generalized_) {
        p << offset_gen(random_engine), offset_gen(random_engine), offset_gen(random_engine);
      }
      X = instance.pose_gt.alpha * p + x * depth_gen(random_engine);
      X = instance.pose_gt.R.transpose() * (X - instance.pose_gt.t);

      // Cross product with random vector to generate line
      Eigen::Vector3d l = x.cross(Eigen::Vector3d(direction_gen(random_engine), direction_gen(random_engine), direction_gen(random_engine)));
      l.normalize();

      if (options.unknown_focal_) {
        // TODO implement this.
      }

      instance.l_line_point_.push_back(l);
      instance.X_line_point_.push_back(X);
      instance.p_line_point_.push_back(p);
    }

    // Line to line correspondences
    instance.l_line_line_.reserve(options.n_line_line_);
    instance.X_line_line_.reserve(options.n_line_line_);
    instance.V_line_line_.reserve(options.n_line_line_);
    instance.p_line_line_.reserve(options.n_line_line_);
    for (int j = 0; j < options.n_line_line_; ++j) {
      Eigen::Vector3d p{0.0, 0.0, 0.0};
      Eigen::Vector3d x{coord_gen(random_engine), coord_gen(random_engine), 1.0};
      x.normalize();
      Eigen::Vector3d X;

      if (options.generalized_) {
        p << offset_gen(random_engine), offset_gen(random_engine), offset_gen(random_engine);
      }
      X = instance.pose_gt.alpha * p + x * depth_gen(random_engine);
      X = instance.pose_gt.R.transpose() * (X - instance.pose_gt.t);

      Eigen::Vector3d V{direction_gen(random_engine), direction_gen(random_engine), direction_gen(random_engine)};
      V.normalize();

      // Translate X such that X.dot(V) = 0
      X = X - V.dot(X) * V;

      Eigen::Vector3d l = x.cross(instance.pose_gt.R * V);
      l.normalize();

      if (options.unknown_focal_) {
        // TODO implement this.
      }

      instance.l_line_line_.push_back(l);
      instance.X_line_line_.push_back(X);
      instance.V_line_line_.push_back(V);
      instance.p_line_line_.push_back(p);
    }

    problem_instances->push_back(instance);
  }
}

}; // namespace pose_lib