#include "problem_generator.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <random>
#include <vector>
#include <iostream> // HACK

#include "PoseLib/misc/radial.h"

namespace poselib {

static const double kPI = 3.14159265358979323846;

double CalibPoseValidator::compute_pose_error(const AbsolutePoseProblemInstance &instance, const CameraPose &pose,
                                              double scale) {
    return (instance.pose_gt.R() - pose.R()).norm() + (instance.pose_gt.t - pose.t).norm() +
           std::abs(instance.scale_gt - scale);
}
double CalibPoseValidator::compute_pose_error(const RelativePoseProblemInstance &instance, const CameraPose &pose) {
    return (instance.pose_gt.R() - pose.R()).norm() + (instance.pose_gt.t - pose.t).norm();
}

bool CalibPoseValidator::is_valid(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, double scale,
                                  double tol) {

    // Point to point correspondences
    // alpha * p + lambda*x = R*X + t
    for (int i = 0; i < instance.x_point_.size(); ++i) {
        double err = 1.0 - std::abs(instance.x_point_[i].dot(
                               (pose.R() * instance.X_point_[i] + pose.t - scale * instance.p_point_[i]).normalized()));
        if (err > tol)
            return false;
    }

    // Point to Line correspondences
    // alpha * p + lambda * x = R*(X + mu*V) + t
    for (int i = 0; i < instance.x_line_.size(); ++i) {
        // lambda * x - mu * R*V = R*X + t - alpha * p
        // x.cross(R*V).dot(R*X+t-alpha.p) = 0
        Eigen::Vector3d X = pose.R() * instance.X_line_[i] + pose.t - scale * instance.p_line_[i];
        double err = instance.x_line_[i].cross(pose.R() * instance.V_line_[i]).normalized().dot(X);

        if (err > tol)
            return false;
    }

    // Line to point correspondences
    // l'*(R*X + t - alpha*p) = 0
    for (int i = 0; i < instance.l_line_point_.size(); ++i) {

        Eigen::Vector3d X = pose.R() * instance.X_line_point_[i] + pose.t - scale * instance.p_line_point_[i];

        double err = std::abs(instance.l_line_point_[i].dot(X.normalized()));
        if (err > tol)
            return false;
    }

    // Line to line correspondences
    // l'*(R*(X + mu*V) + t - alpha*p) = 0
    for (int i = 0; i < instance.l_line_line_.size(); ++i) {

        Eigen::Vector3d X = pose.R() * instance.X_line_line_[i] + pose.t - scale * instance.p_line_line_[i];
        Eigen::Vector3d V = pose.R() * instance.V_line_line_[i];

        double err = std::abs(instance.l_line_line_[i].dot(X.normalized())) +
                     std::abs(instance.l_line_line_[i].dot(V.normalized()));
        if (err > tol)
            return false;
    }

    return true;
}

bool CalibPoseValidator::is_valid(const RelativePoseProblemInstance &instance, const CameraPose &pose, double tol) {
    if ((pose.R().transpose() * pose.R() - Eigen::Matrix3d::Identity()).norm() > tol)
        return false;

    // Point to point correspondences
    // R * (alpha * p1 + lambda1 * x1) + t = alpha * p2 + lambda2 * x2
    //
    // cross(R*x1, x2)' * (alpha * p2 - t - alpha * R*p1) = 0
    for (int i = 0; i < instance.x1_.size(); ++i) {
        double err = std::abs(instance.x2_[i]
                                  .cross(pose.R() * instance.x1_[i])
                                  .normalized()
                                  .dot(pose.R() * instance.p1_[i] + pose.t - instance.p2_[i]));
        if (err > tol)
            return false;
    }

    return true;
}

double HomographyValidator::compute_pose_error(const HomographyProblemInstance &instance, const Eigen::Matrix3d &H) {
    double err1 = (H.normalized() - instance.H_gt.normalized()).norm();
    double err2 = (H.normalized() + instance.H_gt.normalized()).norm();
    return std::min(err1, err2);
}

bool HomographyValidator::is_valid(const HomographyProblemInstance &instance, const Eigen::Matrix3d &H, double tol) {

    for (int i = 0; i < instance.x1_.size(); ++i) {
        Eigen::Vector3d z = H * instance.x1_[i];
        double err = 1.0 - std::abs(z.normalized().dot(instance.x2_[i].normalized()));
        if (err > tol)
            return false;
    }

    return true;
}

double RadialHomographyValidator::compute_pose_error(const HomographyProblemInstance &instance, const Eigen::Matrix3d &H, double distortion_parameter1, double distortion_parameter2) {
    double err1 = (H.normalized() - instance.H_gt.normalized()).norm();
    double err2 = (H.normalized() + instance.H_gt.normalized()).norm();
    return std::min(err1, err2) + 0.5 * std::abs(instance.distortion1_gt - distortion_parameter1) + 0.5 * std::abs(instance.distortion2_gt - distortion_parameter2);
}

bool RadialHomographyValidator::is_valid(const HomographyProblemInstance &instance, const Eigen::Matrix3d &H, double distortion_parameter1, double distortion_parameter2, double tol) {
    /* TODO: Not sure if this should be checked? Kukelova's paper says > 0 is okay...
    if (distortion_parameter1 > 0) {
        return false;
    }
    if (distortion_parameter2 > 0) {
        return false;
    }
    */

    // Measure in rectified space for now
    for (int i = 0; i < instance.x1_.size(); ++i) {
        Eigen::Vector3d z = H * radialundistort(instance.x1_[i].hnormalized(), distortion_parameter1).colwise().homogeneous();
        Eigen::Vector3d x2u = radialundistort(instance.x2_[i].hnormalized(), distortion_parameter2).colwise().homogeneous();
        double err = 1.0 - std::abs(z.normalized().dot(x2u.normalized()));
        if (err > tol)
            return false;
    }

    return true;
}

// Different focal length not yet supported in poselib
double UnknownFocalHomographyValidator::compute_pose_error(const HomographyProblemInstance &instance, const Eigen::Matrix3d &H, double focal_length1, double focal_length2) {
    double err1 = (H.normalized() - instance.H_gt.normalized()).norm();
    double err2 = (H.normalized() + instance.H_gt.normalized()).norm();
    return std::min(err1, err2) + 0.5 * std::abs(instance.focal1_gt - focal_length1) / instance.focal1_gt + 0.5 * std::abs(instance.focal2_gt - focal_length2) / instance.focal2_gt;
}

bool UnknownFocalHomographyValidator::is_valid(const HomographyProblemInstance &instance, const Eigen::Matrix3d &H, double focal_length1, double focal_length2, double tol) {
    if (focal_length1 < 0) {
        return false;
    }
    if (focal_length2 < 0) {
        return false;
    }

    for (int i = 0; i < instance.x1_.size(); ++i) {
        Eigen::Vector3d z = H * instance.x1_[i];
        double err = 1.0 - std::abs(z.normalized().dot(instance.x2_[i].normalized()));
        if (err > tol)
            return false;
    }

    return true;
}

double UnknownFocalAndRadialHomographyValidator::compute_pose_error(const HomographyProblemInstance &instance, const Eigen::Matrix3d &H, double focal_length, double distortion_parameter) {
    double err1 = (H.normalized() - instance.H_gt.normalized()).norm();
    double err2 = (H.normalized() + instance.H_gt.normalized()).norm();
    std::cout << "H =\n" << H.normalized() << std::endl;
    std::cout << "H_gt =\n" << instance.H_gt.normalized() << std::endl;
    std::cout << "f =\n" << focal_length << std::endl;
    std::cout << "f_gt =\n" << instance.focal1_gt << std::endl;
    std::cout << "r =\n" << distortion_parameter << std::endl;
    std::cout << "r_gt =\n" << instance.distortion1_gt << std::endl;
    return std::min(err1, err2) + 0.5 * std::abs(instance.focal1_gt - focal_length) / instance.focal1_gt + 0.5 * std::abs(instance.distortion1_gt - distortion_parameter);
}

bool UnknownFocalAndRadialHomographyValidator::is_valid(const HomographyProblemInstance &instance, const Eigen::Matrix3d &H, double focal_length, double distortion_parameter, double tol) {
    if (focal_length < 0) {
        std::cout << "ERROR - focal length neg." << std::endl;
        return false;
    }

    // Measure in rectified space for now
    for (int i = 0; i < instance.x1_.size(); ++i) {
        Eigen::Vector3d z = H * radialundistort(instance.x1_[i].hnormalized(), distortion_parameter).colwise().homogeneous();
        Eigen::Vector3d x2u = radialundistort(instance.x2_[i].hnormalized(), distortion_parameter).colwise().homogeneous();
        double err = 1.0 - std::abs(z.normalized().dot(x2u.normalized()));
        if (err > tol) {
            std::cout << "ERROR - err[" << i << "]=" << err << std::endl;
            return false;
        }
    }

    return true;
}

double UnknownFocalValidator::compute_pose_error(const AbsolutePoseProblemInstance &instance, const CameraPose &pose,
                                                 double focal) {
    return (instance.pose_gt.R() - pose.R()).norm() + (instance.pose_gt.t - pose.t).norm() +
           std::abs(instance.focal_gt - focal);
}

bool UnknownFocalValidator::is_valid(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, double focal,
                                     double tol) {
    if ((pose.R().transpose() * pose.R() - Eigen::Matrix3d::Identity()).norm() > tol)
        return false;

    if (focal < 0)
        return false;

    Eigen::Matrix3d Kinv;
    Kinv.setIdentity();
    Kinv(2, 2) = focal;
    // lambda*diag(1,1,alpha)*x = R*X + t
    for (int i = 0; i < instance.x_point_.size(); ++i) {
        double err = 1.0 - std::abs((Kinv * instance.x_point_[i])
                                        .normalized()
                                        .dot((pose.R() * instance.X_point_[i] + pose.t).normalized()));
        if (err > tol)
            return false;
    }

    return true;
}

double RadialPoseValidator::compute_pose_error(const AbsolutePoseProblemInstance &instance, const CameraPose &pose,
                                               double scale) {
    // Only compute up to sign for radial cameras

    double err1 = (instance.pose_gt.R().topRows(2) - pose.R().topRows(2)).norm() +
                  (instance.pose_gt.t.topRows(2) - pose.t.topRows(2)).norm();
    double err2 = (instance.pose_gt.R().topRows(2) + pose.R().topRows(2)).norm() +
                  (instance.pose_gt.t.topRows(2) + pose.t.topRows(2)).norm();

    return std::min(err1, err2);
}

bool RadialPoseValidator::is_valid(const AbsolutePoseProblemInstance &instance, const CameraPose &pose, double scale,
                                   double tol) {
    if ((pose.R().transpose() * pose.R() - Eigen::Matrix3d::Identity()).norm() > tol)
        return false;

    // Point to point correspondences -- Convert these to line correspondences
    // alpha * p + lambda*x = R*X + t
    for (int i = 0; i < instance.x_point_.size(); ++i) {
        Eigen::Vector3d radial_line{-instance.x_point_[i](1), instance.x_point_[i](0), 0.0};
        Eigen::Vector3d X = pose.R() * instance.X_point_[i] + pose.t;
        double err = std::abs(radial_line.dot(X.normalized()));
        if (err > tol)
            return false;
    }

    // Line to point correspondences
    // l'*(R*X + t) = 0
    for (int i = 0; i < instance.l_line_point_.size(); ++i) {
        Eigen::Vector3d X = pose.R() * instance.X_line_point_[i] + pose.t;

        double err = std::abs(instance.l_line_point_[i].dot(X.normalized()));
        if (err > tol)
            return false;
    }

    return true;
}

void set_random_pose(CameraPose &pose, bool upright, bool planar) {
    if (upright) {
        Eigen::Vector2d r;
        r.setRandom().normalize();
        Eigen::Matrix3d R;
        R << r(0), 0.0, r(1), 0.0, 1.0, 0.0, -r(1), 0.0, r(0); // y-gravity
        // pose.R << r(0), r(1), 0.0, -r(1), r(0), 0.0, 0.0, 0.0, 1.0; // z-gravity
        pose.q = rotmat_to_quat(R);
    } else {
        pose.q = Eigen::Quaternion<double>::UnitRandom().coeffs();
    }
    pose.t.setRandom();
    if (planar) {
        pose.t.y() = 0;
    }
}

void generate_abspose_problems(int n_problems, std::vector<AbsolutePoseProblemInstance> *problem_instances,
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
        AbsolutePoseProblemInstance instance;
        set_random_pose(instance.pose_gt, options.upright_, options.planar_);

        if (options.unknown_scale_) {
            instance.scale_gt = scale_gen(random_engine);
        }
        if (options.unknown_focal_) {
            instance.focal_gt = focal_gen(random_engine);
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

            X = instance.scale_gt * p + x * depth_gen(random_engine);

            X = instance.pose_gt.R().transpose() * (X - instance.pose_gt.t);

            if (options.unknown_focal_) {
                x.block<2, 1>(0, 0) *= instance.focal_gt;
                x.normalize();
            }

            instance.x_point_.push_back(x);
            instance.X_point_.push_back(X);
            instance.p_point_.push_back(p);
        }

        // This generates instances where the same 3D point is observed twice in a generalized camera
        // This is degenerate case for the 3Q3 based gp3p/gp4ps solver unless specifically handled.
        if (options.generalized_ && options.generalized_duplicate_obs_) {
            std::vector<int> ind = {0, 1, 2, 3};
            assert(options.n_point_point_ >= 4);

            std::random_shuffle(ind.begin(), ind.end());
            instance.X_point_[ind[1]] = instance.X_point_[ind[0]];
            instance.x_point_[ind[1]] = (instance.pose_gt.R() * instance.X_point_[ind[0]] + instance.pose_gt.t -
                                         instance.scale_gt * instance.p_point_[ind[1]])
                                            .normalized();
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
            X = instance.scale_gt * p + x * depth_gen(random_engine);
            X = instance.pose_gt.R().transpose() * (X - instance.pose_gt.t);

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
            X = instance.scale_gt * p + x * depth_gen(random_engine);
            X = instance.pose_gt.R().transpose() * (X - instance.pose_gt.t);

            // Cross product with random vector to generate line
            Eigen::Vector3d l;
            if (options.radial_lines_) {
                // Line passing through image center
                l = x.cross(Eigen::Vector3d{0.0, 0.0, 1.0});
            } else {
                // Random line
                l = x.cross(Eigen::Vector3d(direction_gen(random_engine), direction_gen(random_engine),
                                            direction_gen(random_engine)));
            }

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
            X = instance.scale_gt * p + x * depth_gen(random_engine);
            X = instance.pose_gt.R().transpose() * (X - instance.pose_gt.t);

            Eigen::Vector3d V{direction_gen(random_engine), direction_gen(random_engine), direction_gen(random_engine)};
            V.normalize();

            // Translate X such that X.dot(V) = 0
            X = X - V.dot(X) * V;

            Eigen::Vector3d l = x.cross(instance.pose_gt.R() * V);
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

void generate_relpose_problems(int n_problems, std::vector<RelativePoseProblemInstance> *problem_instances,
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
        RelativePoseProblemInstance instance;
        set_random_pose(instance.pose_gt, options.upright_, options.planar_);

        if (options.unknown_scale_) {
            instance.scale_gt = scale_gen(random_engine);
        }
        if (options.unknown_focal_) {
            instance.focal_gt = focal_gen(random_engine);
        }

        if (!options.generalized_) {
            instance.pose_gt.t.normalize();
        }

        // Point to point correspondences
        instance.p1_.reserve(options.n_point_point_);
        instance.x1_.reserve(options.n_point_point_);
        instance.p2_.reserve(options.n_point_point_);
        instance.x2_.reserve(options.n_point_point_);

        for (int j = 0; j < options.n_point_point_; ++j) {

            Eigen::Vector3d p1{0.0, 0.0, 0.0};
            Eigen::Vector3d p2{0.0, 0.0, 0.0};
            Eigen::Vector3d x1{coord_gen(random_engine), coord_gen(random_engine), 1.0};
            x1.normalize();
            Eigen::Vector3d X;

            if (options.generalized_) {
                p1 << offset_gen(random_engine), offset_gen(random_engine), offset_gen(random_engine);
                p2 << offset_gen(random_engine), offset_gen(random_engine), offset_gen(random_engine);

                if (j > 0 && j < options.generalized_first_cam_obs_) {
                    p1 = instance.p1_[0];
                    p2 = instance.p2_[0];
                }
            }

            X = instance.scale_gt * p1 + x1 * depth_gen(random_engine);
            // Map into second image
            X = instance.pose_gt.R() * X + instance.pose_gt.t;

            Eigen::Vector3d x2 = (X - instance.scale_gt * p2).normalized();

            if (options.unknown_focal_) {
                // NYI
                assert(false);
            }

            // TODO: ensure FoV of second cameras as well...

            instance.p1_.push_back(p1);
            instance.x1_.push_back(x1);
            instance.p2_.push_back(p2);
            instance.x2_.push_back(x2);
        }

        problem_instances->push_back(instance);
    }
}

void generate_homography_problems(int n_problems, std::vector<HomographyProblemInstance> *problem_instances,
                                  const ProblemOptions &options) {
    problem_instances->clear();
    problem_instances->reserve(n_problems);

    double fov_scale = std::tan(options.camera_fov_ / 2.0 * kPI / 180.0);

    // Random generators
    std::default_random_engine random_engine;
    std::uniform_real_distribution<double> depth_gen(options.min_depth_, options.max_depth_);
    std::uniform_real_distribution<double> coord_gen(-fov_scale, fov_scale);
    std::uniform_real_distribution<double> focal_gen(options.min_focal_, options.max_focal_);
    std::uniform_real_distribution<double> distortion_gen(options.min_distortion_, options.max_distortion_);
    std::normal_distribution<double> direction_gen(0.0, 1.0);
    std::normal_distribution<double> offset_gen(0.0, 1.0);

    while (problem_instances->size() < n_problems) {
        HomographyProblemInstance instance;
        set_random_pose(instance.pose1_gt, options.upright_, options.planar_);
        set_random_pose(instance.pose2_gt, options.upright_, options.planar_);
        instance.pose1_gt.t.normalize();
        instance.pose2_gt.t.normalize();

        if (options.unknown_focal_) {
            instance.focal1_gt = focal_gen(random_engine);
            if (options.same_focal_) {
                instance.focal2_gt = instance.focal1_gt;
            } else {
                instance.focal2_gt = focal_gen(random_engine);
            }
        }
        if (options.unknown_distortion_) {
            instance.distortion1_gt = distortion_gen(random_engine);
            if (options.same_distortion_) {
                instance.distortion2_gt = instance.distortion1_gt;
            } else {
                instance.distortion2_gt = distortion_gen(random_engine);
            }
        }
        Eigen::Matrix<double, 3, 4> P1, P2;
        Eigen::Matrix3d K1, K2;
        P1 = instance.pose1_gt.Rt();
        P2 = instance.pose2_gt.Rt();
        if (options.unknown_focal_) {
            K1 = Eigen::Vector3d(instance.focal1_gt, instance.focal1_gt, 1).asDiagonal();
            K2 = Eigen::Vector3d(instance.focal2_gt, instance.focal2_gt, 1).asDiagonal();
            P1 = K1 * P1;
            P2 = K2 * P2;
        }

        // Point to point correspondences
        instance.x1_.reserve(options.n_point_point_);
        instance.x2_.reserve(options.n_point_point_);

        /*
        // Generate points
        Eigen::Vector3d n;
        if (options.ground_plane_) {
            n << 0, 1, 0;
        } else {
            n << direction_gen(random_engine), direction_gen(random_engine), direction_gen(random_engine);
            n.normalize();
        }

        // Choose depth of plane such that center point of image 1 is at depth d
        double d_center = depth_gen(random_engine);
        double alpha = d_center / n(2);
        // plane is n'*X = alpha
        // ground truth homography
        instance.H_gt = alpha * instance.pose_gt.R() + instance.pose_gt.t * n.transpose();
        */
        Eigen::Matrix3d tmp1 = P1(Eigen::seq(0, 2), {0, 2, 3});
        Eigen::Matrix3d tmp2 = P2(Eigen::seq(0, 2), {0, 2, 3});
        instance.H_gt = ((tmp1.transpose()).colPivHouseholderQr().solve(tmp2.transpose())).transpose();

        bool failed_instance = false;
        for (int j = 0; j < options.n_point_point_; ++j) {
            bool point_okay = false;
            for (int trials = 0; trials < 10; ++trials) {
                // Eigen::Vector3d x1{coord_gen(random_engine), coord_gen(random_engine), 1.0};
                // x1.normalize();
                Eigen::Vector3d X{coord_gen(random_engine), 0, coord_gen(random_engine)};

                // compute depth
                //double lambda = alpha / n.dot(x1);
                //X = x1 * lambda;
                // Map into second image
                //X = instance.pose_gt.R() * X + instance.pose_gt.t;

                //Eigen::Vector3d x2 = X.normalized();
                Eigen::Vector3d x1 = (P1 * X.homogeneous()).normalized();
                Eigen::Vector3d x2 = (P2 * X.homogeneous()).normalized();

                // Check consistency
                Eigen::Vector3d z = instance.H_gt * x1;
                double err = 1.0 - std::abs(z.normalized().dot(x2));
                if (err > 1e-14) {
                    // Something went wrong...
                    assert(false);
                }
                

                /*
                // Check cheirality
                if (x2(2) < 0 || lambda < 0) {
                    // try to generate another point
                    continue;
                }

                // Check FoV of second camera
                Eigen::Vector2d x2h = x2.hnormalized();
                if (x2h(0) < -fov_scale || x2h(0) > fov_scale || x2h(1) < -fov_scale || x2h(1) > fov_scale) {
                    // try to generate another point
                    continue;
                }
                */
                if (options.unknown_distortion_) {
                    x1 = poselib::radialdistort(x1.hnormalized(), instance.distortion1_gt).colwise().homogeneous();
                    x2 = poselib::radialdistort(x2.hnormalized(), instance.distortion2_gt).colwise().homogeneous();
                }

                instance.x1_.push_back(x1);
                instance.x2_.push_back(x2);
                point_okay = true;
                break;
            }
            if (!point_okay) {
                failed_instance = true;
                break;
            }
        }
        if (failed_instance) {
            continue;
        }
        problem_instances->push_back(instance);
    }
}

}; // namespace poselib
