#include "bundle.h"
#include <iostream>

namespace pose_lib {

template <typename JacobianAccumulator>
int bundle_adjust_impl(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const JacobianAccumulator &accum, CameraPose *pose, const BundleOptions &opt) {
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtr;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw;
    sw.setZero();

    // Compute initial cost
    double cost = accum.residual(x, X, *pose);

    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            accum.accumulate(x, X, *pose, JtJ, Jtr);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        JtJ(0, 0) += lambda;
        JtJ(1, 1) += lambda;
        JtJ(2, 2) += lambda;
        JtJ(3, 3) += lambda;
        JtJ(4, 4) += lambda;
        JtJ(5, 5) += lambda;

        Eigen::Matrix<double, 6, 1> sol = -JtJ.selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        if (sol.norm() < opt.step_tol) {
            break;
        }

        Eigen::Vector3d w = sol.block<3, 1>(0, 0);
        const double theta = w.norm();
        w /= theta;
        const double a = std::sin(theta);
        const double b = std::cos(theta);
        sw(0, 1) = -w(2);
        sw(0, 2) = w(1);
        sw(1, 2) = -w(0);
        sw(1, 0) = w(2);
        sw(2, 0) = -w(1);
        sw(2, 1) = w(0);

        CameraPose pose_new;
        pose_new.R = pose->R + (a * sw + (1 - b) * sw * sw) * pose->R;
        pose_new.t = pose->t + sol.block<3, 1>(3, 0);
        double cost_new = accum.residual(x, X, pose_new);

        if (cost_new < cost) {
            *pose = pose_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            JtJ(0, 0) -= lambda;
            JtJ(1, 1) -= lambda;
            JtJ(2, 2) -= lambda;
            JtJ(3, 3) -= lambda;
            JtJ(4, 4) -= lambda;
            JtJ(5, 5) -= lambda;
            lambda *= 10;
            recompute_jac = false;
        }
    }

    return iter;
}

template <typename LossFunction>
class CalibratedJacobianAccumulator {
  public:
    CalibratedJacobianAccumulator(const LossFunction &l) : loss_fcn(l) {}

  private:
    const LossFunction &loss_fcn;

  public:
    double residual(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const CameraPose &pose) const {
        double cost = 0;
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.R * X[i] + pose.t;
            const double inv_z = 1.0 / Z(2);
            const double r0 = Z(0) * inv_z - x[i](0);
            const double r1 = Z(1) * inv_z - x[i](1);
            const double r_squared = r0 * r0 + r1 * r1;
            cost += loss_fcn.loss(r_squared);
        }
        return cost;
    }
    void accumulate(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) const {
        JtJ.setZero();
        Jtr.setZero();
        // compute JtJ and Jtr
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d RX = pose.R * X[i];
            const Eigen::Vector3d Z = RX + pose.t;
            const double inv_z = 1.0 / Z(2);
            const double p0 = Z(0) * inv_z;
            const double p1 = Z(1) * inv_z;
            const double r0 = p0 - x[i](0);
            const double r1 = p1 - x[i](1);
            const double r_squared = r0 * r0 + r1 * r1;
            const double weight = loss_fcn.weight(r_squared);

            if(weight == 0.0) {
                continue;
            }

            const double RX0 = RX(0);
            const double RX1 = RX(1);
            const double RX2 = RX(2);

            // pre-compute common sub-expressions
            const double inv_z2 = inv_z * inv_z;
            const double inv_z_p0 = inv_z * p0;
            const double inv_z_p1 = inv_z * p1;
            const double inv_z_r0 = inv_z * r0;
            const double inv_z_r1 = inv_z * r1;
            const double t1 = (RX2 * inv_z + RX1 * inv_z_p1);
            const double t2 = (RX2 * inv_z + RX0 * inv_z_p0);

            Jtr(0) += weight * (-r1 * t1 - RX1 * inv_z_p0 * r0);
            Jtr(1) += weight * (r0 * t2 + RX0 * inv_z_p1 * r1);
            Jtr(2) += weight * (RX0 * inv_z_r1 - RX1 * inv_z_r0);
            Jtr(3) += weight * (inv_z_r0);
            Jtr(4) += weight * (inv_z_r1);
            Jtr(5) += weight * (-inv_z_p0 * r0 - inv_z_p1 * r1);

            JtJ(0, 0) += weight * (t1 * t1 + RX1 * RX1 * inv_z_p0 * inv_z_p0);
            JtJ(1, 0) += weight * (-RX0 * inv_z_p1 * t1 - RX1 * inv_z_p0 * t2);
            JtJ(2, 0) += weight * (RX1 * RX1 * inv_z * inv_z_p0 - RX0 * inv_z * t1);
            JtJ(3, 0) += weight * (-RX1 * inv_z * inv_z_p0);
            JtJ(4, 0) += weight * (-inv_z * t1);
            JtJ(5, 0) += weight * (inv_z_p1 * t1 + RX1 * inv_z_p0 * inv_z_p0);
            JtJ(1, 1) += weight * (t2 * t2 + RX0 * RX0 * inv_z_p1 * inv_z_p1);
            JtJ(2, 1) += weight * (RX0 * RX0 * inv_z * inv_z_p1 - RX1 * inv_z * t2);
            JtJ(3, 1) += weight * (inv_z * t2);
            JtJ(4, 1) += weight * (RX0 * inv_z * inv_z_p1);
            JtJ(5, 1) += weight * (-inv_z_p0 * t2 - RX0 * inv_z_p1 * inv_z_p1);
            JtJ(2, 2) += weight * (RX0 * RX0 * inv_z2 + RX1 * RX1 * inv_z2);
            JtJ(3, 2) += weight * (-RX1 * inv_z2);
            JtJ(4, 2) += weight * (RX0 * inv_z2);
            JtJ(5, 2) += weight * (RX1 * inv_z * inv_z_p0 - RX0 * inv_z * inv_z_p1);
            JtJ(3, 3) += weight * (inv_z2);
            JtJ(4, 3) += weight * (0);
            JtJ(5, 3) += weight * (-inv_z * inv_z_p0);
            JtJ(4, 4) += weight * (inv_z2);
            JtJ(5, 4) += weight * (-inv_z * inv_z_p1);
            JtJ(5, 5) += weight * (inv_z_p0 * inv_z_p0 + inv_z_p1 * inv_z_p1);
        }

        JtJ /= (x.size());
        Jtr /= (x.size());
    }
};

template <typename CameraModel, typename LossFunction>
class CameraModelJacobianAccumulator {
  public:
    CameraModelJacobianAccumulator(const Camera &cam, const LossFunction &loss) : camera(cam), loss_fcn(loss) {}

    double residual(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const CameraPose &pose) const {
        double cost = 0;
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.R * X[i] + pose.t;
            const double inv_z = 1.0 / Z(2);
            Eigen::Vector2d p(Z(0) * inv_z, Z(1) * inv_z);
            CameraModel::project(camera.params, p, &p);
            const double r0 = p(0) - x[i](0);
            const double r1 = p(1) - x[i](1);
            const double r_squared = r0 * r0 + r1 * r1;
            cost += loss_fcn.loss(r_squared);
        }
        return cost;
    }

    void accumulate(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) const {
        JtJ.setZero();
        Jtr.setZero();
        Eigen::Matrix2d Jcam, JtJcam;
        Eigen::Vector2d xp;
        // compute JtJ and Jtr
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d RX = pose.R * X[i];
            const Eigen::Vector3d Z = RX + pose.t;
            const double inv_z = 1.0 / Z(2);
            Eigen::Vector2d p(Z(0) * inv_z, Z(1) * inv_z);

            // Project with intrinsics
            CameraModel::project_with_jac(camera.params, p, &xp, &Jcam);
            JtJcam = Jcam.transpose() * Jcam;

            const double r0 = xp(0) - x[i](0);
            const double r1 = xp(1) - x[i](1);
            const double r_squared = r0 * r0 + r1 * r1;
            const double weight = loss_fcn.weight(r_squared);

            if(weight == 0.0) {
                continue;
            }

            const double RX0 = RX(0);
            const double RX1 = RX(1);
            const double RX2 = RX(2);

            // pre-compute common sub-expressions
            const double inv_z2 = inv_z * inv_z;
            const double inv_z_p0 = inv_z * p(0);
            const double inv_z_p1 = inv_z * p(1);
            const double inv_z_r0 = inv_z * r0;
            const double inv_z_r1 = inv_z * r1;
            const double t1 = (RX2 * inv_z + RX1 * inv_z_p1);
            const double t2 = (RX2 * inv_z + RX0 * inv_z_p0);

            Jtr(0) += weight * (-r0 * (Jcam(0, 1) * t1 + Jcam(0, 0) * RX1 * inv_z_p0) - r1 * (Jcam(1, 1) * t1 + Jcam(1, 0) * RX1 * inv_z_p0));
            Jtr(1) += weight * (r0 * (Jcam(0, 0) * t2 + Jcam(0, 1) * RX0 * inv_z_p1) + r1 * (Jcam(1, 0) * t2 + Jcam(1, 1) * RX0 * inv_z_p1));
            Jtr(2) += weight * (-r0 * (Jcam(0, 0) * RX1 * inv_z - Jcam(0, 1) * RX0 * inv_z) - r1 * (Jcam(1, 0) * RX1 * inv_z - Jcam(1, 1) * RX0 * inv_z));
            Jtr(3) += weight * (Jcam(0, 0) * inv_z_r0 + Jcam(1, 0) * inv_z_r1);
            Jtr(4) += weight * (Jcam(0, 1) * inv_z_r0 + Jcam(1, 1) * inv_z_r1);
            Jtr(5) += weight * (-r0 * (Jcam(0, 0) * inv_z_p0 + Jcam(0, 1) * inv_z_p1) - r1 * (Jcam(1, 0) * inv_z_p0 + Jcam(1, 1) * inv_z_p1));

            JtJ(0, 0) += weight * (t1 * (JtJcam(1, 1) * t1 + JtJcam(0, 1) * RX1 * inv_z_p0) + RX1 * inv_z_p0 * (JtJcam(1, 0) * t1 + JtJcam(0, 0) * RX1 * inv_z_p0));
            JtJ(1, 0) += weight * (-t1 * (JtJcam(0, 1) * t2 + JtJcam(1, 1) * RX0 * inv_z_p1) - RX1 * inv_z_p0 * (JtJcam(0, 0) * t2 + JtJcam(1, 0) * RX0 * inv_z_p1));
            JtJ(2, 0) += weight * (t1 * (JtJcam(0, 1) * RX1 * inv_z - JtJcam(1, 1) * RX0 * inv_z) + RX1 * inv_z_p0 * (JtJcam(0, 0) * RX1 * inv_z - JtJcam(1, 0) * RX0 * inv_z));
            JtJ(3, 0) += weight * (-JtJcam(0, 1) * inv_z * t1 - JtJcam(0, 0) * RX1 * inv_z * inv_z_p0);
            JtJ(4, 0) += weight * (-JtJcam(1, 1) * inv_z * t1 - JtJcam(1, 0) * RX1 * inv_z * inv_z_p0);
            JtJ(5, 0) += weight * (t1 * (JtJcam(0, 1) * inv_z_p0 + JtJcam(1, 1) * inv_z_p1) + RX1 * inv_z_p0 * (JtJcam(0, 0) * inv_z_p0 + JtJcam(1, 0) * inv_z_p1));
            JtJ(1, 1) += weight * (t2 * (JtJcam(0, 0) * t2 + JtJcam(1, 0) * RX0 * inv_z_p1) + RX0 * inv_z_p1 * (JtJcam(0, 1) * t2 + JtJcam(1, 1) * RX0 * inv_z_p1));
            JtJ(2, 1) += weight * (-t2 * (JtJcam(0, 0) * RX1 * inv_z - JtJcam(1, 0) * RX0 * inv_z) - RX0 * inv_z_p1 * (JtJcam(0, 1) * RX1 * inv_z - JtJcam(1, 1) * RX0 * inv_z));
            JtJ(3, 1) += weight * (JtJcam(0, 0) * inv_z * t2 + JtJcam(0, 1) * RX0 * inv_z * inv_z_p1);
            JtJ(4, 1) += weight * (JtJcam(1, 0) * inv_z * t2 + JtJcam(1, 1) * RX0 * inv_z * inv_z_p1);
            JtJ(5, 1) += weight * (-t2 * (JtJcam(0, 0) * inv_z_p0 + JtJcam(1, 0) * inv_z_p1) - RX0 * inv_z_p1 * (JtJcam(0, 1) * inv_z_p0 + JtJcam(1, 1) * inv_z_p1));
            JtJ(2, 2) += weight * (RX1 * inv_z * (JtJcam(0, 0) * RX1 * inv_z - JtJcam(1, 0) * RX0 * inv_z) - RX0 * inv_z * (JtJcam(0, 1) * RX1 * inv_z - JtJcam(1, 1) * RX0 * inv_z));
            JtJ(3, 2) += weight * (JtJcam(0, 1) * RX0 * inv_z2 - JtJcam(0, 0) * RX1 * inv_z2);
            JtJ(4, 2) += weight * (JtJcam(1, 1) * RX0 * inv_z2 - JtJcam(1, 0) * RX1 * inv_z2);
            JtJ(5, 2) += weight * (RX1 * inv_z * (JtJcam(0, 0) * inv_z_p0 + JtJcam(1, 0) * inv_z_p1) - RX0 * inv_z * (JtJcam(0, 1) * inv_z_p0 + JtJcam(1, 1) * inv_z_p1));
            JtJ(3, 3) += weight * (JtJcam(0, 0) * inv_z2);
            JtJ(4, 3) += weight * (JtJcam(1, 0) * inv_z2);
            JtJ(5, 3) += weight * (-inv_z * (JtJcam(0, 0) * inv_z_p0 + JtJcam(1, 0) * inv_z_p1));
            JtJ(4, 4) += weight * (JtJcam(1, 1) * inv_z2);
            JtJ(5, 4) += weight * (-inv_z * (JtJcam(0, 1) * inv_z_p0 + JtJcam(1, 1) * inv_z_p1));
            JtJ(5, 5) += weight * (inv_z_p0 * (JtJcam(0, 0) * inv_z_p0 + JtJcam(1, 0) * inv_z_p1) + inv_z_p1 * (JtJcam(0, 1) * inv_z_p0 + JtJcam(1, 1) * inv_z_p1));
        }

        JtJ /= (x.size());
        Jtr /= (x.size());
    }

  private:
    const Camera &camera;
    const LossFunction &loss_fcn;
};

// Robust loss functions
class TrivialLoss {
  public:
    double loss(double r2) const {
        return r2;
    }
    double weight(double r2) const {
        return 1.0;
    }
};

class TruncatedLoss {
  public:
    TruncatedLoss(double threshold) : squared_thr(threshold * threshold) {}
    double loss(double r2) const {
        return std::min(r2, squared_thr);
    }
    double weight(double r2) const {
        return (r2 < squared_thr) ? 1.0 : 0.0;
    }

  private:
    const double squared_thr;
};

class HuberLoss {
  public:
    HuberLoss(double threshold) : thr(threshold) {}
    double loss(double r2) const {        
        const double r = std::sqrt(r2);
        if(r <= thr) {
            return r2;
        } else {
            return 2.0*thr*(r - thr);
        }        
    }
    double weight(double r2) const {
        const double r = std::sqrt(r2);
        if(r <= thr) {
            return 1.0;
        } else {
            return thr / r;
        }        
    }

  private:
    const double thr;
};
class CauchyLoss {
public:
    CauchyLoss(double threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}
    double loss(double r2) const {
        return std::log1p(r2 * inv_sq_thr);
    }
    double weight(double r2) const {
        return std::max(std::numeric_limits<double>::min(), 1.0 / (1.0 + r2 * inv_sq_thr));
    }

  private:
    const double inv_sq_thr;
};

int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, CameraPose *pose, const BundleOptions &opt) {
    switch (opt.loss_type) {
    case BundleOptions::LossType::TRIVIAL: {
        TrivialLoss loss_fn;
        CalibratedJacobianAccumulator<TrivialLoss> accum(loss_fn);
        return bundle_adjust_impl<decltype(accum)>(x, X, accum, pose, opt);
    }
    case BundleOptions::LossType::TRUNCATED: {        
        TruncatedLoss loss_fn(opt.loss_scale);
        CalibratedJacobianAccumulator<TruncatedLoss> accum(loss_fn);
        return bundle_adjust_impl<decltype(accum)>(x, X, accum, pose, opt);
    }
    case BundleOptions::LossType::HUBER: {
        HuberLoss loss_fn(opt.loss_scale);
        CalibratedJacobianAccumulator<HuberLoss> accum(loss_fn);
        return bundle_adjust_impl<decltype(accum)>(x, X, accum, pose, opt);
    }
    case BundleOptions::LossType::CAUCHY: {
        CauchyLoss loss_fn(opt.loss_scale);
        CalibratedJacobianAccumulator<CauchyLoss> accum(loss_fn);
        return bundle_adjust_impl<decltype(accum)>(x, X, accum, pose, opt);
    }
    default:
        return -1;
    };
}

// helper function to dispatch to the correct camera model (we do it once here to avoid doing it in every iteration)
template <typename LossFunction>
int dispatch_bundle_camera_model(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt, const LossFunction &loss) {
    switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model) \
    case Model::model_id:               \
        return bundle_adjust_impl<CameraModelJacobianAccumulator<Model, decltype(loss)>>(x, X, CameraModelJacobianAccumulator<Model, decltype(loss)>(camera, loss), pose, opt);

        SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
    }
    return -1;
}

int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt) {
    // TODO try rescaling image camera.rescale(1.0 / camera.focal()) and image points
    switch (opt.loss_type) {
    case BundleOptions::LossType::TRIVIAL: {
        TrivialLoss loss_fn;
        return dispatch_bundle_camera_model<TrivialLoss>(x, X, camera, pose, opt, loss_fn);
    }
    case BundleOptions::LossType::TRUNCATED: {
        TruncatedLoss loss_fn(opt.loss_scale);
        return dispatch_bundle_camera_model<TruncatedLoss>(x, X, camera, pose, opt, loss_fn);
    }
    case BundleOptions::LossType::HUBER: {
        HuberLoss loss_fn(opt.loss_scale);
        return dispatch_bundle_camera_model<HuberLoss>(x, X, camera, pose, opt, loss_fn);
    }
    case BundleOptions::LossType::CAUCHY: {
        CauchyLoss loss_fn(opt.loss_scale);
        return dispatch_bundle_camera_model<CauchyLoss>(x, X, camera, pose, opt, loss_fn);
    }
    default:
        return -1;
    }
}

} // namespace pose_lib