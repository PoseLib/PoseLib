#include "bundle.h"

namespace pose_lib {

template <typename JacobianAccumulator>
int bundle_adjust_impl(const JacobianAccumulator &accum, CameraPose *pose, const BundleOptions &opt) {
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtr;
    double lambda = opt.initial_lambda;
    Eigen::Matrix3d sw;
    sw.setZero();

    // Compute initial cost
    double cost = accum.residual(*pose);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            accum.accumulate(*pose, JtJ, Jtr);
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
        pose_new.R = pose->R + pose->R * (a * sw + (1 - b) * sw * sw);
        pose_new.t = pose->t + pose->R * sol.block<3, 1>(3, 0);
        double cost_new = accum.residual(pose_new);

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

template <typename CameraModel, typename LossFunction>
class CameraJacobianAccumulator {
  public:
    CameraJacobianAccumulator(
        const std::vector<Eigen::Vector2d> &points2D,
        const std::vector<Eigen::Vector3d> &points3D,
        const Camera &cam, const LossFunction &loss) : x(points2D), X(points3D), camera(cam), loss_fcn(loss) {}

    double residual(const CameraPose &pose) const {
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

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    void accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) const {

        Eigen::Matrix2d Jcam;
        Jcam.setIdentity(); // we initialize to identity here (this is for the calibrated case)
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.R * X[i] + pose.t;
            const Eigen::Vector2d z = Z.hnormalized();

            // Project with intrinsics
            Eigen::Vector2d zp = z;
            CameraModel::project_with_jac(camera.params, z, &zp, &Jcam);

            // Setup residual
            Eigen::Vector2d r = zp - x[i];
            const double r_squared = r.squaredNorm();
            const double weight = loss_fcn.weight(r_squared) / static_cast<double>(x.size());

            if (weight == 0.0) {
                continue;
            }

            // Compute jacobian w.r.t. Z (times R)
            Eigen::Matrix<double, 2, 3> dZ;
            dZ.block<2, 2>(0, 0) = Jcam;
            dZ.col(2) = -Jcam * z;
            dZ *= 1.0 / Z(2);
            dZ *= pose.R;

            const double X0 = X[i](0);
            const double X1 = X[i](1);
            const double X2 = X[i](2);
            const double dZtdZ_0_0 = weight * dZ.col(0).dot(dZ.col(0));
            const double dZtdZ_1_0 = weight * dZ.col(1).dot(dZ.col(0));
            const double dZtdZ_1_1 = weight * dZ.col(1).dot(dZ.col(1));
            const double dZtdZ_2_0 = weight * dZ.col(2).dot(dZ.col(0));
            const double dZtdZ_2_1 = weight * dZ.col(2).dot(dZ.col(1));
            const double dZtdZ_2_2 = weight * dZ.col(2).dot(dZ.col(2));
            JtJ(0, 0) += X2 * (X2 * dZtdZ_1_1 - X1 * dZtdZ_2_1) + X1 * (X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1);
            JtJ(1, 0) += -X2 * (X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1) - X1 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
            JtJ(2, 0) += X1 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0) - X2 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
            JtJ(3, 0) += X1 * dZtdZ_2_0 - X2 * dZtdZ_1_0;
            JtJ(4, 0) += X1 * dZtdZ_2_1 - X2 * dZtdZ_1_1;
            JtJ(5, 0) += X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1;
            JtJ(1, 1) += X2 * (X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0) + X0 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
            JtJ(2, 1) += -X2 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) - X0 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0);
            JtJ(3, 1) += X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0;
            JtJ(4, 1) += X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1;
            JtJ(5, 1) += X2 * dZtdZ_2_0 - X0 * dZtdZ_2_2;
            JtJ(2, 2) += X1 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) + X0 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
            JtJ(3, 2) += X0 * dZtdZ_1_0 - X1 * dZtdZ_0_0;
            JtJ(4, 2) += X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0;
            JtJ(5, 2) += X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0;
            JtJ(3, 3) += dZtdZ_0_0;
            JtJ(4, 3) += dZtdZ_1_0;
            JtJ(5, 3) += dZtdZ_2_0;
            JtJ(4, 4) += dZtdZ_1_1;
            JtJ(5, 4) += dZtdZ_2_1;
            JtJ(5, 5) += dZtdZ_2_2;
            r *= weight;
            Jtr(0) += (r(0) * (X1 * dZ(0, 2) - X2 * dZ(0, 1)) + r(1) * (X1 * dZ(1, 2) - X2 * dZ(1, 1)));
            Jtr(1) += (-r(0) * (X0 * dZ(0, 2) - X2 * dZ(0, 0)) - r(1) * (X0 * dZ(1, 2) - X2 * dZ(1, 0)));
            Jtr(2) += (r(0) * (X0 * dZ(0, 1) - X1 * dZ(0, 0)) + r(1) * (X0 * dZ(1, 1) - X1 * dZ(1, 0)));
            Jtr(3) += (dZ(0, 0) * r(0) + dZ(1, 0) * r(1));
            Jtr(4) += (dZ(0, 1) * r(0) + dZ(1, 1) * r(1));
            Jtr(5) += (dZ(0, 2) * r(0) + dZ(1, 2) * r(1));
        }
    }

  private:
    const std::vector<Eigen::Vector2d> &x;
    const std::vector<Eigen::Vector3d> &X;
    const Camera &camera;
    const LossFunction &loss_fcn;
};

template <typename LossFunction>
class GeneralizedCameraJacobianAccumulator {
  public:
    GeneralizedCameraJacobianAccumulator(
        const std::vector<std::vector<Eigen::Vector2d>> &points2D,
        const std::vector<std::vector<Eigen::Vector3d>> &points3D,
        const std::vector<CameraPose> &camera_ext,
        const std::vector<Camera> &camera_int,
        const LossFunction &l) : num_cams(points2D.size()), x(points2D), X(points3D),
                                 rig_poses(camera_ext), cameras(camera_int), loss_fn(l) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        for (size_t k = 0; k < num_cams; ++k) {
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.R = rig_poses[k].R * pose.R;
            full_pose.t = rig_poses[k].R * pose.t + rig_poses[k].t;

            switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                             \
    case Model::model_id: {                                                                         \
        CameraJacobianAccumulator<Model, decltype(loss_fn)> accum(x[k], X[k], cameras[k], loss_fn); \
        cost += accum.residual(full_pose);                                                          \
        break;                                                                                      \
    }
                SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
            }
        }
        return cost;
    }

    void accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) const {
        for (size_t k = 0; k < num_cams; ++k) {
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.R = rig_poses[k].R * pose.R;
            full_pose.t = rig_poses[k].R * pose.t + rig_poses[k].t;

            switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                             \
    case Model::model_id: {                                                                         \
        CameraJacobianAccumulator<Model, decltype(loss_fn)> accum(x[k], X[k], cameras[k], loss_fn); \
        accum.accumulate(full_pose, JtJ, Jtr);                                                      \
        break;                                                                                      \
    }
                SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
            }
        }
    }

  private:
    const size_t num_cams;
    const std::vector<std::vector<Eigen::Vector2d>> &x;
    const std::vector<std::vector<Eigen::Vector3d>> &X;
    const std::vector<CameraPose> &rig_poses;
    const std::vector<Camera> &cameras;
    const LossFunction &loss_fn;
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
        if (r <= thr) {
            return r2;
        } else {
            return 2.0 * thr * (r - thr);
        }
    }
    double weight(double r2) const {
        const double r = std::sqrt(r2);
        if (r <= thr) {
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

struct IdentityCameraModel {
    static void project(const std::vector<double> &params, const Eigen::Vector2d &x, Eigen::Vector2d *xp) {}
    static void project_with_jac(const std::vector<double> &params, const Eigen::Vector2d &x, Eigen::Vector2d *xp, Eigen::Matrix2d *jac) {}
};

int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, CameraPose *pose, const BundleOptions &opt) {
    pose_lib::Camera camera;
    camera.model_id = -1;
    return bundle_adjust(x, X, camera, pose, opt);
}

// helper function to dispatch to the correct camera model (we do it once here to avoid doing it in every iteration)
template <typename LossFunction>
int dispatch_bundle_camera_model(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt, const LossFunction &loss) {
    switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model) \
    case Model::model_id:               \
        return bundle_adjust_impl<CameraJacobianAccumulator<Model, decltype(loss)>>(CameraJacobianAccumulator<Model, decltype(loss)>(x, X, camera, loss), pose, opt);

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

int generalized_bundle_adjust(const std::vector<std::vector<Eigen::Vector2d>> &x, const std::vector<std::vector<Eigen::Vector3d>> &X, const std::vector<CameraPose> &camera_ext, CameraPose *pose, const BundleOptions &opt) {
    std::vector<Camera> dummy_cameras;
    dummy_cameras.resize(x.size());
    for (size_t k = 0; k < x.size(); ++k) {
        dummy_cameras[k].model_id = -1;
    }
    return generalized_bundle_adjust(x, X, camera_ext, dummy_cameras, pose, opt);
}

int generalized_bundle_adjust(const std::vector<std::vector<Eigen::Vector2d>> &x, const std::vector<std::vector<Eigen::Vector3d>> &X, const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &cameras, CameraPose *pose, const BundleOptions &opt) {

    switch (opt.loss_type) {
    case BundleOptions::LossType::TRIVIAL: {
        TrivialLoss loss_fn;
        GeneralizedCameraJacobianAccumulator<TrivialLoss> accum(x, X, camera_ext, cameras, loss_fn);
        return bundle_adjust_impl<decltype(accum)>(accum, pose, opt);
    }
    case BundleOptions::LossType::TRUNCATED: {
        TruncatedLoss loss_fn(opt.loss_scale);
        GeneralizedCameraJacobianAccumulator<TruncatedLoss> accum(x, X, camera_ext, cameras, loss_fn);
        return bundle_adjust_impl<decltype(accum)>(accum, pose, opt);
    }
    case BundleOptions::LossType::HUBER: {
        HuberLoss loss_fn(opt.loss_scale);
        GeneralizedCameraJacobianAccumulator<HuberLoss> accum(x, X, camera_ext, cameras, loss_fn);
        return bundle_adjust_impl<decltype(accum)>(accum, pose, opt);
    }
    case BundleOptions::LossType::CAUCHY: {
        CauchyLoss loss_fn(opt.loss_scale);
        GeneralizedCameraJacobianAccumulator<CauchyLoss> accum(x, X, camera_ext, cameras, loss_fn);
        return bundle_adjust_impl<decltype(accum)>(accum, pose, opt);
    }
    default:
        return -1;
    };
    return 0;
}

} // namespace pose_lib