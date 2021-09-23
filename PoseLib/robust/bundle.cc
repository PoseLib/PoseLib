#include "bundle.h"
#include "jacobian_impl.h"
#include "robust_loss.h"

namespace pose_lib {

/*
 Templated implementation of Levenberg-Marquadt.

 The Problem class must provide
    Problem::num_params - number of parameters to optimize over
    Problem::params_t - type for the parameters which optimize over
    Problem::accumulate(param, JtJ, Jtr) - compute jacobians
    Problem::residual(param) - compute the current residuals
    Problem::step(delta_params, param) - take a step in parameter space

    Check jacobian_impl.h for examples
*/
template <typename Problem, typename Param = typename Problem::param_t>
int lm_impl(Problem &problem, Param *parameters, const BundleOptions &opt) {
    constexpr int n_params = Problem::num_params;
    Eigen::Matrix<double, n_params, n_params> JtJ;
    Eigen::Matrix<double, n_params, 1> Jtr;
    double lambda = opt.initial_lambda;

    // Compute initial cost
    double cost = problem.residual(*parameters);
    bool recompute_jac = true;
    int iter;
    for (iter = 0; iter < opt.max_iterations; ++iter) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            JtJ.setZero();
            Jtr.setZero();
            problem.accumulate(*parameters, JtJ, Jtr);
            if (Jtr.norm() < opt.gradient_tol) {
                break;
            }
        }

        // Add dampening
        for (size_t k = 0; k < n_params; ++k) {
            JtJ(k, k) += lambda;
        }

        Eigen::Matrix<double, n_params, 1> sol = -JtJ.template selfadjointView<Eigen::Lower>().llt().solve(Jtr);

        if (sol.norm() < opt.step_tol) {
            break;
        }

        Param parameters_new = problem.step(sol, *parameters);

        double cost_new = problem.residual(parameters_new);

        if (cost_new < cost) {
            *parameters = parameters_new;
            lambda /= 10;
            cost = cost_new;
            recompute_jac = true;
        } else {
            // Remove dampening
            for (size_t k = 0; k < n_params; ++k) {
                JtJ(k, k) -= lambda;
            }
            lambda *= 10;
            recompute_jac = false;
        }
    }

    return iter;
}


////////////////////////////////////////////////////////////////////////
// Below here we have wrappers for the refinement
// These are super messy due to the loss functions being templated
// and the hack we use to handle weights
//   (see UniformWeightVector in jacobian_impl.h)
// TODO: Figure out a good way of refactoring this without performance penalties


#define SWITCH_LOSS_FUNCTIONS                     \
    case BundleOptions::LossType::TRIVIAL:        \
        SWITCH_LOSS_FUNCTION_CASE(TrivialLoss);   \
        break;                                    \
    case BundleOptions::LossType::TRUNCATED:      \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLoss); \
        break;                                    \
    case BundleOptions::LossType::HUBER:          \
        SWITCH_LOSS_FUNCTION_CASE(HuberLoss);     \
        break;                                    \
    case BundleOptions::LossType::CAUCHY:         \
        SWITCH_LOSS_FUNCTION_CASE(CauchyLoss);    \
        break;


int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, CameraPose *pose, const BundleOptions &opt, const std::vector<double> &weights) {
    pose_lib::Camera camera;
    camera.model_id = NullCameraModel::model_id;
    return bundle_adjust(x, X, camera, pose, opt);
}

// helper function to dispatch to the correct camera model (we do it once here to avoid doing it in every iteration)
template <typename LossFunction>
int dispatch_bundle_camera_model(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt, const LossFunction &loss, const std::vector<double> &weights) {
    if (weights.size() == x.size()) {
        // We have per-residual weights
        switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                       \
    case Model::model_id: {                                                   \
        CameraJacobianAccumulator<Model, decltype(loss), std::vector<double>> \
            accum(x, X, camera, loss, weights);                               \
        return lm_impl<decltype(accum)>(accum, pose, opt);                    \
    }

            SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
        }
    } else {
        // Uniform weights

        switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                    \
    case Model::model_id: {                                \
        CameraJacobianAccumulator<Model, decltype(loss)>   \
            accum(x, X, camera, loss);                     \
        return lm_impl<decltype(accum)>(accum, pose, opt); \
    }
            SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
        }
    }
    return -1;
}

int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt, const std::vector<double> &weights) {
    // TODO try rescaling image camera.rescale(1.0 / camera.focal()) and image points

    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                       \
    {                                                                                                 \
        LossFunction loss_fn(opt.loss_scale);                                                         \
        return dispatch_bundle_camera_model<LossFunction>(x, X, camera, pose, opt, loss_fn, weights); \
    }

        SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

    default:
        return -1;
    };
}

int generalized_bundle_adjust(const std::vector<std::vector<Eigen::Vector2d>> &x, const std::vector<std::vector<Eigen::Vector3d>> &X, const std::vector<CameraPose> &camera_ext, CameraPose *pose, const BundleOptions &opt, const std::vector<std::vector<double>> &weights) {
    std::vector<Camera> dummy_cameras;
    dummy_cameras.resize(x.size());
    for (size_t k = 0; k < x.size(); ++k) {
        dummy_cameras[k].model_id = -1;
    }
    return generalized_bundle_adjust(x, X, camera_ext, dummy_cameras, pose, opt, weights);
}

int generalized_bundle_adjust(const std::vector<std::vector<Eigen::Vector2d>> &x, const std::vector<std::vector<Eigen::Vector3d>> &X, const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &cameras, CameraPose *pose, const BundleOptions &opt, const std::vector<std::vector<double>> &weights) {

    if (weights.size() == x.size()) {
        // We have per-residual weights

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                              \
    {                                                                                        \
        LossFunction loss_fn(opt.loss_scale);                                                \
        GeneralizedCameraJacobianAccumulator<LossFunction, std::vector<std::vector<double>>> \
            accum(x, X, camera_ext, cameras, loss_fn, weights);                              \
        return lm_impl<decltype(accum)>(accum, pose, opt);                                   \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };

    } else {
        // Uniform weights for the residuals

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)            \
    {                                                      \
        LossFunction loss_fn(opt.loss_scale);              \
        GeneralizedCameraJacobianAccumulator<LossFunction> \
            accum(x, X, camera_ext, cameras, loss_fn);     \
        return lm_impl<decltype(accum)>(accum, pose, opt); \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }
}

int refine_relpose(const std::vector<Eigen::Vector2d> &x1,
                   const std::vector<Eigen::Vector2d> &x2,
                   CameraPose *pose, const BundleOptions &opt,
                   const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        // We have per-residual weights

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                             \
    {                                                                                                       \
        LossFunction loss_fn(opt.loss_scale);                                                               \
        RelativePoseJacobianAccumulator<LossFunction, std::vector<double>> accum(x1, x2, loss_fn, weights); \
        return lm_impl<decltype(accum)>(accum, pose, opt);                                                  \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    } else {

        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                               \
    {                                                                         \
        LossFunction loss_fn(opt.loss_scale);                                 \
        RelativePoseJacobianAccumulator<LossFunction> accum(x1, x2, loss_fn); \
        return lm_impl<decltype(accum)>(accum, pose, opt);                    \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

int refine_fundamental(const std::vector<Eigen::Vector2d> &x1,
                       const std::vector<Eigen::Vector2d> &x2,
                       Eigen::Matrix3d *F,
                       const BundleOptions &opt,
                       const std::vector<double> &weights) {

    FactorizedFundamentalMatrix FF(*F);
    if (weights.size() == x1.size()) {
        // We have per-residual weights

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                            \
    {                                                                                                      \
        LossFunction loss_fn(opt.loss_scale);                                                              \
        FundamentalJacobianAccumulator<LossFunction, std::vector<double>> accum(x1, x2, loss_fn, weights); \
        size_t iter = lm_impl<decltype(accum)>(accum, &FF, opt);                                           \
        *F = FF.F();                                                                                       \
        return iter;                                                                                       \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    } else {

        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                              \
    {                                                                        \
        LossFunction loss_fn(opt.loss_scale);                                \
        FundamentalJacobianAccumulator<LossFunction> accum(x1, x2, loss_fn); \
        size_t iter = lm_impl<decltype(accum)>(accum, &FF, opt);             \
        *F = FF.F();                                                         \
        return iter;                                                         \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

int refine_generalized_relpose(const std::vector<PairwiseMatches> &matches,
                               const std::vector<CameraPose> &camera1_ext, const std::vector<CameraPose> &camera2_ext,
                               CameraPose *pose, const BundleOptions &opt, const std::vector<std::vector<double>> &weights) {

    if (weights.size() == matches.size()) {

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                    \
    {                                                                                              \
        LossFunction loss_fn(opt.loss_scale);                                                      \
        GeneralizedRelativePoseJacobianAccumulator<LossFunction, std::vector<std::vector<double>>> \
            accum(matches, camera1_ext, camera2_ext, loss_fn, weights);                            \
        return lm_impl<decltype(accum)>(accum, pose, opt);                                         \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };

    } else {

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                  \
    {                                                            \
        LossFunction loss_fn(opt.loss_scale);                    \
        GeneralizedRelativePoseJacobianAccumulator<LossFunction> \
            accum(matches, camera1_ext, camera2_ext, loss_fn);   \
        return lm_impl<decltype(accum)>(accum, pose, opt);       \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

int refine_hybrid_pose(const std::vector<Eigen::Vector2d> &x,
                       const std::vector<Eigen::Vector3d> &X,
                       const std::vector<PairwiseMatches> &matches_2D_2D,
                       const std::vector<CameraPose> &map_ext,
                       CameraPose *pose, const BundleOptions &opt, double loss_scale_epipolar,
                       const std::vector<double> &weights_abs,
                       const std::vector<std::vector<double>> &weights_rel) {

    if (weights_abs.size() == x.size()) {
        // Per-residual weights

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                              \
    {                                                                                        \
        LossFunction loss_fn(opt.loss_scale);                                                \
        LossFunction loss_fn_epipolar(loss_scale_epipolar);                                  \
        HybridPoseJacobianAccumulator<LossFunction,                                          \
                                      std::vector<double>, std::vector<std::vector<double>>> \
            accum(x, X, matches_2D_2D, map_ext, loss_fn, loss_fn_epipolar,                   \
                  weights_abs, weights_rel);                                                 \
        return lm_impl<decltype(accum)>(accum, pose, opt);                                   \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };

    } else {
        // Uniform weights
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                             \
    {                                                                       \
        LossFunction loss_fn(opt.loss_scale);                               \
        LossFunction loss_fn_epipolar(loss_scale_epipolar);                 \
        HybridPoseJacobianAccumulator<LossFunction>                         \
            accum(x, X, matches_2D_2D, map_ext, loss_fn, loss_fn_epipolar); \
        return lm_impl<decltype(accum)>(accum, pose, opt);                  \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

// Minimizes the 1D radial reprojection error. Assumes that the image points are centered
// Returns number of iterations.
int bundle_adjust_1D_radial(const std::vector<Eigen::Vector2d> &x,
                            const std::vector<Eigen::Vector3d> &X,
                            CameraPose *pose,
                            const BundleOptions &opt,
                            const std::vector<double> &weights) {

    if (weights.size() == x.size()) {
        // We have per-residual weights

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                       \
    {                                                                                                 \
        LossFunction loss_fn(opt.loss_scale);                                                         \
        Radial1DJacobianAccumulator<LossFunction, std::vector<double>> accum(x, X, loss_fn, weights); \
        return lm_impl<decltype(accum)>(accum, pose, opt);                                            \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    } else {

        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                         \
    {                                                                   \
        LossFunction loss_fn(opt.loss_scale);                           \
        Radial1DJacobianAccumulator<LossFunction> accum(x, X, loss_fn); \
        return lm_impl<decltype(accum)>(accum, pose, opt);              \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return -1;
        };
    }

    return 0;
}

#undef SWITCH_LOSS_FUNCTIONS

} // namespace pose_lib