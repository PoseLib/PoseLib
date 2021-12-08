#include "bundle.h"
#include "jacobian_impl.h"
#include "lm_impl.h"
#include "robust_loss.h"
namespace pose_lib {

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

BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose, const BundleOptions &opt, const std::vector<double> &weights) {
    pose_lib::Camera camera;
    camera.model_id = NullCameraModel::model_id;
    return bundle_adjust(x, X, camera, pose, opt);
}

BundleStats bundle_adjust(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                          const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                          CameraPose *pose, const BundleOptions &opt,
                          const std::vector<double> &weights_pts, const std::vector<double> &weights_line) {

    if (weights_pts.size() == points2D.size() && weights_line.size() == lines2D.size()) {
        // We have per-residual weights

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                       \
    {                                                                                                 \
        LossFunction loss_fn(opt.loss_scale);                                                         \
        PointLineJacobianAccumulator<LossFunction, std::vector<double>, std::vector<double>>          \
            accum(points2D, points3D, lines2D, lines3D, loss_fn, loss_fn, weights_pts, weights_line); \
        return lm_impl<decltype(accum)>(accum, pose, opt);                                            \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return BundleStats();
        };

    } else {
        // Uniform weights for the residuals

        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                            \
    {                                                                      \
        LossFunction loss_fn(opt.loss_scale);                              \
        PointLineJacobianAccumulator<LossFunction>                         \
            accum(points2D, points3D, lines2D, lines3D, loss_fn, loss_fn); \
        return lm_impl<decltype(accum)>(accum, pose, opt);                 \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return BundleStats();
        };
    }
}

// helper function to dispatch to the correct camera model (we do it once here to avoid doing it in every iteration)
template <typename LossFunction>
BundleStats dispatch_bundle_camera_model(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt, const LossFunction &loss, const std::vector<double> &weights) {
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
    return BundleStats();
}

BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt, const std::vector<double> &weights) {
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
        return BundleStats();
    };
}

BundleStats generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x, const std::vector<std::vector<Point3D>> &X, const std::vector<CameraPose> &camera_ext, CameraPose *pose, const BundleOptions &opt, const std::vector<std::vector<double>> &weights) {
    std::vector<Camera> dummy_cameras;
    dummy_cameras.resize(x.size());
    for (size_t k = 0; k < x.size(); ++k) {
        dummy_cameras[k].model_id = -1;
    }
    return generalized_bundle_adjust(x, X, camera_ext, dummy_cameras, pose, opt, weights);
}

BundleStats generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x, const std::vector<std::vector<Point3D>> &X, const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &cameras, CameraPose *pose, const BundleOptions &opt, const std::vector<std::vector<double>> &weights) {

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
            return BundleStats();
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
            return BundleStats();
        };
    }
}

BundleStats refine_relpose(const std::vector<Point2D> &x1,
                           const std::vector<Point2D> &x2,
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
            return BundleStats();
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
            return BundleStats();
        };
    }
}

BundleStats refine_fundamental(const std::vector<Point2D> &x1,
                               const std::vector<Point2D> &x2,
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
        BundleStats stats = lm_impl<decltype(accum)>(accum, &FF, opt);                                     \
        *F = FF.F();                                                                                       \
        return stats;                                                                                      \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return BundleStats();
        };
    } else {

        // Uniformly weighted residuals
        switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                              \
    {                                                                        \
        LossFunction loss_fn(opt.loss_scale);                                \
        FundamentalJacobianAccumulator<LossFunction> accum(x1, x2, loss_fn); \
        BundleStats stats = lm_impl<decltype(accum)>(accum, &FF, opt);       \
        *F = FF.F();                                                         \
        return stats;                                                        \
    }

            SWITCH_LOSS_FUNCTIONS

#undef SWITCH_LOSS_FUNCTION_CASE

        default:
            return BundleStats();
        };
    }
}

BundleStats refine_generalized_relpose(const std::vector<PairwiseMatches> &matches,
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
            return BundleStats();
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
            return BundleStats();
        };
    }
}

BundleStats refine_hybrid_pose(const std::vector<Point2D> &x,
                               const std::vector<Point3D> &X,
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
            return BundleStats();
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
            return BundleStats();
        };
    }
}

// Minimizes the 1D radial reprojection error. Assumes that the image points are centered
// Returns number of iterations.
BundleStats bundle_adjust_1D_radial(const std::vector<Point2D> &x,
                                    const std::vector<Point3D> &X,
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
            return BundleStats();
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
            return BundleStats();
        };
    }
}

#undef SWITCH_LOSS_FUNCTIONS

} // namespace pose_lib