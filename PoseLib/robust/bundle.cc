// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#if __GNUC__ >= 12
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif

#include "bundle.h"

#include "PoseLib/robust/jacobian_impl.h"
#include "PoseLib/robust/lm_impl.h"
#include "PoseLib/robust/robust_loss.h"

#include <iostream>

namespace poselib {

////////////////////////////////////////////////////////////////////////
// Below here we have wrappers for the refinement
// These are super messy due to the loss functions being templated
// and the hack we use to handle weights
//   (see UniformWeightVector in jacobian_impl.h)

#define SWITCH_LOSS_FUNCTIONS                                                                                          \
    case BundleOptions::LossType::TRIVIAL:                                                                             \
        SWITCH_LOSS_FUNCTION_CASE(TrivialLoss);                                                                        \
        break;                                                                                                         \
    case BundleOptions::LossType::TRUNCATED:                                                                           \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLoss);                                                                      \
        break;                                                                                                         \
    case BundleOptions::LossType::HUBER:                                                                               \
        SWITCH_LOSS_FUNCTION_CASE(HuberLoss);                                                                          \
        break;                                                                                                         \
    case BundleOptions::LossType::CAUCHY:                                                                              \
        SWITCH_LOSS_FUNCTION_CASE(CauchyLoss);                                                                         \
        break;                                                                                                         \
    case BundleOptions::LossType::TRUNCATED_LE_ZACH:                                                                   \
        SWITCH_LOSS_FUNCTION_CASE(TruncatedLossLeZach);                                                                \
        break;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Iteration callbacks (called after each LM iteration)

// Callback which prints debug info from the iterations
void print_iteration(const BundleStats &stats) {
    if (stats.iterations == 0) {
        std::cout << "initial_cost=" << stats.initial_cost << "\n";
    }
    std::cout << "iter=" << stats.iterations << ", cost=" << stats.cost << ", step=" << stats.step_norm
              << ", grad=" << stats.grad_norm << ", lambda=" << stats.lambda << "\n";
}

template <typename LossFunction> IterationCallback setup_callback(const BundleOptions &opt, LossFunction &loss_fn) {
    if (opt.verbose) {
        return print_iteration;
    } else {
        return nullptr;
    }
}

// For using the IRLS scheme proposed by Le and Zach 3DV2021, we have a callback
// for each iteration which updates the mu parameter
template <> IterationCallback setup_callback(const BundleOptions &opt, TruncatedLossLeZach &loss_fn) {
    if (opt.verbose) {
        return [&loss_fn](const BundleStats &stats) {
            print_iteration(stats);
            loss_fn.mu *= TruncatedLossLeZach::alpha;
        };
    } else {
        return [&loss_fn](const BundleStats &stats) { loss_fn.mu *= TruncatedLossLeZach::alpha; };
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Absolute pose with points (PnP)

// Interface for calibrated camera
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                          const BundleOptions &opt, const std::vector<double> &weights) {
    poselib::Camera camera;
    camera.model_id = NullCameraModel::model_id;
    return bundle_adjust(x, X, camera, pose, opt);
}

template <typename WeightType, typename CameraModel, typename LossFunction>
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const Camera &camera,
                          CameraPose *pose, const BundleOptions &opt, const WeightType &weights) {
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    CameraJacobianAccumulator<CameraModel, LossFunction, WeightType> accum(x, X, camera, loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename WeightType, typename CameraModel>
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const Camera &camera,
                          CameraPose *pose, const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return bundle_adjust<WeightType, CameraModel, LossFunction>(x, X, camera, pose, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

template <typename WeightType>
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const Camera &camera,
                          CameraPose *pose, const BundleOptions &opt, const WeightType &weights) {
    switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        return bundle_adjust<WeightType, Model>(x, X, camera, pose, opt, weights);                                     \
    }
        SWITCH_CAMERA_MODELS
#undef SWITCH_CAMERA_MODEL_CASE
    default:
        return BundleStats();
    }
}

// Entry point for PnP refinement
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, const Camera &camera,
                          CameraPose *pose, const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x.size()) {
        return bundle_adjust<std::vector<double>>(x, X, camera, pose, opt, weights);
    } else {
        return bundle_adjust<UniformWeightVector>(x, X, camera, pose, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Absolute pose with points and lines (PnPL)
// Note that we currently do not support different camera models here
// TODO: decide how to handle lines for non-linear camera models...

template <typename PointWeightType, typename LineWeightType, typename PointLossFunction, typename LineLossFunction>
BundleStats bundle_adjust(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                          const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, CameraPose *pose,
                          const BundleOptions &opt, const BundleOptions &opt_line, const PointWeightType &weights_pts,
                          const LineWeightType &weights_lines) {
    PointLossFunction pt_loss_fn(opt.loss_scale);
    LineLossFunction line_loss_fn(opt_line.loss_scale);
    IterationCallback callback = setup_callback(opt, pt_loss_fn);
    PointLineJacobianAccumulator<PointLossFunction, LineLossFunction, PointWeightType, LineWeightType> accum(
        points2D, points3D, lines2D, lines3D, pt_loss_fn, line_loss_fn, weights_pts, weights_lines);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename PointWeightType, typename LineWeightType, typename PointLossFunction>
BundleStats bundle_adjust(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                          const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, CameraPose *pose,
                          const BundleOptions &opt, const BundleOptions &opt_line, const PointWeightType &weights_pts,
                          const LineWeightType &weights_lines) {
    switch (opt_line.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return bundle_adjust<PointWeightType, LineWeightType, PointLossFunction, LossFunction>(                            \
        points2D, points3D, lines2D, lines3D, pose, opt, opt_line, weights_pts, weights_lines);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

template <typename PointWeightType, typename LineWeightType>
BundleStats bundle_adjust(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                          const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, CameraPose *pose,
                          const BundleOptions &opt, const BundleOptions &opt_line, const PointWeightType &weights_pts,
                          const LineWeightType &weights_lines) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return bundle_adjust<PointWeightType, LineWeightType, LossFunction>(points2D, points3D, lines2D, lines3D, pose,    \
                                                                        opt, opt_line, weights_pts, weights_lines);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for PnPL refinement
BundleStats bundle_adjust(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                          const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, CameraPose *pose,
                          const BundleOptions &opt, const BundleOptions &opt_line,
                          const std::vector<double> &weights_pts, const std::vector<double> &weights_lines) {
    bool have_pts_weights = weights_pts.size() == points2D.size();
    bool have_line_weights = weights_lines.size() == lines2D.size();

    if (have_pts_weights && have_line_weights) {
        return bundle_adjust<std::vector<double>, std::vector<double>>(points2D, points3D, lines2D, lines3D, pose, opt,
                                                                       opt_line, weights_pts, weights_lines);
    } else if (have_pts_weights && !have_line_weights) {
        return bundle_adjust<std::vector<double>, UniformWeightVector>(points2D, points3D, lines2D, lines3D, pose, opt,
                                                                       opt_line, weights_pts, UniformWeightVector());
    } else if (!have_pts_weights && have_line_weights) {
        return bundle_adjust<UniformWeightVector, std::vector<double>>(points2D, points3D, lines2D, lines3D, pose, opt,
                                                                       opt_line, UniformWeightVector(), weights_lines);
    } else {
        return bundle_adjust<UniformWeightVector, UniformWeightVector>(
            points2D, points3D, lines2D, lines3D, pose, opt, opt_line, UniformWeightVector(), UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Generalized absolute pose with points (GPnP)

// Interface for calibrated camera
BundleStats generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x,
                                      const std::vector<std::vector<Point3D>> &X,
                                      const std::vector<CameraPose> &camera_ext, CameraPose *pose,
                                      const BundleOptions &opt, const std::vector<std::vector<double>> &weights) {
    std::vector<Camera> dummy_cameras;
    dummy_cameras.resize(x.size());
    for (size_t k = 0; k < x.size(); ++k) {
        dummy_cameras[k].model_id = -1;
    }
    return generalized_bundle_adjust(x, X, camera_ext, dummy_cameras, pose, opt, weights);
}

template <typename WeightType, typename LossFunction>
BundleStats generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x,
                                      const std::vector<std::vector<Point3D>> &X,
                                      const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &cameras,
                                      CameraPose *pose, const BundleOptions &opt, const WeightType &weights) {
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    GeneralizedCameraJacobianAccumulator<LossFunction, WeightType> accum(x, X, camera_ext, cameras, loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename WeightType>
BundleStats generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x,
                                      const std::vector<std::vector<Point3D>> &X,
                                      const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &cameras,
                                      CameraPose *pose, const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return generalized_bundle_adjust<WeightType, LossFunction>(x, X, camera_ext, cameras, pose, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for GPnP refinement
BundleStats generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x,
                                      const std::vector<std::vector<Point3D>> &X,
                                      const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &cameras,
                                      CameraPose *pose, const BundleOptions &opt,
                                      const std::vector<std::vector<double>> &weights) {

    if (weights.size() == x.size()) {
        return generalized_bundle_adjust<std::vector<std::vector<double>>>(x, X, camera_ext, cameras, pose, opt,
                                                                           weights);
    } else {
        return generalized_bundle_adjust<UniformWeightVectors>(x, X, camera_ext, cameras, pose, opt,
                                                               UniformWeightVectors());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Relative pose (essential matrix) refinement

template <typename WeightType, typename LossFunction>
BundleStats refine_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, CameraPose *pose,
                           const BundleOptions &opt, const WeightType &weights) {
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    RelativePoseJacobianAccumulator<LossFunction, WeightType> accum(x1, x2, loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename WeightType>
BundleStats refine_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, CameraPose *pose,
                           const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_relpose<WeightType, LossFunction>(x1, x2, pose, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for essential matrix refinement
BundleStats refine_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, CameraPose *pose,
                           const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        return refine_relpose<std::vector<double>>(x1, x2, pose, opt, weights);
    } else {
        return refine_relpose<UniformWeightVector>(x1, x2, pose, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Relative pose (essential matrix) refinement

template <typename WeightType, typename LossFunction>
BundleStats refine_shared_focal_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                        ImagePair *image_pair, const BundleOptions &opt, const WeightType &weights) {
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    SharedFocalRelativePoseJacobianAccumulator<LossFunction, WeightType> accum(x1, x2, loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, image_pair, opt, callback);
}

template <typename WeightType>
BundleStats refine_shared_focal_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                        ImagePair *image_pair, const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_shared_focal_relpose<WeightType, LossFunction>(x1, x2, image_pair, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for essential matrix refinement
BundleStats refine_shared_focal_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                        ImagePair *image_pair, const BundleOptions &opt,
                                        const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        return refine_shared_focal_relpose<std::vector<double>>(x1, x2, image_pair, opt, weights);
    } else {
        return refine_shared_focal_relpose<UniformWeightVector>(x1, x2, image_pair, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Uncalibrated relative pose (fundamental matrix) refinement

template <typename WeightType, typename LossFunction>
BundleStats refine_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *F,
                               const BundleOptions &opt, const WeightType &weights) {
    // We optimize over the SVD-based factorization from Bartoli and Sturm
    FactorizedFundamentalMatrix factorized_fund_mat(*F);
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    FundamentalJacobianAccumulator<LossFunction, WeightType> accum(x1, x2, loss_fn, weights);
    BundleStats stats = lm_impl<decltype(accum)>(accum, &factorized_fund_mat, opt, callback);
    *F = factorized_fund_mat.F();
    return stats;
}

template <typename WeightType>
BundleStats refine_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *F,
                               const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_fundamental<WeightType, LossFunction>(x1, x2, F, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for fundamental matrix refinement
BundleStats refine_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *F,
                               const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        return refine_fundamental<std::vector<double>>(x1, x2, F, opt, weights);
    } else {
        return refine_fundamental<UniformWeightVector>(x1, x2, F, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Homography matrix refinement

template <typename WeightType, typename LossFunction>
BundleStats refine_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *H,
                              const BundleOptions &opt, const WeightType &weights) {

    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    HomographyJacobianAccumulator<LossFunction, WeightType> accum(x1, x2, loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, H, opt, callback);
}

template <typename WeightType>
BundleStats refine_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *H,
                              const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_homography<WeightType, LossFunction>(x1, x2, H, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for fundamental matrix refinement
BundleStats refine_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *H,
                              const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        return refine_homography<std::vector<double>>(x1, x2, H, opt, weights);
    } else {
        return refine_homography<UniformWeightVector>(x1, x2, H, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Generalized relative pose refinement

template <typename WeightType, typename LossFunction>
BundleStats refine_generalized_relpose(const std::vector<PairwiseMatches> &matches,
                                       const std::vector<CameraPose> &camera1_ext,
                                       const std::vector<CameraPose> &camera2_ext, CameraPose *pose,
                                       const BundleOptions &opt, const WeightType &weights) {
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    GeneralizedRelativePoseJacobianAccumulator<LossFunction, WeightType> accum(matches, camera1_ext, camera2_ext,
                                                                               loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename WeightType>
BundleStats refine_generalized_relpose(const std::vector<PairwiseMatches> &matches,
                                       const std::vector<CameraPose> &camera1_ext,
                                       const std::vector<CameraPose> &camera2_ext, CameraPose *pose,
                                       const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_generalized_relpose<WeightType, LossFunction>(matches, camera1_ext, camera2_ext, pose, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for generalized relpose refinement
BundleStats refine_generalized_relpose(const std::vector<PairwiseMatches> &matches,
                                       const std::vector<CameraPose> &camera1_ext,
                                       const std::vector<CameraPose> &camera2_ext, CameraPose *pose,
                                       const BundleOptions &opt, const std::vector<std::vector<double>> &weights) {
    if (weights.size() == matches.size()) {
        return refine_generalized_relpose<std::vector<std::vector<double>>>(matches, camera1_ext, camera2_ext, pose,
                                                                            opt, weights);
    } else {
        return refine_generalized_relpose<UniformWeightVectors>(matches, camera1_ext, camera2_ext, pose, opt,
                                                                UniformWeightVectors());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Hybrid pose refinement (i.e. both 2D-3D and 2D-2D point correspondences)

template <typename AbsWeightType, typename RelWeightType, typename LossFunction>
BundleStats refine_hybrid_pose(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                               const std::vector<PairwiseMatches> &matches_2D_2D,
                               const std::vector<CameraPose> &map_ext, CameraPose *pose, const BundleOptions &opt,
                               double loss_scale_epipolar, const AbsWeightType &weights_abs,
                               const RelWeightType &weights_rel) {
    LossFunction loss_fn(opt.loss_scale);
    LossFunction loss_fn_epipolar(loss_scale_epipolar);
    // TODO: refactor such that the callback can handle multiple loss-functions
    //       currently this only affects TruncatedLossLeZach
    IterationCallback callback = setup_callback(opt, loss_fn);
    HybridPoseJacobianAccumulator<LossFunction, AbsWeightType, RelWeightType> accum(
        x, X, matches_2D_2D, map_ext, loss_fn, loss_fn_epipolar, weights_abs, weights_rel);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename AbsWeightType, typename RelWeightType>
BundleStats refine_hybrid_pose(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                               const std::vector<PairwiseMatches> &matches_2D_2D,
                               const std::vector<CameraPose> &map_ext, CameraPose *pose, const BundleOptions &opt,
                               double loss_scale_epipolar, const AbsWeightType &weights_abs,
                               const RelWeightType &weights_rel) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return refine_hybrid_pose<AbsWeightType, RelWeightType, LossFunction>(                                             \
        x, X, matches_2D_2D, map_ext, pose, opt, loss_scale_epipolar, weights_abs, weights_rel);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for hybrid pose refinement
BundleStats refine_hybrid_pose(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                               const std::vector<PairwiseMatches> &matches_2D_2D,
                               const std::vector<CameraPose> &map_ext, CameraPose *pose, const BundleOptions &opt,
                               double loss_scale_epipolar, const std::vector<double> &weights_abs,
                               const std::vector<std::vector<double>> &weights_rel) {
    bool have_abs_weights = weights_abs.size() == x.size();
    bool have_rel_weights = weights_rel.size() == matches_2D_2D.size();

    if (have_abs_weights && have_rel_weights) {
        return refine_hybrid_pose<std::vector<double>, std::vector<std::vector<double>>>(
            x, X, matches_2D_2D, map_ext, pose, opt, loss_scale_epipolar, weights_abs, weights_rel);
    } else if (have_abs_weights && !have_rel_weights) {
        return refine_hybrid_pose<std::vector<double>, UniformWeightVectors>(
            x, X, matches_2D_2D, map_ext, pose, opt, loss_scale_epipolar, weights_abs, UniformWeightVectors());
    } else if (!have_abs_weights && have_rel_weights) {
        return refine_hybrid_pose<UniformWeightVector, std::vector<std::vector<double>>>(
            x, X, matches_2D_2D, map_ext, pose, opt, loss_scale_epipolar, UniformWeightVector(), weights_rel);
    } else {
        return refine_hybrid_pose<UniformWeightVector, UniformWeightVectors>(x, X, matches_2D_2D, map_ext, pose, opt,
                                                                             loss_scale_epipolar, UniformWeightVector(),
                                                                             UniformWeightVectors());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// 1D-radial absolute pose refinement (1D-radial PnP)

template <typename WeightType, typename LossFunction>
BundleStats bundle_adjust_1D_radial(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                                    const BundleOptions &opt, const WeightType &weights) {
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    Radial1DJacobianAccumulator<LossFunction, WeightType> accum(x, X, loss_fn, weights);
    return lm_impl<decltype(accum)>(accum, pose, opt, callback);
}

template <typename WeightType>
BundleStats bundle_adjust_1D_radial(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                                    const BundleOptions &opt, const WeightType &weights) {
    switch (opt.loss_type) {
#define SWITCH_LOSS_FUNCTION_CASE(LossFunction)                                                                        \
    return bundle_adjust_1D_radial<WeightType, LossFunction>(x, X, pose, opt, weights);
        SWITCH_LOSS_FUNCTIONS
    default:
        return BundleStats();
    }
#undef SWITCH_LOSS_FUNCTION_CASE
}

// Entry point for 1D radial absolute pose refinement (Assumes that the image points are centered)
BundleStats bundle_adjust_1D_radial(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                                    const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x.size()) {
        return bundle_adjust_1D_radial<std::vector<double>>(x, X, pose, opt, weights);
    } else {
        return bundle_adjust_1D_radial<UniformWeightVector>(x, X, pose, opt, UniformWeightVector());
    }
}

#undef SWITCH_LOSS_FUNCTIONS

} // namespace poselib