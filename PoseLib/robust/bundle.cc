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

#include "PoseLib/robust/optim/absolute.h"
#include "PoseLib/robust/optim/fundamental.h"
#include "PoseLib/robust/optim/generalized_absolute.h"
#include "PoseLib/robust/optim/generalized_relative.h"
#include "PoseLib/robust/optim/homography.h"
#include "PoseLib/robust/optim/hybrid.h"
#include "PoseLib/robust/optim/jacobian_accumulator.h"
#include "PoseLib/robust/optim/lm_impl.h"
#include "PoseLib/robust/robust_loss.h"

#include <iostream>
#include <memory>

namespace poselib {

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Iteration callbacks (called after each LM iteration)

// Callback which prints debug info from the iterations
void print_iteration(const BundleStats &stats, RobustLoss *loss_fn) {
    if (stats.iterations == 0) {
        std::cout << "initial_cost=" << stats.initial_cost << "\n";
    }
    std::cout << "iter=" << stats.iterations << ", cost=" << stats.cost << ", step=" << stats.step_norm
              << ", grad=" << stats.grad_norm << ", lambda=" << stats.lambda << "\n";
}

IterationCallback setup_callback(const BundleOptions &opt) {
    if (opt.loss_type == BundleOptions::TRUNCATED_LE_ZACH) {
        // For using the IRLS scheme proposed by Le and Zach 3DV2021, we have a callback
        // for each iteration which updates the mu parameter
        // Similar constructions could be used for graduated non-convexity stuff in the future.
        if (opt.verbose) {
            return [](const BundleStats &stats, RobustLoss *loss_fn) {
                print_iteration(stats, loss_fn);
                TruncatedLossLeZach *loss = static_cast<TruncatedLossLeZach *>(loss_fn);
                loss->mu *= TruncatedLossLeZach::alpha;
            };
        } else {
            return [](const BundleStats &stats, RobustLoss *loss_fn) {
                TruncatedLossLeZach *loss = static_cast<TruncatedLossLeZach *>(loss_fn);
                loss->mu *= TruncatedLossLeZach::alpha;
            };
        }
    } else {
        if (opt.verbose) {
            return print_iteration;
        } else {
            return nullptr;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Absolute pose with points (PnP)

// Interface for calibrated camera
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                          const BundleOptions &opt, const std::vector<double> &weights) {
    Image image;
    image.pose = *pose;
    image.camera.model_id = NullCameraModel::model_id;
    BundleStats stats = bundle_adjust(x, X, &image, opt);
    *pose = image.pose;
    return stats;
}

template <typename WeightType>
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, Image *image,
                          const BundleOptions &opt, const WeightType &weights) {
    std::vector<size_t> camera_refine_idx = image->camera.get_param_refinement_idx(opt);
    IterationCallback callback = setup_callback(opt);
    AbsolutePoseRefiner<WeightType> refiner(x, X, camera_refine_idx, weights);
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, image, opt, callback);
    return stats;
}

// Entry point for PnP refinement
BundleStats bundle_adjust(const std::vector<Point2D> &x, const std::vector<Point3D> &X, Image *image,
                          const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x.size()) {
        return bundle_adjust<std::vector<double>>(x, X, image, opt, weights);
    } else {
        return bundle_adjust<UniformWeightVector>(x, X, image, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Absolute pose with points and lines (PnPL)
// Note that we currently do not support different camera models here
// TODO: decide how to handle lines for non-linear camera models...

template <typename PointWeightType, typename LineWeightType>
BundleStats bundle_adjust(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                          const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, const Camera &camera,
                          CameraPose *pose, const BundleOptions &opt, const BundleOptions &opt_line,
                          const PointWeightType &weights_pts, const LineWeightType &weights_lines) {

    std::vector<size_t> camera_refine_idx = {};
    IterationCallback callback = setup_callback(opt);

    AbsolutePoseRefiner<PointWeightType> pts_refiner(points2D, points3D, camera_refine_idx, weights_pts);
    PinholeLineAbsolutePoseRefiner<LineWeightType> lin_refiner(lines2D, lines3D, weights_lines);
    HybridRefiner<Image> refiner;
    refiner.register_refiner(&pts_refiner);
    refiner.register_refiner(&lin_refiner);

    Image image(*pose, camera);
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, &image, opt, callback);
    *pose = image.pose;
    return stats;
}

// Entry point for PnPL refinement
BundleStats bundle_adjust(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                          const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D, const Camera &camera,
                          CameraPose *pose, const BundleOptions &opt, const BundleOptions &opt_line,
                          const std::vector<double> &weights_pts, const std::vector<double> &weights_lines) {
    bool have_pts_weights = weights_pts.size() == points2D.size();
    bool have_line_weights = weights_lines.size() == lines2D.size();

    if (have_pts_weights && have_line_weights) {
        return bundle_adjust<std::vector<double>, std::vector<double>>(points2D, points3D, lines2D, lines3D, camera,
                                                                       pose, opt, opt_line, weights_pts, weights_lines);
    } else if (have_pts_weights && !have_line_weights) {
        return bundle_adjust<std::vector<double>, UniformWeightVector>(
            points2D, points3D, lines2D, lines3D, camera, pose, opt, opt_line, weights_pts, UniformWeightVector());
    } else if (!have_pts_weights && have_line_weights) {
        return bundle_adjust<UniformWeightVector, std::vector<double>>(
            points2D, points3D, lines2D, lines3D, camera, pose, opt, opt_line, UniformWeightVector(), weights_lines);
    } else {
        return bundle_adjust<UniformWeightVector, UniformWeightVector>(points2D, points3D, lines2D, lines3D, camera,
                                                                       pose, opt, opt_line, UniformWeightVector(),
                                                                       UniformWeightVector());
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

template <typename WeightType>
BundleStats generalized_bundle_adjust(const std::vector<std::vector<Point2D>> &x,
                                      const std::vector<std::vector<Point3D>> &X,
                                      const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &cameras,
                                      CameraPose *pose, const BundleOptions &opt, const WeightType &weights) {
    IterationCallback callback = setup_callback(opt);
    GeneralizedAbsolutePoseRefiner<WeightType> refiner(x, X, camera_ext, cameras, weights);
    return lm_impl<decltype(refiner)>(refiner, pose, opt, callback);
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
// Relative pose (essential matrix) refinement. Identity intrinsics assumed

template <typename WeightType>
BundleStats refine_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, CameraPose *pose,
                           const BundleOptions &opt, const WeightType &weights) {
    IterationCallback callback = setup_callback(opt);
    PinholeRelativePoseRefiner<decltype(weights)> refiner(x1, x2, weights);
    return lm_impl<decltype(refiner)>(refiner, pose, opt, callback);
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
// Relative pose (essential matrix) refinement. Allows for general camera models


template <typename WeightType>
BundleStats refine_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, ImagePair *pair,
                           const BundleOptions &opt, const WeightType &weights) {
    IterationCallback callback = setup_callback(opt);

    bool fixed_cameras = !opt.refine_focal_length && !opt.refine_principal_point && !opt.refine_extra_params;

    if(fixed_cameras) {
        // TODO allow passing this for faster computation
        std::vector<Point3D> d1, d2;
        std::vector<Eigen::Matrix<double,3,2>> J1inv, J2inv;
        pair->camera1.unproject_with_jac(x1, &d1, &J1inv);
        pair->camera2.unproject_with_jac(x2, &d2, &J2inv);
        FixCameraRelativePoseRefiner<decltype(weights)> refiner(d1, d2, J1inv, J2inv, weights);
        return lm_impl<decltype(refiner)>(refiner, &(pair->pose), opt, callback);
    } else {
        throw std::runtime_error("TODO Not implemented yet");
    }    
}

// Entry point for essential matrix refinement
BundleStats refine_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, ImagePair *pair,
                           const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        return refine_relpose<std::vector<double>>(x1, x2, pair, opt, weights);
    } else {
        return refine_relpose<UniformWeightVector>(x1, x2, pair, opt, UniformWeightVector());
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////////
// Relative pose with unknown shared focal refinement

template <typename WeightType>
BundleStats refine_shared_focal_relpose(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                        ImagePair *image_pair, const BundleOptions &opt, const WeightType &weights) {
    IterationCallback callback = setup_callback(opt);
    SharedFocalRelativePoseRefiner<decltype(weights)> refiner(x1, x2, weights);
    return lm_impl<decltype(refiner)>(refiner, image_pair, opt, callback);
}

// Entry point for relative pose with unknown shared focal refinement
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

template <typename WeightType>
BundleStats refine_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *F,
                               const BundleOptions &opt, const WeightType &weights) {
    // We optimize over the SVD-based factorization from Bartoli and Sturm
    FactorizedFundamentalMatrix factorized_fund_mat(*F);
    IterationCallback callback = setup_callback(opt);
    PinholeFundamentalRefiner<WeightType> refiner(x1, x2, weights);
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, &factorized_fund_mat, opt, callback);
    *F = factorized_fund_mat.F();
    return stats;
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
// Uncalibrated relative pose (fundamental matrix) with radial distortion refinement

template <typename WeightType>
BundleStats refine_rd_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                  ProjectiveImagePair *proj_image_pair, const BundleOptions &opt,
                                  const WeightType &weights) {
    // We optimize over the SVD-based factorization from Bartoli and Sturm
    IterationCallback callback = setup_callback(opt);
    RDFundamentalRefiner<WeightType> refiner(x1, x2, weights);
    FactorizedProjectiveImagePair factorized_proj_image_pair(proj_image_pair->F, proj_image_pair->camera1,
                                                             proj_image_pair->camera2);
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, &factorized_proj_image_pair, opt, callback);
    *proj_image_pair = factorized_proj_image_pair.get_nonfactorized();
    return stats;
}

// Entry point for fundamental matrix with radial distortion refinement
BundleStats refine_rd_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                  ProjectiveImagePair *projective_image_pair, const BundleOptions &opt,
                                  const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        return refine_rd_fundamental<std::vector<double>>(x1, x2, projective_image_pair, opt, weights);
    } else {
        return refine_rd_fundamental<UniformWeightVector>(x1, x2, projective_image_pair, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Uncalibrated relative pose (fundamental matrix) with radial distortion refinement

template <typename WeightType>
BundleStats refine_shared_rd_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                         ProjectiveImagePair *proj_image_pair, const BundleOptions &opt,
                                         const WeightType &weights) {
    // We optimize over the SVD-based factorization from Bartoli and Sturm
    IterationCallback callback = setup_callback(opt);
    SharedRDFundamentalRefiner<WeightType> refiner(x1, x2, weights);
    FactorizedProjectiveImagePair factorized_proj_image_pair(proj_image_pair->F, proj_image_pair->camera1,
                                                             proj_image_pair->camera2);
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, &factorized_proj_image_pair, opt, callback);
    *proj_image_pair = factorized_proj_image_pair.get_nonfactorized();
    return stats;
}

// Entry point for fundamental matrix with radial distortion refinement
BundleStats refine_shared_rd_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                         ProjectiveImagePair *projective_image_pair, const BundleOptions &opt,
                                         const std::vector<double> &weights) {
    if (weights.size() == x1.size()) {
        return refine_rd_fundamental<std::vector<double>>(x1, x2, projective_image_pair, opt, weights);
    } else {
        return refine_rd_fundamental<UniformWeightVector>(x1, x2, projective_image_pair, opt, UniformWeightVector());
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Homography matrix refinement

template <typename WeightType>
BundleStats refine_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2, Eigen::Matrix3d *H,
                              const BundleOptions &opt, const WeightType &weights) {
    IterationCallback callback = setup_callback(opt);
    PinholeHomographyRefiner<WeightType> refiner(x1, x2, weights);
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, H, opt, callback);
    return stats;
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

template <typename WeightType>
BundleStats refine_generalized_relpose(const std::vector<PairwiseMatches> &matches,
                                       const std::vector<CameraPose> &camera1_ext,
                                       const std::vector<CameraPose> &camera2_ext, CameraPose *pose,
                                       const BundleOptions &opt, const WeightType &weights) {
    IterationCallback callback = setup_callback(opt);
    GeneralizedPinholeRelativePoseRefiner<WeightType> refiner(matches, camera1_ext, camera2_ext);
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, pose, opt, callback);
    return stats;
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

template <typename AbsWeightType, typename RelWeightType>
BundleStats refine_hybrid_pose(const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                               const std::vector<PairwiseMatches> &matches_2D_2D,
                               const std::vector<CameraPose> &map_ext, CameraPose *pose, const BundleOptions &opt,
                               double loss_scale_epipolar, const AbsWeightType &weights_abs,
                               const RelWeightType &weights_rel) {
    /*
    LossFunction loss_fn(opt.loss_scale);
    IterationCallback callback = setup_callback(opt, loss_fn);
    Camera camera;
    NormalAccumulator<LossFunction> acc(6, loss_fn);
    AbsolutePoseRefiner<decltype(acc), AbsWeightType> pts_refiner(x, X, camera, weights_abs);
    std::vector<CameraPose> camera2_ext = {CameraPose()};
    GeneralizedPinholeRelativePoseRefiner<decltype(acc), RelWeightType> rel_refiner(matches_2D_2D, map_ext,
                                                                                    camera2_ext);

    HybridRefiner<decltype(acc)> refiner;
    refiner.register_refiner(&pts_refiner);
    refiner.register_refiner(&rel_refiner);
    BundleStats stats = lm_impl<decltype(refiner), decltype(acc)>(refiner, acc, pose, opt, callback);
    return stats;
    */
    return BundleStats();
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

template <typename WeightType>
BundleStats bundle_adjust_1D_radial(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                                    const Camera &cam, const BundleOptions &opt, const WeightType &weights) {
    IterationCallback callback = setup_callback(opt);
    Radial1DAbsolutePoseRefiner<WeightType> refiner(x, X, cam, weights);
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, pose, opt, callback);
    return stats;
}

// Entry point for 1D radial absolute pose refinement (Assumes that the image points are centered)
BundleStats bundle_adjust_1D_radial(const std::vector<Point2D> &x, const std::vector<Point3D> &X, CameraPose *pose,
                                    const Camera &cam, const BundleOptions &opt, const std::vector<double> &weights) {
    if (weights.size() == x.size()) {
        return bundle_adjust_1D_radial<std::vector<double>>(x, X, pose, cam, opt, weights);
    } else {
        return bundle_adjust_1D_radial<UniformWeightVector>(x, X, pose, cam, opt, UniformWeightVector());
    }
}

#undef SWITCH_LOSS_FUNCTIONS

} // namespace poselib