#ifndef POSELIB_BUNDLE_H_
#define POSELIB_BUNDLE_H_

#include "../types.h"
#include "colmap_models.h"
#include <Eigen/Dense>

namespace pose_lib {

    struct BundleOptions {
        size_t max_iterations = 100;        
        enum LossType {
            TRIVIAL,
            TRUNCATED,
            HUBER,
            CAUCHY
        } loss_type = LossType::CAUCHY;
        double loss_scale = 1.0;
        double gradient_tol = 1e-8;
        double step_tol = 1e-8;
        double initial_lambda = 1e-3;
    };

    // Minimizes reprojection error. Assumes identity intrinsics (calibrated camera)
    // Returns number of iterations.
    int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, CameraPose *pose, const BundleOptions &opt = BundleOptions());    

    // Uses intrinsic calibration from Camera (see colmap_models.h)    
    // Slightly slower than bundle_adjust above
    int bundle_adjust(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const Camera &camera, CameraPose *pose, const BundleOptions &opt = BundleOptions());    

}

#endif