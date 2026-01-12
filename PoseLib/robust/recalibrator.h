// Copyright (c) 2024, Viktor Larsson
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

#ifndef POSELIB_RECALIBRATOR_H_
#define POSELIB_RECALIBRATOR_H_
#include "../misc/camera_models.h"
#include "../types.h"
#include "optim/jacobian_accumulator.h"
#include "optim/refiner_base.h"

#include <iostream>

namespace poselib {

// Re-calibrate a camera from one model to another given a collection of 2D points.
BundleStats recalibrate(const std::vector<Point2D> &x, const Camera &source, Camera *target,
                        const BundleOptions &opt = BundleOptions(), bool rescale = true);

template <typename Accumulator = NormalAccumulator>
class RecalibratorRefiner : public RefinerBase<Camera, Accumulator> {
  public:
    RecalibratorRefiner(const std::vector<Point2D> &x, const std::vector<Point3D> &x_unproj,
                        const std::vector<size_t> &cam_ref_idx = {})
        : x(x), x_unproj(x_unproj), camera_refine_idx(cam_ref_idx) {
        this->num_params = cam_ref_idx.size();
    }

    template <typename CameraModel> double compute_residual_impl(Accumulator &acc, const Camera &camera) {
        for (int i = 0; i < x.size(); ++i) {
            Eigen::Vector2d xp;
            CameraModel::project(camera.params, x_unproj[i], &xp);
            const Eigen::Vector2d res = xp - x[i];
            acc.add_residual(res);
        }
        return acc.get_residual();
    }

    double compute_residual(Accumulator &acc, const Camera &camera) {
        switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        return compute_residual_impl<Model>(acc, camera);                                                              \
    }
            SWITCH_CAMERA_MODELS
#undef SWITCH_CAMERA_MODEL_CASE
        }
        return std::numeric_limits<double>::max();
    }

    template <typename CameraModel> void compute_jacobian_impl(Accumulator &acc, const Camera &camera) {
        Eigen::Matrix<double, 2, Eigen::Dynamic> J(2, camera_refine_idx.size());
        Eigen::Matrix<double, 2, Eigen::Dynamic> J_param(2, CameraModel::num_params);
        Eigen::Matrix<double, 2, 3> J_cam;

        for (int i = 0; i < x.size(); ++i) {
            Eigen::Vector2d xp;
            CameraModel::project_with_jac(camera.params, x_unproj[i], &xp, &J_cam, &J_param);

            const Eigen::Vector2d res = xp - x[i];
            for (size_t k = 0; k < camera_refine_idx.size(); ++k) {
                J.col(k) = J_param.col(camera_refine_idx[k]);
            }
            // std::cout << "i = " << i << ", J = \n" << J <<  "\n res = " << res.transpose() << "\n";
            // std::cout << "xp = " << xp.transpose() << ", x = " << x[i].transpose() << ", x_unproj = " <<
            // x_unproj[i].transpose() << "\n";
            acc.add_jacobian(res, J);
        }
    }

    void compute_jacobian(Accumulator &acc, const Camera &camera) {
        switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        return compute_jacobian_impl<Model>(acc, camera);                                                              \
    }
            SWITCH_CAMERA_MODELS
#undef SWITCH_CAMERA_MODEL_CASE
        }
    }

    Camera step(const Eigen::VectorXd &dp, const Camera &camera) const {
        Camera camera_new = camera;
        for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
            camera_new.params[camera_refine_idx[i]] += dp(i);
        }
        return camera_new;
    }

    const std::vector<Point2D> &x;
    const std::vector<Point3D> &x_unproj;
    std::vector<size_t> camera_refine_idx = {};
    typedef Camera param_t;
};

} // namespace poselib

#endif