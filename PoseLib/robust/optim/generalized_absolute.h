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

#ifndef POSELIB_GEN_ABSOLUTE_H_
#define POSELIB_GEN_ABSOLUTE_H_

#include "../../types.h"
#include "absolute.h"
#include "jacobian_accumulator.h"
#include "optim_utils.h"

namespace poselib {

template <typename ResidualWeightVectors = UniformWeightVectors, typename Accumulator = NormalAccumulator>
class GeneralizedAbsolutePoseRefiner : public RefinerBase<CameraPose, Accumulator> {
  public:
    GeneralizedAbsolutePoseRefiner(const std::vector<std::vector<Point2D>> &points2D,
                                   const std::vector<std::vector<Point3D>> &points3D,
                                   const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &camera_int,
                                   const ResidualWeightVectors &w = ResidualWeightVectors())
        : num_cams(points2D.size()), x(points2D), X(points3D), rig_poses(camera_ext), cameras(camera_int), weights(w) {
        this->num_params = 6;
    }

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        for (int k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;
            for (int i = 0; i < x[k].size(); ++i) {
                const Eigen::Vector3d Z = full_pose.apply(X[k][i]);
                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Z(2) < 0)
                    continue;
                Eigen::Vector2d xp;
                camera.project(Z, &xp);
                const Eigen::Vector2d res = xp - x[k][i];
                acc.add_residual(res, weights[k][i]);
            }
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {
        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            Image full_pose;
            full_pose.pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;
            full_pose.camera = cameras[k];
            AbsolutePoseRefiner<typename ResidualWeightVectors::value_type, Accumulator> cam_refiner(x[k], X[k], {},
                                                                                                     weights[k]);
            // Rk * (R*X + t) + tk
            // Rk * (R * (X + t)) + tk
            cam_refiner.compute_jacobian(acc, full_pose);
        }
    }

    CameraPose step(const Eigen::VectorXd &dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;

  private:
    const size_t num_cams;
    const std::vector<std::vector<Point2D>> &x;
    const std::vector<std::vector<Point3D>> &X;
    const std::vector<CameraPose> &rig_poses;
    const std::vector<Camera> &cameras;
    const ResidualWeightVectors &weights;
};

// Accumulator which extends the six parameter pose jacobians of the per-camera refiners with
// the column for the scale of the rig centers before forwarding them. The derivative of the
// residual w.r.t. the scale is the derivative w.r.t. a translation along the center offset of
// the camera, i.e. the translation block of the pose jacobian applied to that offset.
template <typename Accumulator> class ScaleColumnAccumulator {
  public:
    ScaleColumnAccumulator() {}

    // The offset is the center offset of the camera expressed in the frame used by the
    // translation block of the pose jacobian (the body frame of the rig pose)
    void set_target(Accumulator *accumulator, const Eigen::Vector3d &offset) {
        acc = accumulator;
        dt = offset;
    }

    void add_residual(const double res, const double w = 1.0) { acc->add_residual(res, w); }
    template <typename Derived> void add_residual(const Eigen::MatrixBase<Derived> &res, const double w = 1.0) {
        acc->add_residual(res.derived(), w);
    }
    double get_residual() const { return acc->get_residual(); }

    template <typename ResDerived, typename JacDerived>
    void add_jacobian(const Eigen::MatrixBase<ResDerived> &res, const Eigen::MatrixBase<JacDerived> &jac,
                      const double w = 1.0) {
        constexpr int ResidualDim = ResDerived::RowsAtCompileTime;
        static_assert(ResidualDim != Eigen::Dynamic, "The residual dimension must be known at compile time.");
        constexpr int PoseDim = JacDerived::ColsAtCompileTime;
        constexpr int ParamsDim = (PoseDim == Eigen::Dynamic) ? Eigen::Dynamic : PoseDim + 1;

        Eigen::Matrix<double, ResidualDim, ParamsDim> J(res.rows(), jac.cols() + 1);
        J.leftCols(jac.cols()) = jac;
        // The translation block of the pose jacobian applied to the center offset
        J.col(jac.cols()) = jac.template block<ResidualDim, 3>(0, 3) * dt;
        acc->add_jacobian(res.derived(), J, w);
    }

  private:
    Accumulator *acc = nullptr;
    Eigen::Vector3d dt = Eigen::Vector3d::Zero();
};

// Generalized absolute pose refinement which also optimizes the scale of the rig centers
// w.r.t. the 3D points (seven parameters: rotation, translation and scale). The rigid
// refinement above is the special case where the scale is known to be one.
template <typename ResidualWeightVectors = UniformWeightVectors, typename Accumulator = NormalAccumulator>
class GeneralizedAbsolutePoseScaleRefiner : public RefinerBase<ScaledCameraPose, Accumulator> {
  public:
    GeneralizedAbsolutePoseScaleRefiner(const std::vector<std::vector<Point2D>> &points2D,
                                        const std::vector<std::vector<Point3D>> &points3D,
                                        const std::vector<CameraPose> &camera_ext,
                                        const std::vector<Camera> &camera_int,
                                        const ResidualWeightVectors &w = ResidualWeightVectors())
        : num_cams(points2D.size()), x(points2D), X(points3D), rig_poses(camera_ext), cameras(camera_int), weights(w) {
        this->num_params = 7;
    }

    double compute_residual(Accumulator &acc, const ScaledCameraPose &scaled_pose) {
        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            Image full_pose(scaled_pose.camera_pose(rig_poses[k]), cameras[k]);
            AbsolutePoseRefiner<typename ResidualWeightVectors::value_type, Accumulator> cam_refiner(x[k], X[k], {},
                                                                                                     weights[k]);
            cam_refiner.compute_residual(acc, full_pose);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ScaledCameraPose &scaled_pose) {
        ScaleColumnAccumulator<Accumulator> scale_acc;
        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            Image full_pose(scaled_pose.camera_pose(rig_poses[k]), cameras[k]);
            // Z = Rk * (R*X + t) + scale * tk, so d/d(scale) is the derivative w.r.t. a
            // translation of the rig pose along R'*Rk'*tk = -R'*center
            scale_acc.set_target(&acc, -scaled_pose.pose.derotate(rig_poses[k].center()));
            AbsolutePoseRefiner<typename ResidualWeightVectors::value_type, ScaleColumnAccumulator<Accumulator>>
                cam_refiner(x[k], X[k], {}, weights[k]);
            cam_refiner.compute_jacobian(scale_acc, full_pose);
        }
    }

    ScaledCameraPose step(const Eigen::VectorXd &dp, const ScaledCameraPose &scaled_pose) const {
        ScaledCameraPose pose_new;
        pose_new.pose.q = quat_step_post(scaled_pose.pose.q, dp.block<3, 1>(0, 0));
        pose_new.pose.t = scaled_pose.pose.t + scaled_pose.pose.rotate(dp.block<3, 1>(3, 0));
        pose_new.scale = scaled_pose.scale + dp(6);
        return pose_new;
    }
    typedef ScaledCameraPose param_t;

  private:
    const size_t num_cams;
    const std::vector<std::vector<Point2D>> &x;
    const std::vector<std::vector<Point3D>> &X;
    const std::vector<CameraPose> &rig_poses;
    const std::vector<Camera> &cameras;
    const ResidualWeightVectors &weights;
};

// Generalized absolute pose and scale refinement for any central camera model using 3D unit
// bearing vectors instead of 2D normalized pixels. Minimizes the chord distance on the unit
// sphere (see BearingAbsolutePoseRefiner) over the seven parameters of the scaled rig pose.
template <typename ResidualWeightVectors = UniformWeightVectors, typename Accumulator = NormalAccumulator>
class BearingGeneralizedAbsolutePoseScaleRefiner : public RefinerBase<ScaledCameraPose, Accumulator> {
  public:
    BearingGeneralizedAbsolutePoseScaleRefiner(const std::vector<std::vector<Point3D>> &bearings,
                                               const std::vector<std::vector<Point3D>> &points3D,
                                               const std::vector<CameraPose> &camera_ext,
                                               const ResidualWeightVectors &w = ResidualWeightVectors())
        : num_cams(bearings.size()), b(bearings), X(points3D), rig_poses(camera_ext), weights(w) {
        this->num_params = 7;
    }

    double compute_residual(Accumulator &acc, const ScaledCameraPose &scaled_pose) {
        for (size_t k = 0; k < num_cams; ++k) {
            if (b[k].size() == 0) {
                continue;
            }
            BearingAbsolutePoseRefiner<typename ResidualWeightVectors::value_type, Accumulator> cam_refiner(b[k], X[k],
                                                                                                            weights[k]);
            cam_refiner.compute_residual(acc, scaled_pose.camera_pose(rig_poses[k]));
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ScaledCameraPose &scaled_pose) {
        ScaleColumnAccumulator<Accumulator> scale_acc;
        for (size_t k = 0; k < num_cams; ++k) {
            if (b[k].size() == 0) {
                continue;
            }
            // Same scale column as in GeneralizedAbsolutePoseScaleRefiner above
            scale_acc.set_target(&acc, -scaled_pose.pose.derotate(rig_poses[k].center()));
            BearingAbsolutePoseRefiner<typename ResidualWeightVectors::value_type, ScaleColumnAccumulator<Accumulator>>
                cam_refiner(b[k], X[k], weights[k]);
            cam_refiner.compute_jacobian(scale_acc, scaled_pose.camera_pose(rig_poses[k]));
        }
    }

    ScaledCameraPose step(const Eigen::VectorXd &dp, const ScaledCameraPose &scaled_pose) const {
        ScaledCameraPose pose_new;
        pose_new.pose.q = quat_step_post(scaled_pose.pose.q, dp.block<3, 1>(0, 0));
        pose_new.pose.t = scaled_pose.pose.t + scaled_pose.pose.rotate(dp.block<3, 1>(3, 0));
        pose_new.scale = scaled_pose.scale + dp(6);
        return pose_new;
    }
    typedef ScaledCameraPose param_t;

  private:
    const size_t num_cams;
    const std::vector<std::vector<Point3D>> &b;
    const std::vector<std::vector<Point3D>> &X;
    const std::vector<CameraPose> &rig_poses;
    const ResidualWeightVectors &weights;
};
} // namespace poselib

#endif