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

#ifndef POSELIB_ABSOLUTE_H_
#define POSELIB_ABSOLUTE_H_

#include "../../camera_pose.h"
#include "../../types.h"
#include "optim_utils.h"
#include "refiner_base.h"

namespace poselib {

template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class AbsolutePoseRefiner : public RefinerBase<Image, Accumulator> {
  public:
    AbsolutePoseRefiner(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                        const std::vector<size_t> &cam_ref_idx = {},
                        const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), camera_refine_idx(cam_ref_idx), weights(w) {
        this->num_params = 6 + cam_ref_idx.size();
    }

    template <typename CameraModel> double compute_residual_impl(Accumulator &acc, const Image &image) {
        const CameraPose &pose = image.pose;
        const Camera &camera = image.camera;
        const Eigen::Matrix3d R = pose.R();
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = R * X[i] + pose.t;
            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;
            Eigen::Vector2d xp;
            CameraModel::project(camera.params, Z, &xp);
            const Eigen::Vector2d res = xp - x[i];
            acc.add_residual(res, weights[i]);
        }
        return acc.get_residual();
    }

    double compute_residual(Accumulator &acc, const Image &image) {
        switch (image.camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        return compute_residual_impl<Model>(acc, image);                                                               \
    }
            SWITCH_CAMERA_MODELS
#undef SWITCH_CAMERA_MODEL_CASE
        }
        return std::numeric_limits<double>::max();
    }

    template <typename CameraModel, int JacDim> void compute_jacobian_impl(Accumulator &acc, const Image &image) {
        const CameraPose &pose = image.pose;
        const Camera &camera = image.camera;
        Eigen::Matrix3d R = pose.R();
        Eigen::Matrix<double, 2, 3> Jproj;
        Eigen::Matrix<double, 2, Eigen::Dynamic> J_params;
        Eigen::Matrix<double, 2, JacDim> J(2, this->num_params);
        Eigen::Matrix<double, 2, 1> res;

        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Xi = X[i];
            const Eigen::Vector3d Z = R * Xi + pose.t;

            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;

            // Project with intrinsics
            Eigen::Vector2d zp;
            if (this->num_params > 6) {
                CameraModel::project_with_jac(camera.params, Z, &zp, &Jproj, &J_params);
            } else {
                CameraModel::project_with_jac(camera.params, Z, &zp, &Jproj);
            }

            // Compute reprojection error
            Eigen::Vector2d res = zp - x[i];

            // Jacobian w.r.t. rotation w
            Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
            J.col(0) = -Xi(2) * dZ.col(1) + Xi(1) * dZ.col(2);
            J.col(1) = Xi(2) * dZ.col(0) - Xi(0) * dZ.col(2);
            J.col(2) = -Xi(1) * dZ.col(0) + Xi(0) * dZ.col(1);
            // Jacobian w.r.t. translation t
            J.template block<2, 3>(0, 3) = dZ;
            // Jacobian w.r.t. camera parameters
            for (size_t k = 0; k < camera_refine_idx.size(); ++k) {
                J.col(6 + k) = J_params.col(camera_refine_idx[k]);
            }
            acc.add_jacobian(res, J, weights[i]);
        }
    }

    template <typename CameraModel> void compute_jacobian_impl_dispatch(Accumulator &acc, const Image &image) {
        // We do this so that we can have stack allocated jacobians for small models
        // Initial tests suggests it's circa 20% faster
        switch (camera_refine_idx.size()) {
        case 0:
            compute_jacobian_impl<CameraModel, 6>(acc, image);
            return;
        case 1:
            compute_jacobian_impl<CameraModel, 7>(acc, image);
            return;
        case 2:
            compute_jacobian_impl<CameraModel, 8>(acc, image);
            return;
        default:
            compute_jacobian_impl<CameraModel, Eigen::Dynamic>(acc, image);
            return;
        }
    }
    void compute_jacobian(Accumulator &acc, const Image &image) {
        switch (image.camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        return compute_jacobian_impl_dispatch<Model>(acc, image);                                                      \
    }
            SWITCH_CAMERA_MODELS
#undef SWITCH_CAMERA_MODEL_CASE
        }
    }

    Image step(const Eigen::VectorXd &dp, const Image &image) const {
        Image image_new;
        image_new.camera = image.camera;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        // The pose is updated as
        //     R * dR * (X + dt) + t
        image_new.pose.q = quat_step_post(image.pose.q, dp.block<3, 1>(0, 0));
        image_new.pose.t = image.pose.t + image.pose.rotate(dp.block<3, 1>(3, 0));
        for (size_t i = 0; i < camera_refine_idx.size(); ++i) {
            image_new.camera.params[camera_refine_idx[i]] += dp(6 + i);
        }
        return image_new;
    }

    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    std::vector<size_t> camera_refine_idx = {};
    const ResidualWeightVector &weights;
    typedef Image param_t;
};

template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class PinholeLineAbsolutePoseRefiner : public RefinerBase<Image, Accumulator> {
  public:
    PinholeLineAbsolutePoseRefiner(const std::vector<Line2D> &lin2D, const std::vector<Line3D> &lin3D,
                                   const ResidualWeightVector &w = ResidualWeightVector())
        : lines2D(lin2D), lines3D(lin3D), weights(w) {
        this->num_params = 6;
    }

    double compute_residual(Accumulator &acc, const Image &image) {
        const CameraPose &pose = image.pose;
        const Camera &camera = image.camera;
        const Eigen::Matrix3d K = camera.calib_matrix();
        const Eigen::Matrix3d KinvT = K.inverse().transpose();

        Eigen::Matrix3d R = pose.R();
        for (size_t i = 0; i < lines2D.size(); ++i) {
            const Eigen::Vector3d Z1 = R * lines3D[i].X1 + pose.t;
            const Eigen::Vector3d Z2 = R * lines3D[i].X2 + pose.t;
            Eigen::Vector3d l = KinvT * Z1.cross(Z2);
            l /= l.topRows<2>().norm();

            const double r0 = l.dot(lines2D[i].x1.homogeneous());
            const double r1 = l.dot(lines2D[i].x2.homogeneous());
            acc.add_residual(Eigen::Vector2d(r0, r1), weights[i]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const Image &image) {
        const CameraPose &pose = image.pose;
        const Camera &camera = image.camera;
        const Eigen::Matrix3d K = camera.calib_matrix();
        const Eigen::Matrix3d KinvT = K.inverse().transpose();

        Eigen::Matrix3d E, R;
        R = pose.R();
        E << pose.t.cross(R.col(0)), pose.t.cross(R.col(1)), pose.t.cross(R.col(2));
        for (size_t k = 0; k < lines2D.size(); ++k) {
            const Eigen::Vector3d Z1 = R * lines3D[k].X1 + pose.t;
            const Eigen::Vector3d Z2 = R * lines3D[k].X2 + pose.t;

            const Eigen::Vector3d X12 = lines3D[k].X1.cross(lines3D[k].X2);
            const Eigen::Vector3d dX = lines3D[k].X1 - lines3D[k].X2;

            // Projected line
            const Eigen::Vector3d l = KinvT * Z1.cross(Z2);

            // Normalized line by first two coordinates
            Eigen::Vector2d alpha = l.topRows<2>();
            double beta = l(2);
            const double n_alpha = alpha.norm();
            alpha /= n_alpha;
            beta /= n_alpha;

            // Compute residual
            Eigen::Vector2d r;
            r << alpha.dot(lines2D[k].x1) + beta, alpha.dot(lines2D[k].x2) + beta;

            Eigen::Matrix<double, 3, 6> dl_drt;
            // Differentiate line with respect to rotation parameters
            dl_drt.block<1, 3>(0, 0) = E.row(0).cross(dX) - R.row(0).cross(X12);
            dl_drt.block<1, 3>(1, 0) = E.row(1).cross(dX) - R.row(1).cross(X12);
            dl_drt.block<1, 3>(2, 0) = E.row(2).cross(dX) - R.row(2).cross(X12);
            // and translation params
            dl_drt.block<1, 3>(0, 3) = R.row(0).cross(dX);
            dl_drt.block<1, 3>(1, 3) = R.row(1).cross(dX);
            dl_drt.block<1, 3>(2, 3) = R.row(2).cross(dX);
            dl_drt = KinvT * dl_drt;

            // Differentiate normalized line w.r.t. original line
            Eigen::Matrix3d dln_dl;
            dln_dl.block<2, 2>(0, 0) = (Eigen::Matrix2d::Identity() - alpha * alpha.transpose()) / n_alpha;
            dln_dl.block<1, 2>(2, 0) = -beta * alpha / n_alpha;
            dln_dl.block<2, 1>(0, 2).setZero();
            dln_dl(2, 2) = 1 / n_alpha;

            // Differentiate residual w.r.t. line
            Eigen::Matrix<double, 2, 3> dr_dl;
            dr_dl.row(0) << lines2D[k].x1.transpose(), 1.0;
            dr_dl.row(1) << lines2D[k].x2.transpose(), 1.0;

            Eigen::Matrix<double, 2, 6> J = dr_dl * dln_dl * dl_drt;
            acc.add_jacobian(r, J, weights[k]);
        }
    }

    Image step(const Eigen::VectorXd &dp, const Image &image) const {
        Image image_new;
        image_new.camera = image.camera;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        // The pose is updated as
        //     R * dR * (X + dt) + t
        image_new.pose.q = quat_step_post(image.pose.q, dp.block<3, 1>(0, 0));
        image_new.pose.t = image.pose.t + image.pose.rotate(dp.block<3, 1>(3, 0));
        return image_new;
    }

    typedef Image param_t;
    const std::vector<Line2D> &lines2D;
    const std::vector<Line3D> &lines3D;
    const ResidualWeightVector &weights;
};

// Note this optimization is not consistent with the other 6 DoF optimization
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class Radial1DAbsolutePoseRefiner : public RefinerBase<CameraPose, Accumulator> {
  public:
    Radial1DAbsolutePoseRefiner(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                const Camera &cam, const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), camera(cam), weights(w) {
        this->num_params = 5;
    }

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.apply(X[i]);
            Eigen::Vector2d xp;
            Radial1DCameraModel::project(camera.params, Z, &xp);
            Eigen::Vector2d x0 = x[i] - camera.principal_point();
            const double alpha = xp.dot(x0);
            if (alpha < 0)
                continue;
            const Eigen::Vector2d res = alpha * xp - x0;
            acc.add_residual(res, weights[i]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {
        Eigen::Matrix3d R = pose.R();
        Eigen::Matrix<double, 2, 3> Jproj;
        Eigen::Matrix<double, 2, 5> J;
        Eigen::Matrix<double, 2, 1> res;
        for (int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d RX = R * X[i];
            const Eigen::Vector3d Z = RX + pose.t;

            Eigen::Vector2d xp;
            Radial1DCameraModel::project_with_jac(camera.params, Z, &xp, &Jproj);
            Eigen::Vector2d x0 = x[i] - camera.principal_point();
            const double alpha = xp.dot(x0);
            if (alpha < 0)
                continue;
            const Eigen::Vector2d res = alpha * xp - x0;

            // differentiate residual with respect to z
            Eigen::Matrix2d dr_dz =
                (xp * x0.transpose() + alpha * Eigen::Matrix2d::Identity()) * Jproj.block<2, 2>(0, 0);
            Eigen::Matrix<double, 2, 5> dz;
            dz << 0.0, RX(2), -RX(1), 1.0, 0.0, -RX(2), 0.0, RX(0), 0.0, 1.0;
            J = dr_dz * dz;

            acc.add_jacobian(res, J, weights[i]);
        }
    }

    CameraPose step(const Eigen::VectorXd &dp, const CameraPose &pose) const {
        CameraPose pose_new;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = expm([delta]_x) * R
        // The pose is updated as
        //     dR * R * X + t + dt
        pose_new.q = quat_step_pre(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t.topRows<2>() = pose.t.topRows<2>() + dp.block<2, 1>(3, 0);
        return pose_new;
    }

    typedef CameraPose param_t;
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const Camera &camera;
    const ResidualWeightVector &weights;
};

} // namespace poselib

#endif