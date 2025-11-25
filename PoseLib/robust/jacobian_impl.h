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

#ifndef POSELIB_JACOBIAN_IMPL_H_
#define POSELIB_JACOBIAN_IMPL_H_

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/colmap_models.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/types.h"

namespace poselib {

// For the accumulators we support supplying a vector<double> with point-wise weights for the residuals
// In case we don't want to have weighted residuals, we can pass UniformWeightVector instead of filling a std::vector
// with 1.0 The multiplication is then hopefully is optimized away since it always returns 1.0
class UniformWeightVector {
  public:
    UniformWeightVector() {}
    constexpr double operator[](std::size_t idx) const { return 1.0; }
};
class UniformWeightVectors { // this corresponds to std::vector<std::vector<double>> used for generalized cameras etc
  public:
    UniformWeightVectors() {}
    constexpr const UniformWeightVector &operator[](std::size_t idx) const { return w; }
    const UniformWeightVector w;
    typedef UniformWeightVector value_type;
};

template <typename CameraModel, typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class CameraJacobianAccumulator {
  public:
    CameraJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                              const Camera &cam, const LossFunction &loss,
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), camera(cam), loss_fn(loss), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.apply(X[i]);
            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;
            const double inv_z = 1.0 / Z(2);
            Eigen::Vector2d p(Z(0) * inv_z, Z(1) * inv_z);
            CameraModel::project(camera.params, p, &p);
            const double r0 = p(0) - x[i](0);
            const double r1 = p(1) - x[i](1);
            const double r_squared = r0 * r0 + r1 * r1;
            cost += weights[i] * loss_fn.loss(r_squared);
        }
        return cost;
    }

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        Eigen::Matrix3d R = pose.R();
        Eigen::Matrix2d Jcam;
        Jcam.setIdentity(); // we initialize to identity here (this is for the calibrated case)
        size_t num_residuals = 0;
        for (size_t i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = R * X[i] + pose.t;
            const Eigen::Vector2d z = Z.hnormalized();

            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;

            // Project with intrinsics
            Eigen::Vector2d zp = z;
            CameraModel::project_with_jac(camera.params, z, &zp, &Jcam);

            // Setup residual
            Eigen::Vector2d r = zp - x[i];
            const double r_squared = r.squaredNorm();
            const double weight = weights[i] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute jacobian w.r.t. Z (times R)
            Eigen::Matrix<double, 2, 3> dZ;
            dZ.block<2, 2>(0, 0) = Jcam;
            dZ.col(2) = -Jcam * z;
            dZ *= 1.0 / Z(2);
            dZ *= R;

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
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));

        // Translation is parameterized as (negative) shift in position
        //  i.e. t(delta) = t + R*delta
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const Camera &camera;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVectors = UniformWeightVectors>
class GeneralizedCameraJacobianAccumulator {
  public:
    GeneralizedCameraJacobianAccumulator(const std::vector<std::vector<Point2D>> &points2D,
                                         const std::vector<std::vector<Point3D>> &points3D,
                                         const std::vector<CameraPose> &camera_ext,
                                         const std::vector<Camera> &camera_int, const LossFunction &l,
                                         const ResidualWeightVectors &w = ResidualWeightVectors())
        : num_cams(points2D.size()), x(points2D), X(points3D), rig_poses(camera_ext), cameras(camera_int), loss_fn(l),
          weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;

            switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        CameraJacobianAccumulator<Model, decltype(loss_fn), typename ResidualWeightVectors::value_type> accum(         \
            x[k], X[k], cameras[k], loss_fn, weights[k]);                                                              \
        cost += accum.residual(full_pose);                                                                             \
        break;                                                                                                         \
    }
                SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
            }
        }
        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        size_t num_residuals = 0;

        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;

            switch (camera.model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id: {                                                                                            \
        CameraJacobianAccumulator<Model, decltype(loss_fn), typename ResidualWeightVectors::value_type> accum(         \
            x[k], X[k], cameras[k], loss_fn, weights[k]);                                                              \
        num_residuals += accum.accumulate(full_pose, JtJ, Jtr);                                                        \
        break;                                                                                                         \
    }
                SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const size_t num_cams;
    const std::vector<std::vector<Point2D>> &x;
    const std::vector<std::vector<Point3D>> &X;
    const std::vector<CameraPose> &rig_poses;
    const std::vector<Camera> &cameras;
    const LossFunction &loss_fn;
    const ResidualWeightVectors &weights;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector> class LineJacobianAccumulator {
  public:
    LineJacobianAccumulator(const std::vector<Line2D> &lines2D_, const std::vector<Line3D> &lines3D_,
                            const LossFunction &loss, const ResidualWeightVector &w = ResidualWeightVector())
        : lines2D(lines2D_), lines3D(lines3D_), loss_fn(loss), weights(w) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d R = pose.R();
        double cost = 0;
        for (size_t i = 0; i < lines2D.size(); ++i) {
            const Eigen::Vector3d Z1 = R * lines3D[i].X1 + pose.t;
            const Eigen::Vector3d Z2 = R * lines3D[i].X2 + pose.t;
            Eigen::Vector3d l = Z1.cross(Z2);
            l /= l.topRows<2>().norm();

            const double r0 = l.dot(lines2D[i].x1.homogeneous());
            const double r1 = l.dot(lines2D[i].x2.homogeneous());
            const double r_squared = r0 * r0 + r1 * r1;
            cost += weights[i] * loss_fn.loss(r_squared);
        }
        return cost;
    }

    // computes J.transpose() * J and J.transpose() * res
    // Only computes the lower half of JtJ
    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {

        Eigen::Matrix3d E, R;
        R = pose.R();
        E << pose.t.cross(R.col(0)), pose.t.cross(R.col(1)), pose.t.cross(R.col(2));
        size_t num_residuals = 0;
        for (size_t k = 0; k < lines2D.size(); ++k) {
            const Eigen::Vector3d Z1 = R * lines3D[k].X1 + pose.t;
            const Eigen::Vector3d Z2 = R * lines3D[k].X2 + pose.t;

            const Eigen::Vector3d X12 = lines3D[k].X1.cross(lines3D[k].X2);
            const Eigen::Vector3d dX = lines3D[k].X1 - lines3D[k].X2;

            // Projected line
            const Eigen::Vector3d l = Z1.cross(Z2);

            // Normalized line by first two coordinates
            Eigen::Vector2d alpha = l.topRows<2>();
            double beta = l(2);
            const double n_alpha = alpha.norm();
            alpha /= n_alpha;
            beta /= n_alpha;

            // Compute residual
            Eigen::Vector2d r;
            r << alpha.dot(lines2D[k].x1) + beta, alpha.dot(lines2D[k].x2) + beta;

            const double r_squared = r.squaredNorm();
            const double weight = weights[k] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            Eigen::Matrix<double, 3, 6> dl_drt;
            // Differentiate line with respect to rotation parameters
            dl_drt.block<1, 3>(0, 0) = E.row(0).cross(dX) - R.row(0).cross(X12);
            dl_drt.block<1, 3>(1, 0) = E.row(1).cross(dX) - R.row(1).cross(X12);
            dl_drt.block<1, 3>(2, 0) = E.row(2).cross(dX) - R.row(2).cross(X12);
            // and translation params
            dl_drt.block<1, 3>(0, 3) = R.row(0).cross(dX);
            dl_drt.block<1, 3>(1, 3) = R.row(1).cross(dX);
            dl_drt.block<1, 3>(2, 3) = R.row(2).cross(dX);

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
            // Accumulate into JtJ and Jtr
            Jtr += weight * J.transpose() * r;
            for (size_t i = 0; i < 6; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J.col(i).dot(J.col(j)));
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        // Translation is parameterized as (negative) shift in position
        //  i.e. t(delta) = t + R*delta
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Line2D> &lines2D;
    const std::vector<Line3D> &lines3D;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename PointLossFunction, typename LineLossFunction, typename PointResidualsVector = UniformWeightVector,
          typename LineResidualsVector = UniformWeightVector>
class PointLineJacobianAccumulator {
  public:
    PointLineJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                 const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                                 const PointLossFunction &l_point, const LineLossFunction &l_line,
                                 const PointResidualsVector &weights_pts = PointResidualsVector(),
                                 const LineResidualsVector &weights_l = LineResidualsVector())
        : pts_accum(points2D, points3D, trivial_camera, l_point, weights_pts),
          line_accum(lines2D, lines3D, l_line, weights_l) {
        trivial_camera.model_id = NullCameraModel::model_id;
    }

    double residual(const CameraPose &pose) const { return pts_accum.residual(pose) + line_accum.residual(pose); }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        return pts_accum.accumulate(pose, JtJ, Jtr) + line_accum.accumulate(pose, JtJ, Jtr);
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        // Both CameraJacobianAccumulator and LineJacobianAccumulator have the same step!
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    Camera trivial_camera;
    CameraJacobianAccumulator<NullCameraModel, PointLossFunction, PointResidualsVector> pts_accum;
    LineJacobianAccumulator<LineLossFunction, LineResidualsVector> line_accum;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class RelativePoseJacobianAccumulator {
  public:
    RelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                    const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());
            double nJc_sq = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ, Eigen::Matrix<double, 5, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(pose.t.x()) < std::abs(pose.t.y())) {
            // x < y
            if (std::abs(pose.t.x()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(pose.t.y()) < std::abs(pose.t.z())) {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(pose.t).normalized();

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

// Hybrid optimization for absolute pose with two monodepths, optimizes both symmetric reprojection error + sampson
// considers also scale and both shifts
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class MonoDepthRelPoseShiftJacobianAccumulator {
  public:
    MonoDepthRelPoseShiftJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                             const std::vector<Point2D> &points2D_2, const std::vector<double> &d1,
                                             const std::vector<double> &d2, const LossFunction &l,
                                             const double scale_reproj, const double weight_sampson,
                                             const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), d1(d1), d2(d2), loss_fn(l), scale_reproj(scale_reproj),
          weight_sampson(weight_sampson), weights(w) {}

    double residual(const MonoDepthTwoViewGeometry &geometry) const {
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        Eigen::Matrix3d R = geometry.pose.R();
        Eigen::Vector3d t = geometry.pose.t;
        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);

        double cost = 0;
        for (size_t i = 0; i < x1.size(); ++i) {
            if (weight_sampson > 0.0) {
                double C = x2[i].homogeneous().dot(E * x1[i].homogeneous());
                double nJc_sq = (E.block<2, 3>(0, 0) * x1[i].homogeneous()).squaredNorm() +
                                (E.block<3, 2>(0, 0).transpose() * x2[i].homogeneous()).squaredNorm();
                double r2 = (C * C) / nJc_sq;

                cost += weights[i] * loss_fn.loss(r2) * weight_sampson;
            }

            if (scale_reproj > 0.0) {
                const Eigen::Vector3d Z1 = R * (d1[i] + shift_1) * x1[i].homogeneous().eval() + t;
                const Eigen::Vector3d Z2 = R.transpose() * (scale * (d2[i] + shift_2) * x2[i].homogeneous().eval() - t);

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Z1(2) > 0) {
                    const double inv_z1 = 1.0 / Z1(2);
                    const double r10 = Z1(0) * inv_z1 - x2[i](0);
                    const double r11 = Z1(1) * inv_z1 - x2[i](1);
                    const double r_squared1 = r10 * r10 + r11 * r11;
                    cost += weights[i] * loss_fn.loss(scale_reproj * r_squared1);
                }

                if (Z2(2) > 0) {
                    const double inv_z2 = 1.0 / Z2(2);
                    const double r20 = Z2(0) * inv_z2 - x1[i](0);
                    const double r21 = Z2(1) * inv_z2 - x1[i](1);
                    const double r_squared2 = r20 * r20 + r21 * r21;
                    cost += weights[i] * loss_fn.loss(scale_reproj * r_squared2);
                }
            }
        }
        return cost;
    }

    size_t accumulate(const MonoDepthTwoViewGeometry &geometry, Eigen::Matrix<double, 9, 9> &JtJ,
                      Eigen::Matrix<double, 9, 1> &Jtr) const {
        Eigen::Matrix3d R = geometry.pose.R();
        Eigen::Matrix3d Rt = R.transpose();
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;

        Eigen::Matrix<double, 2, 9> J;
        Eigen::Matrix<double, 2, 3> Jproj;
        Jproj.setZero();

        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 3> dt;
        Eigen::Matrix<double, 1, 9> J_sam;
        J_sam.setZero();

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        Eigen::Matrix3d dt_0, dt_1, dt_2;
        dt_0.row(0).setZero();
        dt_0.row(1) = -R.row(2);
        dt_0.row(2) = R.row(1);

        dt_1.row(0) = R.row(2);
        dt_1.row(1).setZero();
        dt_1.row(2) = -R.row(0);

        dt_2.row(0) = -R.row(1);
        dt_2.row(1) = R.row(0);
        dt_2.row(2).setZero();

        dt.col(0) = Eigen::Map<Eigen::VectorXd>(dt_0.data(), dt_0.size());
        dt.col(1) = Eigen::Map<Eigen::VectorXd>(dt_1.data(), dt_1.size());
        dt.col(2) = Eigen::Map<Eigen::VectorXd>(dt_2.data(), dt_2.size());

        //        Eigen::Matrix<double, 9, 1> Jtr_check;

        size_t num_residuals = 0;
        for (size_t i = 0; i < x1.size(); ++i) {
            //            Jtr_check.setZero();
            const Eigen::Vector3d X1o = x1[i].homogeneous().eval();
            const Eigen::Vector3d X1i = (d1[i] + shift_1) * X1o;
            const Eigen::Vector3d Z1 = R * X1i + geometry.pose.t;

            const Eigen::Vector3d X2o = x2[i].homogeneous().eval();
            const Eigen::Vector3d X2s = (d2[i] + shift_2) * X2o;
            const Eigen::Vector3d X2i = scale * X2s;
            const Eigen::Vector3d Z2 = Rt * (X2i - geometry.pose.t);

            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization

            if (Z1(2) > 0) {
                Eigen::Vector2d zp;
                const double inv_z = 1.0 / Z1(2);
                zp[0] = Z1(0) * inv_z;
                zp[1] = Z1(1) * inv_z;

                Jproj(0, 0) = inv_z;
                Jproj(1, 1) = inv_z;
                Jproj(0, 2) = -Z1(0) * inv_z * inv_z;
                Jproj(1, 2) = -Z1(1) * inv_z * inv_z;

                // Compute reprojection error
                Eigen::Vector2d res = zp - x2[i];

                const double r_squared = scale_reproj * res.squaredNorm();
                const double weight = weights[i] * loss_fn.weight(r_squared);

                if (weight > 0.0) {
                    num_residuals++;

                    // Jacobian w.r.t. rotation w
                    Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
                    J.col(0) = -X1i(2) * dZ.col(1) + X1i(1) * dZ.col(2);
                    J.col(1) = X1i(2) * dZ.col(0) - X1i(0) * dZ.col(2);
                    J.col(2) = -X1i(1) * dZ.col(0) + X1i(0) * dZ.col(1);
                    // Jacobian w.r.t. translation t
                    J.block<2, 3>(0, 3) = Jproj;
                    // Jacobian w.r.t. shift1
                    J(0, 7) = inv_z * R.row(0).dot(X1o) - Z1(0) * inv_z * inv_z * R.row(2).dot(X1o);
                    J(1, 7) = inv_z * R.row(1).dot(X1o) - Z1(1) * inv_z * inv_z * R.row(2).dot(X1o);

                    J.col(6).setZero();
                    J.col(8).setZero();

                    for (int k = 0; k < 9; ++k) {
                        for (int j = 0; j <= k; ++j) {
                            JtJ(k, j) += scale_reproj * weight * (J.col(k).dot(J.col(j)));
                        }
                    }
                    Jtr += J.transpose() * (scale_reproj * weight * res);
                }
            }

            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z2(2) > 0) {
                Eigen::Vector2d zp;
                const double inv_z = 1.0 / Z2(2);
                zp[0] = Z2(0) * inv_z;
                zp[1] = Z2(1) * inv_z;

                Jproj(0, 0) = inv_z;
                Jproj(1, 1) = inv_z;
                Jproj(0, 2) = -Z2(0) * inv_z * inv_z;
                Jproj(1, 2) = -Z2(1) * inv_z * inv_z;

                // Compute reprojection error
                Eigen::Vector2d res = zp - x1[i];

                const double r_squared = scale_reproj * res.squaredNorm();
                const double weight = weights[i] * loss_fn.weight(r_squared);

                if (weight > 0.0) {
                    num_residuals++;

                    Eigen::Vector3d X2t = X2i - geometry.pose.t; // - geometry.t;

                    Eigen::Matrix3d dZdr;
                    dZdr.diagonal().setZero();
                    dZdr(1, 0) = X2t.dot(R.col(2));
                    dZdr(2, 0) = -X2t.dot(R.col(1));
                    dZdr(0, 1) = -X2t.dot(R.col(2));
                    dZdr(2, 1) = X2t.dot(R.col(0));
                    dZdr(0, 2) = X2t.dot(R.col(1));
                    dZdr(1, 2) = -X2t.dot(R.col(0));
                    J.block<2, 3>(0, 0) = Jproj * dZdr;

                    // Jacobian w.r.t. translation t
                    J.block<2, 3>(0, 3) = -Jproj * Rt;
                    // Jacobian w.r.t. camera parameters
                    J.col(6) = Jproj * Rt * X2s;
                    J.col(8) = scale * Jproj * Rt * X2o;
                    J.col(7).setZero();

                    for (int k = 0; k < 9; ++k) {
                        for (int j = 0; j <= k; ++j) {
                            JtJ(k, j) += scale_reproj * weight * (J.col(k).dot(J.col(j)));
                        }
                    }
                    Jtr += J.transpose() * (scale_reproj * weight * res);
                }
            }

            double C = x2[i].homogeneous().dot(E * x1[i].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * x2[i].homogeneous(), E.block<2, 3>(0, 0) * x1[i].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[i] * loss_fn.weight(r * r) * weight_sampson;
            if (weight > 0) {
                num_residuals++;

                // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
                Eigen::Matrix<double, 1, 9> dF;
                dF << x1[i](0) * x2[i](0), x1[i](0) * x2[i](1), x1[i](0), x1[i](1) * x2[i](0), x1[i](1) * x2[i](1),
                    x1[i](1), x2[i](0), x2[i](1), 1.0;
                const double s = C * inv_nJ_C * inv_nJ_C;
                dF(0) -= s * (J_C(2) * x1[i](0) + J_C(0) * x2[i](0));
                dF(1) -= s * (J_C(3) * x1[i](0) + J_C(0) * x2[i](1));
                dF(2) -= s * (J_C(0));
                dF(3) -= s * (J_C(2) * x1[i](1) + J_C(1) * x2[i](0));
                dF(4) -= s * (J_C(3) * x1[i](1) + J_C(1) * x2[i](1));
                dF(5) -= s * (J_C(1));
                dF(6) -= s * (J_C(2));
                dF(7) -= s * (J_C(3));
                dF *= inv_nJ_C;

                J_sam.block<1, 3>(0, 0) = dF * dR;
                J_sam.block<1, 3>(0, 3) = dF * dt;

                for (int k = 0; k < 9; ++k) {
                    for (int j = 0; j <= k; ++j) {
                        JtJ(k, j) += weight_sampson * weight * (J_sam(k) * J_sam(j));
                    }
                }
                Jtr += weight_sampson * weight * C * inv_nJ_C * J_sam.transpose();
            }
        }

        return num_residuals;
    }

    MonoDepthTwoViewGeometry step(Eigen::Matrix<double, 9, 1> dp, const MonoDepthTwoViewGeometry &geometry) const {
        MonoDepthTwoViewGeometry geometry_new;
        geometry_new.pose.q = quat_step_post(geometry.pose.q, dp.block<3, 1>(0, 0));
        //        geometry_new.t = geometry.t + geometry.rotate(dp.block<3, 1>(3, 0));
        geometry_new.pose.t = geometry.pose.t + dp.block<3, 1>(3, 0);
        geometry_new.scale = geometry.scale + dp(6, 0);
        geometry_new.shift1 = geometry.shift1 + dp(7, 0);
        geometry_new.shift2 = geometry.shift2 + dp(8, 0);
        return geometry_new;
    }
    typedef MonoDepthTwoViewGeometry param_t;
    static constexpr size_t num_params = 9;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;
    const LossFunction &loss_fn;
    const double scale_reproj, weight_sampson;
    const ResidualWeightVector &weights;
};

// Hybrid optimization for absolute pose with two monodepths, optimizes both symmetric reprojection error + sampson
// optimizes scale but not shift
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class MonoDepthRelPoseJacobianAccumulator {
  public:
    MonoDepthRelPoseJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                        const std::vector<double> &d1, const std::vector<double> &d2,
                                        const LossFunction &l, const double scale_reproj, const double weight_sampson,
                                        const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), d1(d1), d2(d2), loss_fn(l), scale_reproj(scale_reproj),
          weight_sampson(weight_sampson), weights(w) {}

    double residual(const MonoDepthTwoViewGeometry &geometry) const {
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        Eigen::Matrix3d R = geometry.pose.R();
        Eigen::Vector3d t = geometry.pose.t;
        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);

        double cost = 0;
        for (size_t i = 0; i < x1.size(); ++i) {
            if (weight_sampson > 0.0) {
                double C = x2[i].homogeneous().dot(E * x1[i].homogeneous());
                double nJc_sq = (E.block<2, 3>(0, 0) * x1[i].homogeneous()).squaredNorm() +
                                (E.block<3, 2>(0, 0).transpose() * x2[i].homogeneous()).squaredNorm();
                double r2 = (C * C) / nJc_sq;

                cost += weights[i] * loss_fn.loss(r2) * weight_sampson;
            }

            if (scale_reproj > 0.0) {
                const Eigen::Vector3d Z1 = R * (d1[i] + shift_1) * x1[i].homogeneous().eval() + t;
                const Eigen::Vector3d Z2 = R.transpose() * (scale * (d2[i] + shift_2) * x2[i].homogeneous().eval() - t);

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Z1(2) > 0) {
                    const double inv_z1 = 1.0 / Z1(2);
                    const double r10 = Z1(0) * inv_z1 - x2[i](0);
                    const double r11 = Z1(1) * inv_z1 - x2[i](1);
                    const double r_squared1 = r10 * r10 + r11 * r11;
                    cost += weights[i] * loss_fn.loss(scale_reproj * r_squared1);
                }

                if (Z2(2) > 0) {
                    const double inv_z2 = 1.0 / Z2(2);
                    const double r20 = Z2(0) * inv_z2 - x1[i](0);
                    const double r21 = Z2(1) * inv_z2 - x1[i](1);
                    const double r_squared2 = r20 * r20 + r21 * r21;
                    cost += weights[i] * loss_fn.loss(scale_reproj * r_squared2);
                }
            }
        }
        return cost;
    }

    size_t accumulate(const MonoDepthTwoViewGeometry &geometry, Eigen::Matrix<double, 7, 7> &JtJ,
                      Eigen::Matrix<double, 7, 1> &Jtr) const {
        Eigen::Matrix3d R = geometry.pose.R();
        Eigen::Matrix3d Rt = R.transpose();
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        // const double scale = 1.0;

        Eigen::Matrix<double, 2, 7> J;
        Eigen::Matrix<double, 2, 3> Jproj;
        Jproj.setZero();

        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 3> dt;
        Eigen::Matrix<double, 1, 7> J_sam;
        J_sam.setZero();

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        Eigen::Matrix3d dt_0, dt_1, dt_2;
        dt_0.row(0).setZero();
        dt_0.row(1) = -R.row(2);
        dt_0.row(2) = R.row(1);

        dt_1.row(0) = R.row(2);
        dt_1.row(1).setZero();
        dt_1.row(2) = -R.row(0);

        dt_2.row(0) = -R.row(1);
        dt_2.row(1) = R.row(0);
        dt_2.row(2).setZero();

        dt.col(0) = Eigen::Map<Eigen::VectorXd>(dt_0.data(), dt_0.size());
        dt.col(1) = Eigen::Map<Eigen::VectorXd>(dt_1.data(), dt_1.size());
        dt.col(2) = Eigen::Map<Eigen::VectorXd>(dt_2.data(), dt_2.size());

        size_t num_residuals = 0;
        for (size_t i = 0; i < x1.size(); ++i) {
            if (scale_reproj > 0.0) {
                const Eigen::Vector3d X1o = x1[i].homogeneous().eval();
                const Eigen::Vector3d X1i = (d1[i] + shift_1) * X1o;
                const Eigen::Vector3d Z1 = R * X1i + geometry.pose.t;

                const Eigen::Vector3d X2o = x2[i].homogeneous().eval();
                const Eigen::Vector3d X2s = (d2[i] + shift_2) * X2o;
                const Eigen::Vector3d X2i = scale * X2s;
                const Eigen::Vector3d Z2 = Rt * (X2i - geometry.pose.t);

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization

                if (Z1(2) > 0) {
                    Eigen::Vector2d zp;
                    const double inv_z = 1.0 / Z1(2);
                    zp[0] = Z1(0) * inv_z;
                    zp[1] = Z1(1) * inv_z;

                    Jproj(0, 0) = inv_z;
                    Jproj(1, 1) = inv_z;
                    Jproj(0, 2) = -Z1(0) * inv_z * inv_z;
                    Jproj(1, 2) = -Z1(1) * inv_z * inv_z;

                    // Compute reprojection error
                    Eigen::Vector2d res = zp - x2[i];

                    const double r_squared = scale_reproj * res.squaredNorm();
                    const double weight = weights[i] * loss_fn.weight(r_squared);

                    if (weight > 0.0) {
                        num_residuals++;

                        // Jacobian w.r.t. rotation w
                        Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
                        J.col(0) = -X1i(2) * dZ.col(1) + X1i(1) * dZ.col(2);
                        J.col(1) = X1i(2) * dZ.col(0) - X1i(0) * dZ.col(2);
                        J.col(2) = -X1i(1) * dZ.col(0) + X1i(0) * dZ.col(1);
                        // Jacobian w.r.t. translation t
                        J.block<2, 3>(0, 3) = Jproj;
                        // Jacobian w.r.t. shift1
                        J.col(6).setZero();

                        for (int k = 0; k < 7; ++k) {
                            for (int j = 0; j <= k; ++j) {
                                JtJ(k, j) += scale_reproj * weight * (J.col(k).dot(J.col(j)));
                            }
                        }
                        Jtr += J.transpose() * (scale_reproj * weight * res);
                    }
                }

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Z2(2) > 0) {
                    Eigen::Vector2d zp;
                    const double inv_z = 1.0 / Z2(2);
                    zp[0] = Z2(0) * inv_z;
                    zp[1] = Z2(1) * inv_z;

                    Jproj(0, 0) = inv_z;
                    Jproj(1, 1) = inv_z;
                    Jproj(0, 2) = -Z2(0) * inv_z * inv_z;
                    Jproj(1, 2) = -Z2(1) * inv_z * inv_z;

                    // Compute reprojection error
                    Eigen::Vector2d res = zp - x1[i];

                    const double r_squared = scale_reproj * res.squaredNorm();
                    const double weight = weights[i] * loss_fn.weight(r_squared);

                    if (weight > 0.0) {
                        num_residuals++;

                        Eigen::Vector3d X2t = X2i - geometry.pose.t; // - geometry.t;

                        Eigen::Matrix3d dZdr;
                        dZdr.diagonal().setZero();
                        dZdr(1, 0) = X2t.dot(R.col(2));
                        dZdr(2, 0) = -X2t.dot(R.col(1));
                        dZdr(0, 1) = -X2t.dot(R.col(2));
                        dZdr(2, 1) = X2t.dot(R.col(0));
                        dZdr(0, 2) = X2t.dot(R.col(1));
                        dZdr(1, 2) = -X2t.dot(R.col(0));
                        J.block<2, 3>(0, 0) = Jproj * dZdr;

                        // Jacobian w.r.t. translation t
                        J.block<2, 3>(0, 3) = -Jproj * Rt;
                        // Jacobian w.r.t. camera parameters
                        J.col(6) = Jproj * Rt * X2s;

                        for (int k = 0; k < 7; ++k) {
                            for (int j = 0; j <= k; ++j) {
                                JtJ(k, j) += scale_reproj * weight * (J.col(k).dot(J.col(j)));
                            }
                        }
                        Jtr += J.transpose() * (scale_reproj * weight * res);
                    }
                }
            }

            if (weight_sampson > 0.0) {
                double C = x2[i].homogeneous().dot(E * x1[i].homogeneous());

                // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
                Eigen::Vector4d J_C;
                J_C << E.block<3, 2>(0, 0).transpose() * x2[i].homogeneous(), E.block<2, 3>(0, 0) * x1[i].homogeneous();
                const double nJ_C = J_C.norm();
                const double inv_nJ_C = 1.0 / nJ_C;
                const double r = C * inv_nJ_C;

                // Compute weight from robust loss function (used in the IRLS)
                const double weight = weights[i] * loss_fn.weight(r * r) * weight_sampson;
                if (weight > 0) {
                    num_residuals++;

                    // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
                    Eigen::Matrix<double, 1, 9> dF;
                    dF << x1[i](0) * x2[i](0), x1[i](0) * x2[i](1), x1[i](0), x1[i](1) * x2[i](0), x1[i](1) * x2[i](1),
                        x1[i](1), x2[i](0), x2[i](1), 1.0;
                    const double s = C * inv_nJ_C * inv_nJ_C;
                    dF(0) -= s * (J_C(2) * x1[i](0) + J_C(0) * x2[i](0));
                    dF(1) -= s * (J_C(3) * x1[i](0) + J_C(0) * x2[i](1));
                    dF(2) -= s * (J_C(0));
                    dF(3) -= s * (J_C(2) * x1[i](1) + J_C(1) * x2[i](0));
                    dF(4) -= s * (J_C(3) * x1[i](1) + J_C(1) * x2[i](1));
                    dF(5) -= s * (J_C(1));
                    dF(6) -= s * (J_C(2));
                    dF(7) -= s * (J_C(3));
                    dF *= inv_nJ_C;

                    J_sam.block<1, 3>(0, 0) = dF * dR;
                    J_sam.block<1, 3>(0, 3) = dF * dt;

                    for (int k = 0; k < 7; ++k) {
                        for (int j = 0; j <= k; ++j) {
                            JtJ(k, j) += weight_sampson * weight * (J_sam(k) * J_sam(j));
                        }
                    }
                    Jtr += weight_sampson * weight * C * inv_nJ_C * J_sam.transpose();
                }
            }
        }
        return num_residuals;
    }

    MonoDepthTwoViewGeometry step(Eigen::Matrix<double, 7, 1> dp, const MonoDepthTwoViewGeometry &geometry) const {
        MonoDepthTwoViewGeometry geometry_new;
        geometry_new.pose.q = quat_step_post(geometry.pose.q, dp.block<3, 1>(0, 0));
        //        geometry_new.t = geometry.t + geometry.rotate(dp.block<3, 1>(3, 0));
        geometry_new.pose.t = geometry.pose.t + dp.block<3, 1>(3, 0);
        geometry_new.scale = geometry.scale + dp(6, 0);
        return geometry_new;
    }
    typedef MonoDepthTwoViewGeometry param_t;
    static constexpr size_t num_params = 7;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;
    const LossFunction &loss_fn;
    const double scale_reproj, weight_sampson;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class SharedFocalRelativePoseJacobianAccumulator {
  public:
    SharedFocalRelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                               const std::vector<Point2D> &points2D_2, const LossFunction &l,
                                               const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const ImagePair &image_pair) const {
        Eigen::Matrix3d E;
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d K_inv;
        K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();

        Eigen::Matrix3d F = K_inv * (E * K_inv);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const ImagePair &image_pair, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(image_pair.pose.t.x()) < std::abs(image_pair.pose.t.y())) {
            // x < y
            if (std::abs(image_pair.pose.t.x()) < std::abs(image_pair.pose.t.z())) {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(image_pair.pose.t.y()) < std::abs(image_pair.pose.t.z())) {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(image_pair.pose.t).normalized();

        double focal = image_pair.camera1.focal();
        Eigen::Matrix3d K_inv;
        K_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, focal;

        Eigen::Matrix3d E, R;
        R = image_pair.pose.R();
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d F = K_inv * (E * K_inv);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        dR.row(2) *= focal;
        dR.row(5) *= focal;
        dR.row(6) *= focal;
        dR.row(7) *= focal;
        dR.row(8) *= focal * focal;

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        dt.row(2) *= focal;
        dt.row(5) *= focal;
        dt.row(6) *= focal;
        dt.row(7) *= focal;
        dt.row(8) *= focal * focal;

        Eigen::Matrix<double, 9, 1> df;

        df << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), E(0, 2), E(1, 2), 2 * E(2, 2) * focal;

        size_t num_residuals = 0;

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;
            J(0, 5) = dF * df;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            JtJ(0, 0) += weight * (J(0) * J(0));
            JtJ(1, 0) += weight * (J(1) * J(0));
            JtJ(1, 1) += weight * (J(1) * J(1));
            JtJ(2, 0) += weight * (J(2) * J(0));
            JtJ(2, 1) += weight * (J(2) * J(1));
            JtJ(2, 2) += weight * (J(2) * J(2));
            JtJ(3, 0) += weight * (J(3) * J(0));
            JtJ(3, 1) += weight * (J(3) * J(1));
            JtJ(3, 2) += weight * (J(3) * J(2));
            JtJ(3, 3) += weight * (J(3) * J(3));
            JtJ(4, 0) += weight * (J(4) * J(0));
            JtJ(4, 1) += weight * (J(4) * J(1));
            JtJ(4, 2) += weight * (J(4) * J(2));
            JtJ(4, 3) += weight * (J(4) * J(3));
            JtJ(4, 4) += weight * (J(4) * J(4));
            JtJ(5, 0) += weight * (J(5) * J(0));
            JtJ(5, 1) += weight * (J(5) * J(1));
            JtJ(5, 2) += weight * (J(5) * J(2));
            JtJ(5, 3) += weight * (J(5) * J(3));
            JtJ(5, 4) += weight * (J(5) * J(4));
            JtJ(5, 5) += weight * (J(5) * J(5));
        }
        return num_residuals;
    }

    ImagePair step(Eigen::Matrix<double, 6, 1> dp, const ImagePair &image_pair) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(image_pair.pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = image_pair.pose.t + tangent_basis * dp.block<2, 1>(3, 0);

        Camera camera_new =
            Camera(SimplePinholeCameraModel::model_id,
                   std::vector<double>{std::max(image_pair.camera1.focal() + dp(5, 0), 0.0), 0.0, 0.0}, -1, -1);
        ImagePair calib_pose_new(pose_new, camera_new, camera_new);
        return calib_pose_new;
    }
    typedef ImagePair param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class VaryingFocalRelativePoseJacobianAccumulator {
  public:
    VaryingFocalRelativePoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                                const std::vector<Point2D> &points2D_2, const LossFunction &l,
                                                const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const ImagePair &image_pair) const {
        Eigen::Matrix3d E;
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d K1_inv, K2_inv;
        K1_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera1.focal();
        K2_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, image_pair.camera2.focal();

        Eigen::Matrix3d F = K2_inv * (E * K1_inv);

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const ImagePair &image_pair, Eigen::Matrix<double, 7, 7> &JtJ, Eigen::Matrix<double, 7, 1> &Jtr) {
        // We start by setting up a basis for the updates in the translation (orthogonal to t)
        // We find the minimum element of t and cross product with the corresponding basis vector.
        // (this ensures that the first cross product is not close to the zero vector)
        if (std::abs(image_pair.pose.t.x()) < std::abs(image_pair.pose.t.y())) {
            // x < y
            if (std::abs(image_pair.pose.t.x()) < std::abs(image_pair.pose.t.z())) {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitX()).normalized();
            } else {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        } else {
            // x > y
            if (std::abs(image_pair.pose.t.y()) < std::abs(image_pair.pose.t.z())) {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitY()).normalized();
            } else {
                tangent_basis.col(0) = image_pair.pose.t.cross(Eigen::Vector3d::UnitZ()).normalized();
            }
        }
        tangent_basis.col(1) = tangent_basis.col(0).cross(image_pair.pose.t).normalized();

        double f1 = image_pair.camera1.focal();
        double f2 = image_pair.camera2.focal();
        Eigen::Matrix3d K1_inv, K2_inv;
        K1_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, f1;
        K2_inv << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, f2;

        Eigen::Matrix3d E, R;
        R = image_pair.pose.R();
        essential_from_motion(image_pair.pose, &E);
        Eigen::Matrix3d F = K2_inv * (E * K1_inv);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        dR.row(2) *= f2;
        dR.row(5) *= f2;
        dR.row(6) *= f1;
        dR.row(7) *= f1;
        dR.row(8) *= f2 * f1;

        // Each column is vec(skew(tangent_basis[k])*R)
        dt.block<3, 1>(0, 0) = tangent_basis.col(0).cross(R.col(0));
        dt.block<3, 1>(0, 1) = tangent_basis.col(1).cross(R.col(0));
        dt.block<3, 1>(3, 0) = tangent_basis.col(0).cross(R.col(1));
        dt.block<3, 1>(3, 1) = tangent_basis.col(1).cross(R.col(1));
        dt.block<3, 1>(6, 0) = tangent_basis.col(0).cross(R.col(2));
        dt.block<3, 1>(6, 1) = tangent_basis.col(1).cross(R.col(2));

        dt.row(2) *= f2;
        dt.row(5) *= f2;
        dt.row(6) *= f1;
        dt.row(7) *= f1;
        dt.row(8) *= f2 * f1;

        Eigen::Matrix<double, 9, 1> df1, df2;

        df1 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, E(0, 2), E(1, 2), E(2, 2) * f2;
        df2 << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), 0.0, 0.0, E(2, 2) * f1;

        size_t num_residuals = 0;

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 7> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;
            J(0, 5) = dF * df1;
            J(0, 6) = dF * df2;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            for (size_t i = 0; i < 7; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }
        return num_residuals;
    }

    ImagePair step(Eigen::Matrix<double, 7, 1> dp, const ImagePair &image_pair) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(image_pair.pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = image_pair.pose.t + tangent_basis * dp.block<2, 1>(3, 0);

        Camera camera1_new =
            Camera(SimplePinholeCameraModel::model_id,
                   std::vector<double>{std::max(image_pair.camera1.focal() + dp(5, 0), 0.0), 0.0, 0.0}, -1, -1);
        Camera camera2_new =
            Camera(SimplePinholeCameraModel::model_id,
                   std::vector<double>{std::max(image_pair.camera2.focal() + dp(6, 0), 0.0), 0.0, 0.0}, -1, -1);
        ImagePair calib_pose_new(pose_new, camera1_new, camera2_new);
        return calib_pose_new;
    }
    typedef ImagePair param_t;
    static constexpr size_t num_params = 7;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class MonoDepthSharedFocalRelPoseJacobianAccumulator {
  public:
    MonoDepthSharedFocalRelPoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                                   const std::vector<Point2D> &points2D_2,
                                                   const std::vector<double> &d1, const std::vector<double> &d2,
                                                   const LossFunction &l, double scale_reproj, double weight_sampson,
                                                   const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), d1(d1), d2(d2), loss_fn(l), scale_reproj(scale_reproj),
          weight_sampson(weight_sampson), weights(w) {}

    double residual(const MonoDepthImagePair &image_pair) const {
        const double focal = image_pair.camera1.focal();
        const double scale = image_pair.geometry.scale;
        Eigen::Matrix3d E;
        essential_from_motion(image_pair.geometry.pose, &E);

        Eigen::DiagonalMatrix<double, 3> K_inv(1 / focal, 1 / focal, 1);
        Eigen::DiagonalMatrix<double, 3> K_inv2(1, 1, focal);
        Eigen::Matrix3d F = K_inv2 * (E * K_inv2);

        Eigen::Matrix3d R = image_pair.geometry.pose.R();
        Eigen::Vector3d t = image_pair.geometry.pose.t;
        double cost = 0;
        for (size_t i = 0; i < x1.size(); ++i) {

            if (weight_sampson > 0.0) {

                double C = x2[i].homogeneous().dot(F * x1[i].homogeneous());
                double nJc_sq = (F.block<2, 3>(0, 0) * x1[i].homogeneous()).squaredNorm() +
                                (F.block<3, 2>(0, 0).transpose() * x2[i].homogeneous()).squaredNorm();

                double r2 = (C * C) / nJc_sq;
                cost += weights[i] * loss_fn.loss(r2) * weight_sampson;
            }
            if (scale_reproj > 0.0) {

                const Eigen::Vector3d Z1 = R * d1[i] * K_inv * x1[i].homogeneous().eval() + t;
                const Eigen::Vector3d Z2 = R.transpose() * (scale * d2[i] * K_inv * x2[i].homogeneous().eval() - t);
                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization

                if (Z1(2) > 0) {
                    const double inv_z1 = 1.0 / Z1(2);
                    const double r10 = Z1(0) * inv_z1 * focal - x2[i](0);
                    const double r11 = Z1(1) * inv_z1 * focal - x2[i](1);
                    const double r_squared1 = r10 * r10 + r11 * r11;
                    cost += weights[i] * loss_fn.loss(scale_reproj * r_squared1);
                }

                if (Z2(2) > 0) {
                    const double inv_z2 = 1.0 / Z2(2);
                    const double r20 = Z2(0) * inv_z2 * focal - x1[i](0);
                    const double r21 = Z2(1) * inv_z2 * focal - x1[i](1);
                    const double r_squared2 = r20 * r20 + r21 * r21;
                    cost += weights[i] * loss_fn.loss(scale_reproj * r_squared2);
                }
            }
        }
        return cost;
    }

    size_t accumulate(const MonoDepthImagePair &image_pair, Eigen::Matrix<double, 8, 8> &JtJ,
                      Eigen::Matrix<double, 8, 1> &Jtr) const {
        Eigen::Matrix3d R = image_pair.geometry.pose.R();
        const double scale = image_pair.geometry.scale;

        Eigen::Matrix<double, 2, 8> J;
        J.setZero();
        Eigen::Matrix<double, 2, 8> Jn;
        Eigen::Matrix<double, 2, 3> Jproj;
        Eigen::Matrix<double, 2, 1> J_params;
        Jproj.setZero();
        Eigen::Matrix<double, 2, 3> Jprojn;
        Jprojn.setZero();

        double focal = image_pair.camera1.focal();
        double inv_f = 1.0 / focal;
        Eigen::DiagonalMatrix<double, 3> K_inv(inv_f, inv_f, 1);
        Eigen::DiagonalMatrix<double, 3> K_inv2(1, 1, focal);

        Eigen::Matrix3d E;
        essential_from_motion(image_pair.geometry.pose, &E);
        Eigen::Matrix3d F = K_inv2 * (E * K_inv2);

        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 3> dt;
        Eigen::Matrix<double, 1, 8> J_sam;
        J_sam.setZero();

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        dR.row(2) *= focal;
        dR.row(5) *= focal;
        dR.row(6) *= focal;
        dR.row(7) *= focal;
        dR.row(8) *= focal * focal;

        Eigen::Matrix3d dt_0, dt_1, dt_2;
        dt_0.row(0).setZero();
        dt_0.row(1) = -R.row(2);
        dt_0.row(2) = R.row(1);

        dt_1.row(0) = R.row(2);
        dt_1.row(1).setZero();
        dt_1.row(2) = -R.row(0);

        dt_2.row(0) = -R.row(1);
        dt_2.row(1) = R.row(0);
        dt_2.row(2).setZero();

        dt.col(0) = Eigen::Map<Eigen::VectorXd>(dt_0.data(), dt_0.size());
        dt.col(1) = Eigen::Map<Eigen::VectorXd>(dt_1.data(), dt_1.size());
        dt.col(2) = Eigen::Map<Eigen::VectorXd>(dt_2.data(), dt_2.size());

        dt.row(2) *= focal;
        dt.row(5) *= focal;
        dt.row(6) *= focal;
        dt.row(7) *= focal;
        dt.row(8) *= focal * focal;

        Eigen::Matrix<double, 9, 1> df;

        df << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), E(0, 2), E(1, 2), 2 * E(2, 2) * focal;

        size_t num_residuals = 0;
        for (size_t i = 0; i < x1.size(); ++i) {

            if (scale_reproj > 0.0) {
                const Eigen::Vector3d Xi = d1[i] * K_inv * x1[i].homogeneous().eval();
                const Eigen::Vector3d Z = R * Xi + image_pair.geometry.pose.t;

                const Eigen::Vector3d Xn = (d2[i]) * K_inv * x2[i].homogeneous().eval();
                const Eigen::Vector3d Xns = scale * Xn;
                const Eigen::Vector3d Zn = R.transpose() * (Xns - image_pair.geometry.pose.t);

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization

                if (Z(2) > 0) {
                    // Project with intrinsics
                    Eigen::Vector2d zp;

                    const double inv_z = 1.0 / Z(2);
                    zp[0] = focal * Z(0) * inv_z;
                    zp[1] = focal * Z(1) * inv_z;

                    double dXdx0 = -d1[i] * inv_f * inv_f * x1[i](0);
                    double dXdx1 = -d1[i] * inv_f * inv_f * x1[i](1);

                    double dZ0df = R(0, 0) * dXdx0 + R(0, 1) * dXdx1;
                    double dZ1df = R(1, 0) * dXdx0 + R(1, 1) * dXdx1;
                    double dZ2df = R(2, 0) * dXdx0 + R(2, 1) * dXdx1;

                    J_params(0) = Z(0) * inv_z;
                    J_params(1) = Z(1) * inv_z;

                    Jproj(0, 0) = focal * inv_z;
                    Jproj(1, 1) = focal * inv_z;
                    Jproj(0, 2) = -focal * Z(0) * inv_z * inv_z;
                    Jproj(1, 2) = -focal * Z(1) * inv_z * inv_z;

                    // Compute reprojection error
                    Eigen::Vector2d res = zp - x2[i];

                    const double r_squared = res.squaredNorm();
                    const double weight = weights[i] * loss_fn.weight(scale_reproj * r_squared);

                    if (weight != 0.0) {
                        num_residuals++;

                        // Jacobian w.r.t. rotation w
                        Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
                        J.col(0) = -Xi(2) * dZ.col(1) + Xi(1) * dZ.col(2);
                        J.col(1) = Xi(2) * dZ.col(0) - Xi(0) * dZ.col(2);
                        J.col(2) = -Xi(1) * dZ.col(0) + Xi(0) * dZ.col(1);
                        J.block<2, 3>(0, 3) = Jproj;
                        J(0, 6) = J_params(0) + focal * (inv_z * dZ0df - Z(0) * inv_z * inv_z * dZ2df);
                        J(1, 6) = J_params(1) + focal * (inv_z * dZ1df - Z(1) * inv_z * inv_z * dZ2df);

                        for (int k = 0; k < 8; ++k) {
                            for (int j = 0; j <= k; ++j) {
                                JtJ(k, j) += scale_reproj * weight * (J.col(k).dot(J.col(j)));
                            }
                        }
                        Jtr += J.transpose() * (scale_reproj * weight * res);
                    }
                }

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Zn(2) > 0) {

                    // Project with intrinsics
                    Eigen::Vector2d zpn;
                    const double inv_zn = 1.0 / Zn(2);
                    zpn[0] = focal * Zn(0) * inv_zn;
                    zpn[1] = focal * Zn(1) * inv_zn;

                    // Compute reprojection error
                    Eigen::Vector2d rn = zpn - x1[i];
                    const double rn_squared = rn.squaredNorm();
                    const double weightn = weights[i] * loss_fn.weight(scale_reproj * rn_squared);

                    if (weightn != 0.0) {
                        num_residuals++;
                        Jprojn(0, 0) = focal * inv_zn;
                        Jprojn(1, 1) = focal * inv_zn;
                        Jprojn(0, 2) = -focal * Zn(0) * inv_zn * inv_zn;
                        Jprojn(1, 2) = -focal * Zn(1) * inv_zn * inv_zn;

                        Eigen::Matrix3d Zx;
                        Zx << 0, -Zn(2), Zn(1), Zn(2), 0, -Zn(0), -Zn(1), Zn(0), 0;

                        Jn.block<2, 3>(0, 0) = Jprojn * Zx;
                        Jn.block<2, 3>(0, 3) = -Jprojn * R.transpose();

                        double dXdx0n = -scale * d2[i] * inv_f * inv_f * x2[i](0);
                        double dXdx1n = -scale * d2[i] * inv_f * inv_f * x2[i](1);

                        double dZ0dfn = R(0, 0) * dXdx0n + R(1, 0) * dXdx1n;
                        double dZ1dfn = R(0, 1) * dXdx0n + R(1, 1) * dXdx1n;
                        double dZ2dfn = R(0, 2) * dXdx0n + R(1, 2) * dXdx1n;

                        Jn(0, 6) = Zn(0) * inv_zn + focal * (inv_zn * dZ0dfn - Zn(0) * inv_zn * inv_zn * dZ2dfn);
                        Jn(1, 6) = Zn(1) * inv_zn + focal * (inv_zn * dZ1dfn - Zn(1) * inv_zn * inv_zn * dZ2dfn);

                        Jn.block<2, 1>(0, 7) = Jprojn * R.transpose() * Xn;

                        for (int k = 0; k < 8; ++k) {
                            for (int j = 0; j <= k; ++j) {
                                JtJ(k, j) += scale_reproj * weightn * (Jn.col(k).dot(Jn.col(j)));
                            }
                        }
                        Jtr += Jn.transpose() * (scale_reproj * weightn * rn);
                    }
                }
            }

            if (weight_sampson > 0.0) {
                double C = x2[i].homogeneous().dot(F * x1[i].homogeneous());

                // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
                Eigen::Vector4d J_C;
                J_C << F.block<3, 2>(0, 0).transpose() * x2[i].homogeneous(), F.block<2, 3>(0, 0) * x1[i].homogeneous();
                const double nJ_C = J_C.norm();
                const double inv_nJ_C = 1.0 / nJ_C;
                const double r = C * inv_nJ_C;

                // Compute weight from robust loss function (used in the IRLS)
                // std::cout << monodepth_weight_sampson << std::endl;
                const double weight = weights[i] * loss_fn.weight(weight_sampson * r * r) * weight_sampson;
                if (weight > 0) {
                    num_residuals++;

                    // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
                    Eigen::Matrix<double, 1, 9> dF;
                    dF << x1[i](0) * x2[i](0), x1[i](0) * x2[i](1), x1[i](0), x1[i](1) * x2[i](0), x1[i](1) * x2[i](1),
                        x1[i](1), x2[i](0), x2[i](1), 1.0;
                    const double s = C * inv_nJ_C * inv_nJ_C;
                    dF(0) -= s * (J_C(2) * x1[i](0) + J_C(0) * x2[i](0));
                    dF(1) -= s * (J_C(3) * x1[i](0) + J_C(0) * x2[i](1));
                    dF(2) -= s * (J_C(0));
                    dF(3) -= s * (J_C(2) * x1[i](1) + J_C(1) * x2[i](0));
                    dF(4) -= s * (J_C(3) * x1[i](1) + J_C(1) * x2[i](1));
                    dF(5) -= s * (J_C(1));
                    dF(6) -= s * (J_C(2));
                    dF(7) -= s * (J_C(3));
                    dF *= inv_nJ_C;

                    J_sam.block<1, 3>(0, 0) = dF * dR;
                    J_sam.block<1, 3>(0, 3) = dF * dt;
                    J_sam(0, 6) = dF * df;

                    for (int k = 0; k < 8; ++k) {
                        for (int j = 0; j <= k; ++j) {
                            JtJ(k, j) += weight_sampson * weight * (J_sam(k) * J_sam(j));
                        }
                    }
                    Jtr += weight_sampson * weight * C * inv_nJ_C * J_sam.transpose();
                }
            }
        }
        return num_residuals;
    }

    MonoDepthImagePair step(Eigen::Matrix<double, 8, 1> dp, const MonoDepthImagePair &image_pair) const {
        MonoDepthTwoViewGeometry geometry_new;
        geometry_new.pose.q = quat_step_post(image_pair.geometry.pose.q, dp.block<3, 1>(0, 0));
        geometry_new.pose.t = image_pair.geometry.pose.t + (dp.block<3, 1>(3, 0));
        geometry_new.scale = image_pair.geometry.scale + dp(7, 0);

        Camera camera_new =
            Camera(SimplePinholeCameraModel::model_id,
                   std::vector<double>{std::max(image_pair.camera1.focal() + dp(6, 0), 0.0), 0.0, 0.0}, -1, -1);
        MonoDepthImagePair calib_pose_new(geometry_new, camera_new, camera_new);
        return calib_pose_new;
    }
    typedef ImagePair param_t;
    static constexpr size_t num_params = 8;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;
    const LossFunction &loss_fn;
    const double scale_reproj;
    const double weight_sampson;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class MonoDepthVaryingFocalRelPoseJacobianAccumulator {
  public:
    MonoDepthVaryingFocalRelPoseJacobianAccumulator(const std::vector<Point2D> &points2D_1,
                                                    const std::vector<Point2D> &points2D_2,
                                                    const std::vector<double> &d1, const std::vector<double> &d2,
                                                    const LossFunction &l, const double scale_reproj,
                                                    const double weight_sampson,
                                                    const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), d1(d1), d2(d2), loss_fn(l), scale_reproj(scale_reproj),
          weight_sampson(weight_sampson), weights(w) {}

    double residual(const MonoDepthImagePair &image_pair) const {
        const double focal_1 = image_pair.camera1.focal();
        const double focal_2 = image_pair.camera2.focal();
        const double scale = image_pair.geometry.scale;
        Eigen::Matrix3d E;
        essential_from_motion(image_pair.geometry.pose, &E);

        Eigen::DiagonalMatrix<double, 3> K1_inv(1 / focal_1, 1 / focal_1, 1);
        Eigen::DiagonalMatrix<double, 3> K1_inv_p(1, 1, focal_1);
        Eigen::DiagonalMatrix<double, 3> K2_inv(1 / focal_2, 1 / focal_2, 1);
        Eigen::DiagonalMatrix<double, 3> K2_inv_p(1, 1, focal_2);
        Eigen::Matrix3d F = K2_inv_p * (E * K1_inv_p);

        Eigen::Matrix3d R = image_pair.geometry.pose.R();
        Eigen::Vector3d t = image_pair.geometry.pose.t;
        double cost = 0;
        for (size_t i = 0; i < x1.size(); ++i) {

            if (weight_sampson > 0.0) {

                double C = x2[i].homogeneous().dot(F * x1[i].homogeneous());
                double nJc_sq = (F.block<2, 3>(0, 0) * x1[i].homogeneous()).squaredNorm() +
                                (F.block<3, 2>(0, 0).transpose() * x2[i].homogeneous()).squaredNorm();

                double r2 = (C * C) / nJc_sq;
                cost += weights[i] * loss_fn.loss(r2) * weight_sampson;
            }
            if (scale_reproj > 0.0) {

                const Eigen::Vector3d Z1 = R * d1[i] * K1_inv * x1[i].homogeneous().eval() + t;
                const Eigen::Vector3d Z2 = R.transpose() * (scale * d2[i] * K2_inv * x2[i].homogeneous().eval() - t);
                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization

                if (Z1(2) > 0) {
                    const double inv_z1 = 1.0 / Z1(2);
                    const double r10 = Z1(0) * inv_z1 * focal_2 - x2[i](0);
                    const double r11 = Z1(1) * inv_z1 * focal_2 - x2[i](1);
                    const double r_squared1 = r10 * r10 + r11 * r11;
                    cost += weights[i] * loss_fn.loss(scale_reproj * r_squared1);
                }

                if (Z2(2) > 0) {
                    const double inv_z2 = 1.0 / Z2(2);
                    const double r20 = Z2(0) * inv_z2 * focal_1 - x1[i](0);
                    const double r21 = Z2(1) * inv_z2 * focal_1 - x1[i](1);
                    const double r_squared2 = r20 * r20 + r21 * r21;
                    cost += weights[i] * loss_fn.loss(scale_reproj * r_squared2);
                }
            }
        }
        return cost;
    }

    size_t accumulate(const MonoDepthImagePair &image_pair, Eigen::Matrix<double, 9, 9> &JtJ,
                      Eigen::Matrix<double, 9, 1> &Jtr) const {
        Eigen::Matrix3d R = image_pair.geometry.pose.R();
        const double scale = image_pair.geometry.scale;

        Eigen::Matrix<double, 2, 9> J;
        J.setZero();
        Eigen::Matrix<double, 2, 9> Jn;
        Jn.setZero();
        Eigen::Matrix<double, 2, 3> Jproj;
        Jproj.setZero();
        Eigen::Matrix<double, 2, 1> J_params;
        J_params.setZero();

        Eigen::Matrix<double, 2, 3> Jprojn;
        Jprojn.setZero();

        double focal_1 = image_pair.camera1.focal();
        double focal_2 = image_pair.camera2.focal();
        double inv_f_1 = 1.0 / focal_1;
        double inv_f_2 = 1.0 / focal_2;
        Eigen::DiagonalMatrix<double, 3> K1_inv(inv_f_1, inv_f_1, 1);
        Eigen::DiagonalMatrix<double, 3> K1_inv_p(1, 1, focal_1);
        Eigen::DiagonalMatrix<double, 3> K2_inv(inv_f_2, inv_f_2, 1);
        Eigen::DiagonalMatrix<double, 3> K2_inv_p(1, 1, focal_2);

        Eigen::Matrix3d E;
        essential_from_motion(image_pair.geometry.pose, &E);
        Eigen::Matrix3d F = K2_inv_p * (E * K1_inv_p);

        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 3> dt;
        Eigen::Matrix<double, 1, 9> J_sam;
        J_sam.setZero();

        // Each column is vec(E*skew(e_k)) where e_k is k:th basis vector
        dR.block<3, 1>(0, 0).setZero();
        dR.block<3, 1>(0, 1) = -E.col(2);
        dR.block<3, 1>(0, 2) = E.col(1);
        dR.block<3, 1>(3, 0) = E.col(2);
        dR.block<3, 1>(3, 1).setZero();
        dR.block<3, 1>(3, 2) = -E.col(0);
        dR.block<3, 1>(6, 0) = -E.col(1);
        dR.block<3, 1>(6, 1) = E.col(0);
        dR.block<3, 1>(6, 2).setZero();

        dR.row(2) *= focal_2;
        dR.row(5) *= focal_2;
        dR.row(6) *= focal_1;
        dR.row(7) *= focal_1;
        dR.row(8) *= focal_2 * focal_1;

        Eigen::Matrix3d dt_0, dt_1, dt_2;
        dt_0.row(0).setZero();
        dt_0.row(1) = -R.row(2);
        dt_0.row(2) = R.row(1);

        dt_1.row(0) = R.row(2);
        dt_1.row(1).setZero();
        dt_1.row(2) = -R.row(0);

        dt_2.row(0) = -R.row(1);
        dt_2.row(1) = R.row(0);
        dt_2.row(2).setZero();

        dt.col(0) = Eigen::Map<Eigen::VectorXd>(dt_0.data(), dt_0.size());
        dt.col(1) = Eigen::Map<Eigen::VectorXd>(dt_1.data(), dt_1.size());
        dt.col(2) = Eigen::Map<Eigen::VectorXd>(dt_2.data(), dt_2.size());

        dt.row(2) *= focal_2;
        dt.row(5) *= focal_2;
        dt.row(6) *= focal_1;
        dt.row(7) *= focal_1;
        dt.row(8) *= focal_2 * focal_1;

        Eigen::Matrix<double, 9, 1> df1, df2;

        df1 << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, E(0, 2), E(1, 2), E(2, 2) * focal_2;
        df2 << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), 0.0, 0.0, E(2, 2) * focal_1;

        size_t num_residuals = 0;
        for (size_t i = 0; i < x1.size(); ++i) {

            if (scale_reproj > 0.0) {
                const Eigen::Vector3d Xi = d1[i] * K1_inv * x1[i].homogeneous().eval();
                const Eigen::Vector3d Z = R * Xi + image_pair.geometry.pose.t;

                const Eigen::Vector3d Xn = (d2[i]) * K2_inv * x2[i].homogeneous().eval();
                const Eigen::Vector3d Xns = scale * Xn;
                const Eigen::Vector3d Zn = R.transpose() * (Xns - image_pair.geometry.pose.t);

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization

                if (Z(2) > 0) {
                    // Project with intrinsics
                    Eigen::Vector2d zp;

                    const double inv_z = 1.0 / Z(2);
                    zp[0] = focal_2 * Z(0) * inv_z;
                    zp[1] = focal_2 * Z(1) * inv_z;

                    double dXdx0 = -d1[i] * inv_f_1 * inv_f_1 * x1[i](0);
                    double dXdx1 = -d1[i] * inv_f_1 * inv_f_1 * x1[i](1);

                    double dZ0df = R(0, 0) * dXdx0 + R(0, 1) * dXdx1;
                    double dZ1df = R(1, 0) * dXdx0 + R(1, 1) * dXdx1;
                    double dZ2df = R(2, 0) * dXdx0 + R(2, 1) * dXdx1;

                    J_params(0) = Z(0) * inv_z;
                    J_params(1) = Z(1) * inv_z;

                    Jproj(0, 0) = focal_2 * inv_z;
                    Jproj(1, 1) = focal_2 * inv_z;
                    Jproj(0, 2) = -focal_2 * Z(0) * inv_z * inv_z;
                    Jproj(1, 2) = -focal_2 * Z(1) * inv_z * inv_z;

                    // Compute reprojection error
                    Eigen::Vector2d res = zp - x2[i];

                    const double r_squared = res.squaredNorm();
                    const double weight = weights[i] * loss_fn.weight(scale_reproj * r_squared);

                    if (weight != 0.0) {
                        num_residuals++;

                        // Jacobian w.r.t. rotation w
                        Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
                        J.col(0) = -Xi(2) * dZ.col(1) + Xi(1) * dZ.col(2);
                        J.col(1) = Xi(2) * dZ.col(0) - Xi(0) * dZ.col(2);
                        J.col(2) = -Xi(1) * dZ.col(0) + Xi(0) * dZ.col(1);
                        J.block<2, 3>(0, 3) = Jproj;
                        J(0, 6) = focal_2 * (inv_z * dZ0df - Z(0) * inv_z * inv_z * dZ2df);
                        J(1, 6) = focal_2 * (inv_z * dZ1df - Z(1) * inv_z * inv_z * dZ2df);
                        J.col(7) = J_params;

                        for (int k = 0; k < 9; ++k) {
                            for (int j = 0; j <= k; ++j) {
                                JtJ(k, j) += scale_reproj * weight * (J.col(k).dot(J.col(j)));
                            }
                        }
                        Jtr += J.transpose() * (scale_reproj * weight * res);
                    }
                }

                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Zn(2) > 0) {

                    // Project with intrinsics
                    Eigen::Vector2d zpn;
                    const double inv_zn = 1.0 / Zn(2);
                    zpn[0] = focal_1 * Zn(0) * inv_zn;
                    zpn[1] = focal_1 * Zn(1) * inv_zn;

                    // Compute reprojection error
                    Eigen::Vector2d rn = zpn - x1[i];
                    const double rn_squared = rn.squaredNorm();
                    const double weightn = weights[i] * loss_fn.weight(scale_reproj * rn_squared);

                    if (weightn != 0.0) {
                        num_residuals++;
                        Jprojn(0, 0) = focal_1 * inv_zn;
                        Jprojn(1, 1) = focal_1 * inv_zn;
                        Jprojn(0, 2) = -focal_1 * Zn(0) * inv_zn * inv_zn;
                        Jprojn(1, 2) = -focal_1 * Zn(1) * inv_zn * inv_zn;

                        Eigen::Matrix3d Zx;
                        Zx << 0, -Zn(2), Zn(1), Zn(2), 0, -Zn(0), -Zn(1), Zn(0), 0;

                        Jn.block<2, 3>(0, 0) = Jprojn * Zx;
                        Jn.block<2, 3>(0, 3) = -Jprojn * R.transpose();

                        double dXdx0n = -scale * d2[i] * inv_f_2 * inv_f_2 * x2[i](0);
                        double dXdx1n = -scale * d2[i] * inv_f_2 * inv_f_2 * x2[i](1);

                        double dZ0dfn = R(0, 0) * dXdx0n + R(1, 0) * dXdx1n;
                        double dZ1dfn = R(0, 1) * dXdx0n + R(1, 1) * dXdx1n;
                        double dZ2dfn = R(0, 2) * dXdx0n + R(1, 2) * dXdx1n;

                        Jn(0, 6) = Zn(0) * inv_zn;
                        Jn(1, 6) = Zn(1) * inv_zn;
                        Jn(0, 7) = focal_1 * (inv_zn * dZ0dfn - Zn(0) * inv_zn * inv_zn * dZ2dfn);
                        Jn(1, 7) = focal_1 * (inv_zn * dZ1dfn - Zn(1) * inv_zn * inv_zn * dZ2dfn);

                        Jn.block<2, 1>(0, 8) = Jprojn * R.transpose() * Xn;
                        for (int k = 0; k < 9; ++k) {
                            for (int j = 0; j <= k; ++j) {
                                JtJ(k, j) += scale_reproj * weightn * (Jn.col(k).dot(Jn.col(j)));
                            }
                        }
                        Jtr += Jn.transpose() * (scale_reproj * weightn * rn);
                    }
                }
            }

            if (weight_sampson > 0.0) {
                double C = x2[i].homogeneous().dot(F * x1[i].homogeneous());

                // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
                Eigen::Vector4d J_C;
                J_C << F.block<3, 2>(0, 0).transpose() * x2[i].homogeneous(), F.block<2, 3>(0, 0) * x1[i].homogeneous();
                const double nJ_C = J_C.norm();
                const double inv_nJ_C = 1.0 / nJ_C;
                const double r = C * inv_nJ_C;

                // Compute weight from robust loss function (used in the IRLS)
                // std::cout << monodepth_weight_sampson << std::endl;
                const double weight = weights[i] * loss_fn.weight(weight_sampson * r * r) * weight_sampson;
                if (weight > 0) {
                    num_residuals++;

                    // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
                    Eigen::Matrix<double, 1, 9> dF;
                    dF << x1[i](0) * x2[i](0), x1[i](0) * x2[i](1), x1[i](0), x1[i](1) * x2[i](0), x1[i](1) * x2[i](1),
                        x1[i](1), x2[i](0), x2[i](1), 1.0;
                    const double s = C * inv_nJ_C * inv_nJ_C;
                    dF(0) -= s * (J_C(2) * x1[i](0) + J_C(0) * x2[i](0));
                    dF(1) -= s * (J_C(3) * x1[i](0) + J_C(0) * x2[i](1));
                    dF(2) -= s * (J_C(0));
                    dF(3) -= s * (J_C(2) * x1[i](1) + J_C(1) * x2[i](0));
                    dF(4) -= s * (J_C(3) * x1[i](1) + J_C(1) * x2[i](1));
                    dF(5) -= s * (J_C(1));
                    dF(6) -= s * (J_C(2));
                    dF(7) -= s * (J_C(3));
                    dF *= inv_nJ_C;

                    J_sam.block<1, 3>(0, 0) = dF * dR;
                    J_sam.block<1, 3>(0, 3) = dF * dt;
                    J_sam(0, 6) = dF * df1;
                    J_sam(0, 7) = dF * df2;

                    for (int k = 0; k < 9; ++k) {
                        for (int j = 0; j <= k; ++j) {
                            JtJ(k, j) += weight_sampson * weight * (J_sam(k) * J_sam(j));
                        }
                    }
                    Jtr += weight_sampson * weight * C * inv_nJ_C * J_sam.transpose();
                }
            }
        }
        return num_residuals;
    }

    MonoDepthImagePair step(Eigen::Matrix<double, 9, 1> dp, const MonoDepthImagePair &image_pair) const {
        MonoDepthTwoViewGeometry geometry;
        geometry.pose.q = quat_step_post(image_pair.geometry.pose.q, dp.block<3, 1>(0, 0));
        geometry.pose.t = image_pair.geometry.pose.t + (dp.block<3, 1>(3, 0));
        geometry.scale = image_pair.geometry.scale + dp(8, 0);

        Camera camera_new_1 =
            Camera(SimplePinholeCameraModel::model_id,
                   std::vector<double>{std::max(image_pair.camera1.focal() + dp(6, 0), 0.0), 0.0, 0.0}, -1, -1);
        Camera camera_new_2 =
            Camera(SimplePinholeCameraModel::model_id,
                   std::vector<double>{std::max(image_pair.camera2.focal() + dp(7, 0), 0.0), 0.0, 0.0}, -1, -1);
        MonoDepthImagePair calib_pose_new(geometry, camera_new_1, camera_new_2);
        return calib_pose_new;
    }
    typedef MonoDepthImagePair param_t;
    static constexpr size_t num_params = 9;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;
    const LossFunction &loss_fn;
    const double scale_reproj, weight_sampson;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVectors = UniformWeightVectors>
class GeneralizedRelativePoseJacobianAccumulator {
  public:
    GeneralizedRelativePoseJacobianAccumulator(const std::vector<PairwiseMatches> &pairwise_matches,
                                               const std::vector<CameraPose> &camera1_ext,
                                               const std::vector<CameraPose> &camera2_ext, const LossFunction &l,
                                               const ResidualWeightVectors &w = ResidualWeightVectors())
        : matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            const PairwiseMatches &m = matches[match_k];
            Eigen::Vector4d q1 = rig1_poses[m.cam_id1].q;
            Eigen::Vector3d t1 = rig1_poses[m.cam_id1].t;

            Eigen::Vector4d q2 = rig2_poses[m.cam_id2].q;
            Eigen::Vector3d t2 = rig2_poses[m.cam_id2].t;

            CameraPose relpose;
            relpose.q = quat_multiply(q2, quat_multiply(pose.q, quat_conj(q1)));
            relpose.t = t2 + quat_rotate(q2, pose.t) - relpose.rotate(t1);
            RelativePoseJacobianAccumulator<LossFunction, typename ResidualWeightVectors::value_type> accum(
                m.x1, m.x2, loss_fn, weights[match_k]);
            cost += accum.residual(relpose);
        }
        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        Eigen::Matrix3d R = pose.R();
        size_t num_residuals = 0;
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            const PairwiseMatches &m = matches[match_k];

            // Cameras are
            // [R1 t1]
            // [R2 t2] * [R t; 0 1] = [R2*R t2+R2*t]

            // Relative pose is
            // [R2*R*R1' t2+R2*t-R2*R*R1'*t1]
            // Essential matrix is
            // [t2]_x*R2*R*R1' + [R2*t]_x*R2*R*R1' - R2*R*R1'*[t1]_x

            Eigen::Vector4d q1 = rig1_poses[m.cam_id1].q;
            Eigen::Matrix3d R1 = quat_to_rotmat(q1);
            Eigen::Vector3d t1 = rig1_poses[m.cam_id1].t;

            Eigen::Vector4d q2 = rig2_poses[m.cam_id2].q;
            Eigen::Matrix3d R2 = quat_to_rotmat(q2);
            Eigen::Vector3d t2 = rig2_poses[m.cam_id2].t;

            CameraPose relpose;
            relpose.q = quat_multiply(q2, quat_multiply(pose.q, quat_conj(q1)));
            relpose.t = t2 + R2 * pose.t - relpose.rotate(t1);
            Eigen::Matrix3d E;
            essential_from_motion(relpose, &E);

            Eigen::Matrix3d R2R = R2 * R;
            Eigen::Vector3d Rt = R.transpose() * pose.t;

            // The messy expressions below compute
            // dRdw = [vec(S1) vec(S2) vec(S3)];
            // dR = (kron(R1,skew(t2)*R2R+ R2*skew(t)*R) + kron(skew(t1)*R1,R2*R)) * dRdw
            // dt = [vec(R2*R*S1*R1.') vec(R2*R*S2*R1.') vec(R2*R*S3*R1.')]

            // TODO: Replace with something nice
            Eigen::Matrix<double, 9, 3> dR;
            Eigen::Matrix<double, 9, 3> dt;
            dR(0, 0) = R2R(0, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(0, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(0, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(0, 1) = R2R(0, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(0, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(0, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(0, 2) = R2R(0, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(0, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(0, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(1, 0) = R2R(1, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(1, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(0, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(1, 1) = R2R(1, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(1, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(0, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(1, 2) = R2R(1, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(1, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(0, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(2, 0) = R2R(2, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R2R(2, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
                       R1(0, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(0, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(2, 1) = R2R(2, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R2R(2, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
                       R1(0, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(0, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(2, 2) = R2R(2, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) -
                       R2R(2, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
                       R1(0, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(0, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(3, 0) = R2R(0, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(0, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(1, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(3, 1) = R2R(0, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(0, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(1, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(3, 2) = R2R(0, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(0, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(1, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(4, 0) = R2R(1, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(1, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(1, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(4, 1) = R2R(1, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(1, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(1, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(4, 2) = R2R(1, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(1, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(1, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(5, 0) = R2R(2, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R2R(2, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
                       R1(1, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(1, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(5, 1) = R2R(2, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) -
                       R2R(2, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R1(1, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(1, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(5, 2) = R2R(2, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
                       R2R(2, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
                       R1(1, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(1, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(6, 0) = R2R(0, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(0, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(2, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
            dR(6, 1) = R2R(0, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(0, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
                       R1(2, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(6, 2) = R2R(0, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(0, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
                       R1(2, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
            dR(7, 0) = R2R(1, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(1, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(2, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
            dR(7, 1) = R2R(1, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(1, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
                       R1(2, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(7, 2) = R2R(1, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(1, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
                       R1(2, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
            dR(8, 0) = R2R(2, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R2R(2, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
                       R1(2, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(2, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
            dR(8, 1) = R2R(2, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R2R(2, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
                       R1(2, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
                       R1(2, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dR(8, 2) = R2R(2, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) -
                       R2R(2, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
                       R1(2, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
                       R1(2, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
            dt(0, 0) = R2R(0, 2) * R1(0, 1) - R2R(0, 1) * R1(0, 2);
            dt(0, 1) = R2R(0, 0) * R1(0, 2) - R2R(0, 2) * R1(0, 0);
            dt(0, 2) = R2R(0, 1) * R1(0, 0) - R2R(0, 0) * R1(0, 1);
            dt(1, 0) = R2R(1, 2) * R1(0, 1) - R2R(1, 1) * R1(0, 2);
            dt(1, 1) = R2R(1, 0) * R1(0, 2) - R2R(1, 2) * R1(0, 0);
            dt(1, 2) = R2R(1, 1) * R1(0, 0) - R2R(1, 0) * R1(0, 1);
            dt(2, 0) = R2R(2, 2) * R1(0, 1) - R2R(2, 1) * R1(0, 2);
            dt(2, 1) = R2R(2, 0) * R1(0, 2) - R2R(2, 2) * R1(0, 0);
            dt(2, 2) = R2R(2, 1) * R1(0, 0) - R2R(2, 0) * R1(0, 1);
            dt(3, 0) = R2R(0, 2) * R1(1, 1) - R2R(0, 1) * R1(1, 2);
            dt(3, 1) = R2R(0, 0) * R1(1, 2) - R2R(0, 2) * R1(1, 0);
            dt(3, 2) = R2R(0, 1) * R1(1, 0) - R2R(0, 0) * R1(1, 1);
            dt(4, 0) = R2R(1, 2) * R1(1, 1) - R2R(1, 1) * R1(1, 2);
            dt(4, 1) = R2R(1, 0) * R1(1, 2) - R2R(1, 2) * R1(1, 0);
            dt(4, 2) = R2R(1, 1) * R1(1, 0) - R2R(1, 0) * R1(1, 1);
            dt(5, 0) = R2R(2, 2) * R1(1, 1) - R2R(2, 1) * R1(1, 2);
            dt(5, 1) = R2R(2, 0) * R1(1, 2) - R2R(2, 2) * R1(1, 0);
            dt(5, 2) = R2R(2, 1) * R1(1, 0) - R2R(2, 0) * R1(1, 1);
            dt(6, 0) = R2R(0, 2) * R1(2, 1) - R2R(0, 1) * R1(2, 2);
            dt(6, 1) = R2R(0, 0) * R1(2, 2) - R2R(0, 2) * R1(2, 0);
            dt(6, 2) = R2R(0, 1) * R1(2, 0) - R2R(0, 0) * R1(2, 1);
            dt(7, 0) = R2R(1, 2) * R1(2, 1) - R2R(1, 1) * R1(2, 2);
            dt(7, 1) = R2R(1, 0) * R1(2, 2) - R2R(1, 2) * R1(2, 0);
            dt(7, 2) = R2R(1, 1) * R1(2, 0) - R2R(1, 0) * R1(2, 1);
            dt(8, 0) = R2R(2, 2) * R1(2, 1) - R2R(2, 1) * R1(2, 2);
            dt(8, 1) = R2R(2, 0) * R1(2, 2) - R2R(2, 2) * R1(2, 0);
            dt(8, 2) = R2R(2, 1) * R1(2, 0) - R2R(2, 0) * R1(2, 1);

            for (size_t k = 0; k < m.x1.size(); ++k) {
                double C = m.x2[k].homogeneous().dot(E * m.x1[k].homogeneous());

                // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
                Eigen::Vector4d J_C;
                J_C << E.block<3, 2>(0, 0).transpose() * m.x2[k].homogeneous(),
                    E.block<2, 3>(0, 0) * m.x1[k].homogeneous();
                const double nJ_C = J_C.norm();
                const double inv_nJ_C = 1.0 / nJ_C;
                const double r = C * inv_nJ_C;

                // Compute weight from robust loss function (used in the IRLS)
                const double weight = weights[match_k][k] * loss_fn.weight(r * r);
                if (weight == 0.0) {
                    continue;
                }
                num_residuals++;

                // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
                Eigen::Matrix<double, 1, 9> dF;
                dF << m.x1[k](0) * m.x2[k](0), m.x1[k](0) * m.x2[k](1), m.x1[k](0), m.x1[k](1) * m.x2[k](0),
                    m.x1[k](1) * m.x2[k](1), m.x1[k](1), m.x2[k](0), m.x2[k](1), 1.0;
                const double s = C * inv_nJ_C * inv_nJ_C;
                dF(0) -= s * (J_C(2) * m.x1[k](0) + J_C(0) * m.x2[k](0));
                dF(1) -= s * (J_C(3) * m.x1[k](0) + J_C(0) * m.x2[k](1));
                dF(2) -= s * (J_C(0));
                dF(3) -= s * (J_C(2) * m.x1[k](1) + J_C(1) * m.x2[k](0));
                dF(4) -= s * (J_C(3) * m.x1[k](1) + J_C(1) * m.x2[k](1));
                dF(5) -= s * (J_C(1));
                dF(6) -= s * (J_C(2));
                dF(7) -= s * (J_C(3));
                dF *= inv_nJ_C;

                // and then w.r.t. the pose parameters
                Eigen::Matrix<double, 1, 6> J;
                J.block<1, 3>(0, 0) = dF * dR;
                J.block<1, 3>(0, 3) = dF * dt;

                // Accumulate into JtJ and Jtr
                Jtr += weight * C * inv_nJ_C * J.transpose();
                for (size_t i = 0; i < 6; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        JtJ(i, j) += weight * (J(i) * J(j));
                    }
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;
    const LossFunction &loss_fn;
    const ResidualWeightVectors &weights;
};

template <typename LossFunction, typename AbsResidualsVector = UniformWeightVector,
          typename RelResidualsVectors = UniformWeightVectors>
class HybridPoseJacobianAccumulator {
  public:
    HybridPoseJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                  const std::vector<PairwiseMatches> &pairwise_matches,
                                  const std::vector<CameraPose> &map_ext, const LossFunction &l,
                                  const LossFunction &l_epi,
                                  const AbsResidualsVector &weights_abs = AbsResidualsVector(),
                                  const RelResidualsVectors &weights_rel = RelResidualsVectors())
        : abs_pose_accum(points2D, points3D, trivial_camera, l, weights_abs),
          gen_rel_accum(pairwise_matches, map_ext, trivial_rig, l_epi, weights_rel) {
        trivial_camera.model_id = NullCameraModel::model_id;
        trivial_rig.emplace_back();
    }

    double residual(const CameraPose &pose) const {
        return abs_pose_accum.residual(pose) + gen_rel_accum.residual(pose);
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ,
                      Eigen::Matrix<double, 6, 1> &Jtr) const {
        return abs_pose_accum.accumulate(pose, JtJ, Jtr) + gen_rel_accum.accumulate(pose, JtJ, Jtr);
    }

    CameraPose step(Eigen::Matrix<double, 6, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    Camera trivial_camera;
    std::vector<CameraPose> trivial_rig;
    CameraJacobianAccumulator<NullCameraModel, LossFunction, AbsResidualsVector> abs_pose_accum;
    GeneralizedRelativePoseJacobianAccumulator<LossFunction, RelResidualsVectors> gen_rel_accum;
};

// This is the SVD factorization proposed by Bartoli and Sturm in
// Non-Linear Estimation of the Fundamental Matrix With Minimal Parameters, PAMI 2004
// Though we do different updates (lie vs the euler angles used in the original paper)
struct FactorizedFundamentalMatrix {
    FactorizedFundamentalMatrix() {}
    FactorizedFundamentalMatrix(const Eigen::Matrix3d &F) {
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(F, Eigen::ComputeFullV | Eigen::ComputeFullU);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        if (U.determinant() < 0) {
            U = -U;
        }
        if (V.determinant() < 0) {
            V = -V;
        }
        qU = rotmat_to_quat(U);
        qV = rotmat_to_quat(V);
        Eigen::Vector3d s = svd.singularValues();
        sigma = s(1) / s(0);
    }
    Eigen::Matrix3d F() const {
        Eigen::Matrix3d U = quat_to_rotmat(qU);
        Eigen::Matrix3d V = quat_to_rotmat(qV);
        return U.col(0) * V.col(0).transpose() + sigma * U.col(1) * V.col(1).transpose();
    }

    Eigen::Vector4d qU, qV;
    double sigma;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class FundamentalJacobianAccumulator {
  public:
    FundamentalJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                   const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const FactorizedFundamentalMatrix &FF) const {
        Eigen::Matrix3d F = FF.F();

        double cost = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            double r2 = (C * C) / nJc_sq;
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const FactorizedFundamentalMatrix &FF, Eigen::Matrix<double, 7, 7> &JtJ,
                      Eigen::Matrix<double, 7, 1> &Jtr) const {

        const Eigen::Matrix3d F = FF.F();

        // Matrices contain the jacobians of F w.r.t. the factorized fundamental matrix (U,V,sigma)
        const Eigen::Matrix3d U = quat_to_rotmat(FF.qU);
        const Eigen::Matrix3d V = quat_to_rotmat(FF.qV);

        const Eigen::Matrix3d d_sigma = U.col(1) * V.col(1).transpose();
        Eigen::Matrix<double, 9, 7> dF_dparams;
        dF_dparams << 0, F(2, 0), -F(1, 0), 0, F(0, 2), -F(0, 1), d_sigma(0, 0), -F(2, 0), 0, F(0, 0), 0, F(1, 2),
            -F(1, 1), d_sigma(1, 0), F(1, 0), -F(0, 0), 0, 0, F(2, 2), -F(2, 1), d_sigma(2, 0), 0, F(2, 1), -F(1, 1),
            -F(0, 2), 0, F(0, 0), d_sigma(0, 1), -F(2, 1), 0, F(0, 1), -F(1, 2), 0, F(1, 0), d_sigma(1, 1), F(1, 1),
            -F(0, 1), 0, -F(2, 2), 0, F(2, 0), d_sigma(2, 1), 0, F(2, 2), -F(1, 2), F(0, 1), -F(0, 0), 0, d_sigma(0, 2),
            -F(2, 2), 0, F(0, 2), F(1, 1), -F(1, 0), 0, d_sigma(1, 2), F(1, 2), -F(0, 2), 0, F(2, 1), -F(2, 0), 0,
            d_sigma(2, 2);

        size_t num_residuals = 0;
        for (size_t k = 0; k < x1.size(); ++k) {
            const double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r * r);
            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << x1[k](0) * x2[k](0), x1[k](0) * x2[k](1), x1[k](0), x1[k](1) * x2[k](0), x1[k](1) * x2[k](1),
                x1[k](1), x2[k](0), x2[k](1), 1.0;
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(2) * x1[k](0) + J_C(0) * x2[k](0));
            dF(1) -= s * (J_C(3) * x1[k](0) + J_C(0) * x2[k](1));
            dF(2) -= s * (J_C(0));
            dF(3) -= s * (J_C(2) * x1[k](1) + J_C(1) * x2[k](0));
            dF(4) -= s * (J_C(3) * x1[k](1) + J_C(1) * x2[k](1));
            dF(5) -= s * (J_C(1));
            dF(6) -= s * (J_C(2));
            dF(7) -= s * (J_C(3));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 7> J = dF * dF_dparams;

            // Accumulate into JtJ and Jtr
            Jtr += weight * C * inv_nJ_C * J.transpose();
            for (size_t i = 0; i < 7; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J(i) * J(j));
                }
            }
        }
        return num_residuals;
    }

    FactorizedFundamentalMatrix step(Eigen::Matrix<double, 7, 1> dp, const FactorizedFundamentalMatrix &F) const {
        FactorizedFundamentalMatrix F_new;
        F_new.qU = quat_step_pre(F.qU, dp.block<3, 1>(0, 0));
        F_new.qV = quat_step_pre(F.qV, dp.block<3, 1>(3, 0));
        F_new.sigma = F.sigma + dp(6);
        return F_new;
    }
    typedef FactorizedFundamentalMatrix param_t;
    static constexpr size_t num_params = 7;

  private:
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

// Non-linear refinement of symmetric transfer error |x2 - pi(H*x1)|^2 + |x1 - pi(inv(H)*x2)|^2
// Code is Based on the original single-side transfer error by Viktor Larsson. Implementations of other
// parameterizations (different affine patches, linear least squares, SVD as in Bartoli/Sturm, etc) can be found at
// https://github.com/vlarsson/homopt
// Use adjugate of H to formulate inv(H) since the transfer error is independent of the scale.
// Consider H(2,2) as a constant (not necessary to be 1), we only update the first 8 elements of H.
// Author: Yaqing Ding
template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class HomographyJacobianAccumulator {
  public:
    HomographyJacobianAccumulator(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                  const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), loss_fn(l), weights(w) {}

    double residual(const Eigen::Matrix3d &H) const {
        double cost = 0.0;

        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        const Eigen::Matrix3d G = adjugate(H);
        const double G0_0 = G(0, 0), G0_1 = G(0, 1), G0_2 = G(0, 2);
        const double G1_0 = G(1, 0), G1_1 = G(1, 1), G1_2 = G(1, 2);
        const double G2_0 = G(2, 0), G2_1 = G(2, 1), G2_2 = G(2, 2);

        for (size_t k = 0; k < x1.size(); ++k) {
            // Forward error: |x2 - pi(H*x1)|^2
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double r0 = Hx1_0 * inv_Hx1_2 - x2_0;
            const double r1 = Hx1_1 * inv_Hx1_2 - x2_1;
            const double r2 = r0 * r0 + r1 * r1;

            // Backward error: |x1 - pi(G*x2)|^2
            const double Gx2_0 = G0_0 * x2_0 + G0_1 * x2_1 + G0_2;
            const double Gx2_1 = G1_0 * x2_0 + G1_1 * x2_1 + G1_2;
            const double inv_Gx2_2 = 1.0 / (G2_0 * x2_0 + G2_1 * x2_1 + G2_2);

            const double s0 = Gx2_0 * inv_Gx2_2 - x1_0;
            const double s1 = Gx2_1 * inv_Gx2_2 - x1_1;
            const double s2 = s0 * s0 + s1 * s1;

            cost += weights[k] * (loss_fn.loss(r2) + loss_fn.loss(s2));
        }
        return cost;
    }

    size_t accumulate(const Eigen::Matrix3d &H, Eigen::Matrix<double, 8, 8> &JtJ, Eigen::Matrix<double, 8, 1> &Jtr) {
        Eigen::Matrix<double, 2, 8> dH;

        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        const Eigen::Matrix3d G = adjugate(H);
        const double G0_0 = G(0, 0), G0_1 = G(0, 1), G0_2 = G(0, 2);
        const double G1_0 = G(1, 0), G1_1 = G(1, 1), G1_2 = G(1, 2);
        const double G2_0 = G(2, 0), G2_1 = G(2, 1), G2_2 = G(2, 2);

        size_t num_residuals = 0;

        for (size_t k = 0; k < x1.size(); ++k) {
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            // Forward error
            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double z0 = Hx1_0 * inv_Hx1_2;
            const double z1 = Hx1_1 * inv_Hx1_2;

            const double r0 = z0 - x2_0;
            const double r1 = z1 - x2_1;
            const double r2 = r0 * r0 + r1 * r1;

            // Compute weight from robust loss function (used in the IRLS)
            const double weight = weights[k] * loss_fn.weight(r2);
            if (weight != 0.0) {
                dH << x1_0, 0.0, -x1_0 * z0, x1_1, 0.0, -x1_1 * z0, 1.0, 0.0, // -z0,
                    0.0, x1_0, -x1_0 * z1, 0.0, x1_1, -x1_1 * z1, 0.0, 1.0;   // -z1,
                dH = dH * inv_Hx1_2;

                // accumulate into JtJ and Jtr
                Jtr += dH.transpose() * (weight * Eigen::Vector2d(r0, r1));
                for (size_t i = 0; i < 8; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        JtJ(i, j) += weight * dH.col(i).dot(dH.col(j));
                    }
                }

                num_residuals++;
            }

            const double Gx2_0 = G0_0 * x2_0 + G0_1 * x2_1 + G0_2;
            const double Gx2_1 = G1_0 * x2_0 + G1_1 * x2_1 + G1_2;
            const double inv_Gx2_2 = 1.0 / (G2_0 * x2_0 + G2_1 * x2_1 + G2_2);

            const double y0 = Gx2_0 * inv_Gx2_2;
            const double y1 = Gx2_1 * inv_Gx2_2;

            const double s0 = y0 - x1_0;
            const double s1 = y1 - x1_1;
            const double s2 = s0 * s0 + s1 * s1;

            const double y0x2_1 = y0 * x2_1;
            const double y0x2_0 = y0 * x2_0;
            const double y1x2_1 = y1 * x2_1;
            const double y1x2_0 = y1 * x2_0;

            // Compute weight from robust loss function (used in the IRLS)
            const double weightg = weights[k] * loss_fn.weight(s2);
            if (weightg != 0.0) {

                Eigen::Matrix<double, 2, 8> dH_backward;

                dH_backward << H2_1 * y0x2_1 - H1_1 * y0, H0_1 * y0 - H2_1 * y0x2_0, H1_1 * y0x2_0 - H0_1 * y0x2_1,
                    H1_2 - H2_2 * x2_1 + H1_0 * y0 - H2_0 * y0x2_1, H2_2 * x2_0 - H0_2 - H0_0 * y0 + H2_0 * y0x2_0,
                    H0_2 * x2_1 - H1_2 * x2_0 + H0_0 * y0x2_1 - H1_0 * y0x2_0, H2_1 * x2_1 - H1_1, H0_1 - H2_1 * x2_0,
                    H2_2 * x2_1 - H1_2 - H1_1 * y1 + H2_1 * y1x2_1, H0_2 - H2_2 * x2_0 + H0_1 * y1 - H2_1 * y1x2_0,
                    H1_2 * x2_0 - H0_2 * x2_1 - H0_1 * y1x2_1 + H1_1 * y1x2_0, H1_0 * y1 - H2_0 * y1x2_1,
                    H2_0 * y1x2_0 - H0_0 * y1, H0_0 * y1x2_1 - H1_0 * y1x2_0, H1_0 - H2_0 * x2_1, H2_0 * x2_0 - H0_0;

                dH_backward = dH_backward * inv_Gx2_2;
                // Accumulate backward error
                Jtr += dH_backward.transpose() * (weightg * Eigen::Vector2d(s0, s1));
                for (size_t i = 0; i < 8; ++i) {
                    for (size_t j = 0; j <= i; ++j) {
                        JtJ(i, j) += weightg * dH_backward.col(i).dot(dH_backward.col(j));
                    }
                }

                num_residuals++;
            }
        }

        return num_residuals;
    }

    Eigen::Matrix3d step(Eigen::Matrix<double, 8, 1> dp, const Eigen::Matrix3d &H) const {
        Eigen::Matrix3d H_new = H;
        Eigen::Map<Eigen::Matrix<double, 8, 1>>(H_new.data()) += dp;
        return H_new;
    }
    typedef Eigen::Matrix3d param_t;
    static constexpr size_t num_params = 8;

  private:
    Eigen::Matrix3d adjugate(const Eigen::Matrix3d &H) const {
        Eigen::Matrix3d adj;
        adj(0, 0) = H(1, 1) * H(2, 2) - H(1, 2) * H(2, 1);
        adj(0, 1) = H(0, 2) * H(2, 1) - H(0, 1) * H(2, 2);
        adj(0, 2) = H(0, 1) * H(1, 2) - H(0, 2) * H(1, 1);

        adj(1, 0) = H(1, 2) * H(2, 0) - H(1, 0) * H(2, 2);
        adj(1, 1) = H(0, 0) * H(2, 2) - H(0, 2) * H(2, 0);
        adj(1, 2) = H(0, 2) * H(1, 0) - H(0, 0) * H(1, 2);

        adj(2, 0) = H(1, 0) * H(2, 1) - H(1, 1) * H(2, 0);
        adj(2, 1) = H(0, 1) * H(2, 0) - H(0, 0) * H(2, 1);
        adj(2, 2) = H(0, 0) * H(1, 1) - H(0, 1) * H(1, 0);
        return adj;
    }

    const std::vector<Point2D> &x1, &x2;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

template <typename LossFunction, typename ResidualWeightVector = UniformWeightVector>
class Radial1DJacobianAccumulator {
  public:
    Radial1DJacobianAccumulator(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                const LossFunction &l, const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), loss_fn(l), weights(w) {}

    double residual(const CameraPose &pose) const {
        double cost = 0.0;
        Eigen::Matrix3d R = pose.R();
        for (size_t k = 0; k < x.size(); ++k) {
            Eigen::Vector2d z = (R * X[k] + pose.t).template topRows<2>().normalized();
            double alpha = z.dot(x[k]);
            // This assumes points will not cross the half-space during optimization
            if (alpha < 0)
                continue;
            double r2 = (alpha * z - x[k]).squaredNorm();
            cost += weights[k] * loss_fn.loss(r2);
        }

        return cost;
    }

    size_t accumulate(const CameraPose &pose, Eigen::Matrix<double, 5, 5> &JtJ,
                      Eigen::Matrix<double, 5, 1> &Jtr) const {
        Eigen::Matrix3d R = pose.R();
        size_t num_residuals = 0;
        for (size_t k = 0; k < x.size(); ++k) {
            Eigen::Vector3d RX = R * X[k];
            const Eigen::Vector2d z = (RX + pose.t).topRows<2>();

            const double n_z = z.norm();
            const Eigen::Vector2d zh = z / n_z;
            const double alpha = zh.dot(x[k]);
            // This assumes points will not cross the half-space during optimization
            if (alpha < 0)
                continue;

            // Setup residual
            Eigen::Vector2d r = alpha * zh - x[k];
            const double r_squared = r.squaredNorm();
            const double weight = weights[k] * loss_fn.weight(r_squared);

            if (weight == 0.0) {
                continue;
            }
            num_residuals++;

            // differentiate residual with respect to z
            Eigen::Matrix2d dr_dz = (zh * x[k].transpose() + alpha * Eigen::Matrix2d::Identity()) *
                                    (Eigen::Matrix2d::Identity() - zh * zh.transpose()) / n_z;

            Eigen::Matrix<double, 2, 5> dz;
            dz << 0.0, RX(2), -RX(1), 1.0, 0.0, -RX(2), 0.0, RX(0), 0.0, 1.0;

            Eigen::Matrix<double, 2, 5> J = dr_dz * dz;

            // Accumulate into JtJ and Jtr
            Jtr += weight * J.transpose() * r;
            for (size_t i = 0; i < 5; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    JtJ(i, j) += weight * (J.col(i).dot(J.col(j)));
                }
            }
        }
        return num_residuals;
    }

    CameraPose step(Eigen::Matrix<double, 5, 1> dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_pre(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t(0) = pose.t(0) + dp(3);
        pose_new.t(1) = pose.t(1) + dp(4);
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 5;

  private:
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const LossFunction &loss_fn;
    const ResidualWeightVector &weights;
};

} // namespace poselib

#endif