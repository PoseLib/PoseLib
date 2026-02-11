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

#ifndef POSELIB_MONODEPTH_RELPOSE_H_
#define POSELIB_MONODEPTH_RELPOSE_H_

#include "../../camera_pose.h"
#include "../../misc/essential.h"
#include "../../types.h"
#include "optim_utils.h"
#include "refiner_base.h"

namespace poselib {

// Helper: compute Jacobians of vec(E) w.r.t. 3 translation parameters (not tangent basis)
inline void deriv_essential_wrt_translation(const Eigen::Matrix3d &R, Eigen::Matrix<double, 9, 3> &dt) {
    // Each column is vec(skew(e_k) * R) for k-th standard basis vector
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
}

// Helper: compute Jacobians of vec(E) w.r.t. rotation parameters
inline void deriv_essential_wrt_rotation(const Eigen::Matrix3d &E, Eigen::Matrix<double, 9, 3> &dR) {
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
}

// Helper: compute Sampson Jacobian w.r.t. vec(E) or vec(F) for pinhole points
// Returns the Sampson residual r and the 1x9 Jacobian dF
inline double compute_sampson_jacobian(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2,
                                       const Eigen::Matrix3d &F, Eigen::Matrix<double, 1, 9> &dF) {
    const Eigen::Vector3d x1h = x1.homogeneous();
    const Eigen::Vector3d x2h = x2.homogeneous();

    double C = x2h.dot(F * x1h);

    // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
    Eigen::Vector4d J_C;
    J_C << F.block<3, 2>(0, 0).transpose() * x2h, F.block<2, 3>(0, 0) * x1h;
    const double nJ_C = J_C.norm();
    const double inv_nJ_C = 1.0 / nJ_C;
    const double r = C * inv_nJ_C;

    // Compute Jacobian of Sampson error w.r.t the matrix (3x3)
    dF << x1(0) * x2(0), x1(0) * x2(1), x1(0), x1(1) * x2(0), x1(1) * x2(1),
        x1(1), x2(0), x2(1), 1.0;
    const double s = C * inv_nJ_C * inv_nJ_C;
    dF(0) -= s * (J_C(2) * x1(0) + J_C(0) * x2(0));
    dF(1) -= s * (J_C(3) * x1(0) + J_C(0) * x2(1));
    dF(2) -= s * (J_C(0));
    dF(3) -= s * (J_C(2) * x1(1) + J_C(1) * x2(0));
    dF(4) -= s * (J_C(3) * x1(1) + J_C(1) * x2(1));
    dF(5) -= s * (J_C(1));
    dF(6) -= s * (J_C(2));
    dF(7) -= s * (J_C(3));
    dF *= inv_nJ_C;

    return r;
}

// Helper: compute Sampson residual for a single correspondence
inline double compute_sampson_residual(const Eigen::Vector2d &x1, const Eigen::Vector2d &x2,
                                       const Eigen::Matrix3d &F) {
    const Eigen::Vector3d x1h = x1.homogeneous();
    const Eigen::Vector3d x2h = x2.homogeneous();
    double C = x2h.dot(F * x1h);
    double nJc_sq = (F.block<2, 3>(0, 0) * x1h).squaredNorm() +
                    (F.block<3, 2>(0, 0).transpose() * x2h).squaredNorm();
    return C / std::sqrt(nJc_sq);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Calibrated monodepth relative pose refinement
// Combines Sampson error + symmetric reprojection error using monodepth estimates.
// Parameters: [rotation(3), translation(3), scale(1)] = 7
//   or with shift: [rotation(3), translation(3), scale(1), shift1(1), shift2(1)] = 9
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class MonoDepthRelPoseRefiner : public RefinerBase<MonoDepthTwoViewGeometry, Accumulator> {
  public:
    MonoDepthRelPoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                            const std::vector<double> &d1, const std::vector<double> &d2, const double scale_reproj,
                            const double weight_sampson, const bool refine_shift,
                            const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), d1(d1), d2(d2), scale_reproj(scale_reproj),
          weight_sampson(weight_sampson), refine_shift(refine_shift), weights(w) {
        this->num_params = refine_shift ? 9 : 7;
    }

    double compute_residual(Accumulator &acc, const MonoDepthTwoViewGeometry &geometry) {
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        const Eigen::Matrix3d R = geometry.pose.R();
        const Eigen::Vector3d &t = geometry.pose.t;
        const double sr = std::sqrt(scale_reproj);

        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);

        for (size_t i = 0; i < x1.size(); ++i) {
            if (weight_sampson > 0.0) {
                const double r = compute_sampson_residual(x1[i], x2[i], E);
                acc.add_residual(r, weight_sampson * weights[i]);
            }

            if (scale_reproj > 0.0) {
                const Eigen::Vector3d Z1 = R * ((d1[i] + shift_1) * x1[i].homogeneous().eval()) + t;
                const Eigen::Vector3d Z2 =
                    R.transpose() * (scale * (d2[i] + shift_2) * x2[i].homogeneous().eval() - t);

                if (Z1(2) > 0) {
                    const double inv_z = 1.0 / Z1(2);
                    Eigen::Vector2d res;
                    res << Z1(0) * inv_z - x2[i](0), Z1(1) * inv_z - x2[i](1);
                    res *= sr;
                    acc.add_residual(res, weights[i]);
                }

                if (Z2(2) > 0) {
                    const double inv_z = 1.0 / Z2(2);
                    Eigen::Vector2d res;
                    res << Z2(0) * inv_z - x1[i](0), Z2(1) * inv_z - x1[i](1);
                    res *= sr;
                    acc.add_residual(res, weights[i]);
                }
            }
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const MonoDepthTwoViewGeometry &geometry) {
        const Eigen::Matrix3d R = geometry.pose.R();
        const Eigen::Matrix3d Rt = R.transpose();
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        const int np = this->num_params;
        const double sr = std::sqrt(scale_reproj);

        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);

        // Jacobians of vec(E) w.r.t. rotation and translation
        Eigen::Matrix<double, 9, 3> dR, dt;
        deriv_essential_wrt_rotation(E, dR);
        deriv_essential_wrt_translation(R, dt);

        Eigen::Matrix<double, 2, 3> Jproj;
        Jproj.setZero();

        for (size_t i = 0; i < x1.size(); ++i) {
            // Reprojection error
            if (scale_reproj > 0.0) {
                const Eigen::Vector3d X1o = x1[i].homogeneous();
                const Eigen::Vector3d X1i = (d1[i] + shift_1) * X1o;
                const Eigen::Vector3d Z1 = R * X1i + geometry.pose.t;

                const Eigen::Vector3d X2o = x2[i].homogeneous();
                const Eigen::Vector3d X2s = (d2[i] + shift_2) * X2o;
                const Eigen::Vector3d X2i = scale * X2s;
                const Eigen::Vector3d Z2 = Rt * (X2i - geometry.pose.t);

                // Forward reprojection (cam1 -> cam2)
                if (Z1(2) > 0) {
                    const double inv_z = 1.0 / Z1(2);
                    Eigen::Vector2d res;
                    res << Z1(0) * inv_z - x2[i](0), Z1(1) * inv_z - x2[i](1);

                    Jproj(0, 0) = inv_z;
                    Jproj(1, 1) = inv_z;
                    Jproj(0, 2) = -Z1(0) * inv_z * inv_z;
                    Jproj(1, 2) = -Z1(1) * inv_z * inv_z;

                    Eigen::Matrix<double, 2, Eigen::Dynamic> J(2, np);
                    // Jacobian w.r.t. rotation
                    Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
                    J.col(0) = -X1i(2) * dZ.col(1) + X1i(1) * dZ.col(2);
                    J.col(1) = X1i(2) * dZ.col(0) - X1i(0) * dZ.col(2);
                    J.col(2) = -X1i(1) * dZ.col(0) + X1i(0) * dZ.col(1);
                    // Jacobian w.r.t. translation
                    J.block<2, 3>(0, 3) = Jproj;
                    // Jacobian w.r.t. scale (forward doesn't depend on scale)
                    J.col(6).setZero();

                    if (refine_shift) {
                        // Jacobian w.r.t. shift1
                        J.col(7) = Jproj * R * X1o;
                        // Jacobian w.r.t. shift2
                        J.col(8).setZero();
                    }

                    res *= sr;
                    J *= sr;
                    acc.add_jacobian(res, J, weights[i]);
                }

                // Backward reprojection (cam2 -> cam1)
                if (Z2(2) > 0) {
                    const double inv_z = 1.0 / Z2(2);
                    Eigen::Vector2d res;
                    res << Z2(0) * inv_z - x1[i](0), Z2(1) * inv_z - x1[i](1);

                    Jproj(0, 0) = inv_z;
                    Jproj(1, 1) = inv_z;
                    Jproj(0, 2) = -Z2(0) * inv_z * inv_z;
                    Jproj(1, 2) = -Z2(1) * inv_z * inv_z;

                    Eigen::Matrix<double, 2, Eigen::Dynamic> J(2, np);
                    // Jacobian w.r.t. rotation
                    Eigen::Vector3d X2t = X2i - geometry.pose.t;
                    Eigen::Matrix3d dZdr;
                    dZdr.diagonal().setZero();
                    dZdr(1, 0) = X2t.dot(R.col(2));
                    dZdr(2, 0) = -X2t.dot(R.col(1));
                    dZdr(0, 1) = -X2t.dot(R.col(2));
                    dZdr(2, 1) = X2t.dot(R.col(0));
                    dZdr(0, 2) = X2t.dot(R.col(1));
                    dZdr(1, 2) = -X2t.dot(R.col(0));
                    J.block<2, 3>(0, 0) = Jproj * dZdr;
                    // Jacobian w.r.t. translation
                    J.block<2, 3>(0, 3) = -Jproj * Rt;
                    // Jacobian w.r.t. scale
                    J.col(6) = Jproj * Rt * X2s;

                    if (refine_shift) {
                        // Jacobian w.r.t. shift1
                        J.col(7).setZero();
                        // Jacobian w.r.t. shift2
                        J.col(8) = scale * Jproj * Rt * X2o;
                    }

                    res *= sr;
                    J *= sr;
                    acc.add_jacobian(res, J, weights[i]);
                }
            }

            // Sampson error
            if (weight_sampson > 0.0) {
                Eigen::Matrix<double, 1, 9> dF;
                const double r = compute_sampson_jacobian(x1[i], x2[i], E, dF);

                Eigen::Matrix<double, 1, Eigen::Dynamic> J_sam(1, np);
                J_sam.setZero();
                J_sam.block<1, 3>(0, 0) = dF * dR;
                J_sam.block<1, 3>(0, 3) = dF * dt;
                // Sampson doesn't depend on scale, shift1, shift2

                acc.add_jacobian(r, J_sam, weight_sampson * weights[i]);
            }
        }
    }

    MonoDepthTwoViewGeometry step(const Eigen::VectorXd &dp, const MonoDepthTwoViewGeometry &geometry) const {
        MonoDepthTwoViewGeometry geometry_new;
        geometry_new.pose.q = quat_step_post(geometry.pose.q, dp.block<3, 1>(0, 0));
        geometry_new.pose.t = geometry.pose.t + dp.block<3, 1>(3, 0);
        geometry_new.scale = geometry.scale + dp(6);
        if (refine_shift) {
            geometry_new.shift1 = geometry.shift1 + dp(7);
            geometry_new.shift2 = geometry.shift2 + dp(8);
        } else {
            geometry_new.shift1 = geometry.shift1;
            geometry_new.shift2 = geometry.shift2;
        }
        return geometry_new;
    }

    typedef MonoDepthTwoViewGeometry param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;
    const double scale_reproj, weight_sampson;
    const bool refine_shift;
    const ResidualWeightVector &weights;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Monodepth relative pose with shared focal length refinement.
// Points are in centered pixel coordinates. Focal is shared by both cameras.
// Parameters: [rotation(3), translation(3), scale(1), focal(1)] = 8
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class MonoDepthSharedFocalRelPoseRefiner : public RefinerBase<MonoDepthImagePair, Accumulator> {
  public:
    MonoDepthSharedFocalRelPoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                       const std::vector<double> &d1, const std::vector<double> &d2,
                                       const double scale_reproj, const double weight_sampson,
                                       const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), d1(d1), d2(d2), scale_reproj(scale_reproj),
          weight_sampson(weight_sampson), weights(w) {
        this->num_params = 8;
    }

    double compute_residual(Accumulator &acc, const MonoDepthImagePair &image_pair) {
        const MonoDepthTwoViewGeometry &geometry = image_pair.geometry;
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        const Eigen::Matrix3d R = geometry.pose.R();
        const Eigen::Vector3d &t = geometry.pose.t;
        const double f = image_pair.camera1.focal();
        const double sr = std::sqrt(scale_reproj);

        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);
        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, f);
        Eigen::Matrix3d F = K_inv * E * K_inv;

        for (size_t i = 0; i < x1.size(); ++i) {
            if (weight_sampson > 0.0) {
                const double r = compute_sampson_residual(x1[i], x2[i], F);
                acc.add_residual(r, weight_sampson * weights[i]);
            }

            if (scale_reproj > 0.0) {
                // Unproject to bearing direction using focal
                const Eigen::Vector3d b1(x1[i](0) / f, x1[i](1) / f, 1.0);
                const Eigen::Vector3d b2(x2[i](0) / f, x2[i](1) / f, 1.0);

                const Eigen::Vector3d Z1 = R * ((d1[i] + shift_1) * b1) + t;
                const Eigen::Vector3d Z2 = Rt_apply(R, scale * (d2[i] + shift_2) * b2, t);

                if (Z1(2) > 0) {
                    const double inv_z = 1.0 / Z1(2);
                    Eigen::Vector2d res;
                    res << f * Z1(0) * inv_z - x2[i](0), f * Z1(1) * inv_z - x2[i](1);
                    res *= sr;
                    acc.add_residual(res, weights[i]);
                }

                if (Z2(2) > 0) {
                    const double inv_z = 1.0 / Z2(2);
                    Eigen::Vector2d res;
                    res << f * Z2(0) * inv_z - x1[i](0), f * Z2(1) * inv_z - x1[i](1);
                    res *= sr;
                    acc.add_residual(res, weights[i]);
                }
            }
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const MonoDepthImagePair &image_pair) {
        const MonoDepthTwoViewGeometry &geometry = image_pair.geometry;
        const Eigen::Matrix3d R = geometry.pose.R();
        const Eigen::Matrix3d Rt = R.transpose();
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        const double f = image_pair.camera1.focal();
        const double sr = std::sqrt(scale_reproj);

        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);
        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, f);
        Eigen::Matrix3d F = K_inv * E * K_inv;

        // Jacobians of vec(E) w.r.t. rotation and translation
        Eigen::Matrix<double, 9, 3> dR_E, dt_E;
        deriv_essential_wrt_rotation(E, dR_E);
        deriv_essential_wrt_translation(R, dt_E);

        // Scale by K_inv for F = K_inv * E * K_inv
        // vec(F) = diag(1,1,f,1,1,f,f,f,f^2) * vec(E)
        Eigen::Matrix<double, 9, 3> dR_F = dR_E, dt_F = dt_E;
        dR_F.row(2) *= f;
        dR_F.row(5) *= f;
        dR_F.row(6) *= f;
        dR_F.row(7) *= f;
        dR_F.row(8) *= f * f;
        dt_F.row(2) *= f;
        dt_F.row(5) *= f;
        dt_F.row(6) *= f;
        dt_F.row(7) *= f;
        dt_F.row(8) *= f * f;

        // Jacobian of vec(F) w.r.t. focal
        Eigen::Matrix<double, 9, 1> df_F;
        df_F << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), E(0, 2), E(1, 2), 2 * E(2, 2) * f;

        Eigen::Matrix<double, 2, 3> Jproj;
        Jproj.setZero();

        for (size_t i = 0; i < x1.size(); ++i) {
            // Reprojection error
            if (scale_reproj > 0.0) {
                const Eigen::Vector3d b1(x1[i](0) / f, x1[i](1) / f, 1.0);
                const Eigen::Vector3d b2(x2[i](0) / f, x2[i](1) / f, 1.0);
                const Eigen::Vector3d X1i = (d1[i] + shift_1) * b1;
                const Eigen::Vector3d Z1 = R * X1i + geometry.pose.t;
                const Eigen::Vector3d X2s = (d2[i] + shift_2) * b2;
                const Eigen::Vector3d X2i = scale * X2s;
                const Eigen::Vector3d Z2 = Rt * (X2i - geometry.pose.t);

                // Forward reprojection (cam1 -> cam2)
                if (Z1(2) > 0) {
                    const double inv_z = 1.0 / Z1(2);
                    const Eigen::Vector2d xp_cal(Z1(0) * inv_z, Z1(1) * inv_z);
                    Eigen::Vector2d res = f * xp_cal - x2[i].head<2>();

                    // Projection Jacobian (includes focal: xp = f * Z/Z(2))
                    Jproj(0, 0) = f * inv_z;
                    Jproj(1, 1) = f * inv_z;
                    Jproj(0, 2) = -f * Z1(0) * inv_z * inv_z;
                    Jproj(1, 2) = -f * Z1(1) * inv_z * inv_z;

                    Eigen::Matrix<double, 2, 8> J;
                    // Jacobian w.r.t. rotation
                    Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
                    J.col(0) = -X1i(2) * dZ.col(1) + X1i(1) * dZ.col(2);
                    J.col(1) = X1i(2) * dZ.col(0) - X1i(0) * dZ.col(2);
                    J.col(2) = -X1i(1) * dZ.col(0) + X1i(0) * dZ.col(1);
                    // Jacobian w.r.t. translation
                    J.block<2, 3>(0, 3) = Jproj;
                    // Jacobian w.r.t. scale (forward doesn't depend on scale)
                    J.col(6).setZero();
                    // Jacobian w.r.t. focal
                    // xp = f * Z/Z(2), Z depends on f through b1 = [x1(0)/f, x1(1)/f, 1]
                    Eigen::Vector3d dX1_df = (d1[i] + shift_1) * Eigen::Vector3d(-x1[i](0) / (f * f), -x1[i](1) / (f * f), 0.0);
                    Eigen::Vector3d dZ1_df = R * dX1_df;
                    J.col(7) = xp_cal + Jproj * dZ1_df;

                    res *= sr;
                    J *= sr;
                    acc.add_jacobian(res, J, weights[i]);
                }

                // Backward reprojection (cam2 -> cam1)
                if (Z2(2) > 0) {
                    const double inv_z = 1.0 / Z2(2);
                    const Eigen::Vector2d xp_cal(Z2(0) * inv_z, Z2(1) * inv_z);
                    Eigen::Vector2d res = f * xp_cal - x1[i].head<2>();

                    Jproj(0, 0) = f * inv_z;
                    Jproj(1, 1) = f * inv_z;
                    Jproj(0, 2) = -f * Z2(0) * inv_z * inv_z;
                    Jproj(1, 2) = -f * Z2(1) * inv_z * inv_z;

                    Eigen::Matrix<double, 2, 8> J;
                    // Jacobian w.r.t. rotation
                    Eigen::Vector3d X2t = X2i - geometry.pose.t;
                    Eigen::Matrix3d dZdr;
                    dZdr.diagonal().setZero();
                    dZdr(1, 0) = X2t.dot(R.col(2));
                    dZdr(2, 0) = -X2t.dot(R.col(1));
                    dZdr(0, 1) = -X2t.dot(R.col(2));
                    dZdr(2, 1) = X2t.dot(R.col(0));
                    dZdr(0, 2) = X2t.dot(R.col(1));
                    dZdr(1, 2) = -X2t.dot(R.col(0));
                    J.block<2, 3>(0, 0) = Jproj * dZdr;
                    // Jacobian w.r.t. translation
                    J.block<2, 3>(0, 3) = -Jproj * Rt;
                    // Jacobian w.r.t. scale
                    J.col(6) = Jproj * Rt * X2s;
                    // Jacobian w.r.t. focal
                    Eigen::Vector3d dX2_df = scale * (d2[i] + shift_2) * Eigen::Vector3d(-x2[i](0) / (f * f), -x2[i](1) / (f * f), 0.0);
                    Eigen::Vector3d dZ2_df = Rt * dX2_df;
                    J.col(7) = xp_cal + Jproj * dZ2_df;

                    res *= sr;
                    J *= sr;
                    acc.add_jacobian(res, J, weights[i]);
                }
            }

            // Sampson error (uses F = K_inv * E * K_inv)
            if (weight_sampson > 0.0) {
                Eigen::Matrix<double, 1, 9> dF_mat;
                const double r = compute_sampson_jacobian(x1[i], x2[i], F, dF_mat);

                Eigen::Matrix<double, 1, 8> J_sam;
                J_sam.block<1, 3>(0, 0) = dF_mat * dR_F;
                J_sam.block<1, 3>(0, 3) = dF_mat * dt_F;
                J_sam(0, 6) = 0.0; // Sampson doesn't depend on scale
                J_sam(0, 7) = (dF_mat * df_F)(0, 0);

                acc.add_jacobian(r, J_sam, weight_sampson * weights[i]);
            }
        }
    }

    MonoDepthImagePair step(const Eigen::VectorXd &dp, const MonoDepthImagePair &image_pair) const {
        MonoDepthImagePair result;
        result.geometry.pose.q = quat_step_post(image_pair.geometry.pose.q, dp.block<3, 1>(0, 0));
        result.geometry.pose.t = image_pair.geometry.pose.t + dp.block<3, 1>(3, 0);
        result.geometry.scale = image_pair.geometry.scale + dp(6);
        result.geometry.shift1 = image_pair.geometry.shift1;
        result.geometry.shift2 = image_pair.geometry.shift2;
        double new_focal = image_pair.camera1.focal() + dp(7);
        result.camera1 = Camera("SIMPLE_PINHOLE", {new_focal, 0, 0}, -1, -1);
        result.camera2 = result.camera1;
        return result;
    }

    typedef MonoDepthImagePair param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;
    const double scale_reproj, weight_sampson;
    const ResidualWeightVector &weights;

  private:
    // Helper: R^T * (X - t)
    static Eigen::Vector3d Rt_apply(const Eigen::Matrix3d &R, const Eigen::Vector3d &X, const Eigen::Vector3d &t) {
        return R.transpose() * (X - t);
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// Monodepth relative pose with two different focal lengths.
// Points are in centered pixel coordinates. Each camera has its own focal.
// Parameters: [rotation(3), translation(3), scale(1), focal1(1), focal2(1)] = 9
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class MonoDepthVaryingFocalRelPoseRefiner : public RefinerBase<MonoDepthImagePair, Accumulator> {
  public:
    MonoDepthVaryingFocalRelPoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                        const std::vector<double> &d1, const std::vector<double> &d2,
                                        const double scale_reproj, const double weight_sampson,
                                        const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), d1(d1), d2(d2), scale_reproj(scale_reproj),
          weight_sampson(weight_sampson), weights(w) {
        this->num_params = 9;
    }

    double compute_residual(Accumulator &acc, const MonoDepthImagePair &image_pair) {
        const MonoDepthTwoViewGeometry &geometry = image_pair.geometry;
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        const Eigen::Matrix3d R = geometry.pose.R();
        const Eigen::Vector3d &t = geometry.pose.t;
        const double f1 = image_pair.camera1.focal();
        const double f2 = image_pair.camera2.focal();
        const double sr = std::sqrt(scale_reproj);

        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);
        Eigen::DiagonalMatrix<double, 3> K1_inv(1.0, 1.0, f1);
        Eigen::DiagonalMatrix<double, 3> K2_inv(1.0, 1.0, f2);
        Eigen::Matrix3d F = K2_inv * E * K1_inv;

        for (size_t i = 0; i < x1.size(); ++i) {
            if (weight_sampson > 0.0) {
                const double r = compute_sampson_residual(x1[i], x2[i], F);
                acc.add_residual(r, weight_sampson * weights[i]);
            }

            if (scale_reproj > 0.0) {
                const Eigen::Vector3d b1(x1[i](0) / f1, x1[i](1) / f1, 1.0);
                const Eigen::Vector3d b2(x2[i](0) / f2, x2[i](1) / f2, 1.0);

                const Eigen::Vector3d Z1 = R * ((d1[i] + shift_1) * b1) + t;
                const Eigen::Vector3d Z2 = R.transpose() * (scale * (d2[i] + shift_2) * b2 - t);

                if (Z1(2) > 0) {
                    const double inv_z = 1.0 / Z1(2);
                    Eigen::Vector2d res;
                    res << f2 * Z1(0) * inv_z - x2[i](0), f2 * Z1(1) * inv_z - x2[i](1);
                    res *= sr;
                    acc.add_residual(res, weights[i]);
                }

                if (Z2(2) > 0) {
                    const double inv_z = 1.0 / Z2(2);
                    Eigen::Vector2d res;
                    res << f1 * Z2(0) * inv_z - x1[i](0), f1 * Z2(1) * inv_z - x1[i](1);
                    res *= sr;
                    acc.add_residual(res, weights[i]);
                }
            }
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const MonoDepthImagePair &image_pair) {
        const MonoDepthTwoViewGeometry &geometry = image_pair.geometry;
        const Eigen::Matrix3d R = geometry.pose.R();
        const Eigen::Matrix3d Rt = R.transpose();
        const double scale = geometry.scale;
        const double shift_1 = geometry.shift1;
        const double shift_2 = geometry.shift2;
        const double f1 = image_pair.camera1.focal();
        const double f2 = image_pair.camera2.focal();
        const double sr = std::sqrt(scale_reproj);

        Eigen::Matrix3d E;
        essential_from_motion(geometry.pose, &E);
        Eigen::DiagonalMatrix<double, 3> K1_inv(1.0, 1.0, f1);
        Eigen::DiagonalMatrix<double, 3> K2_inv(1.0, 1.0, f2);
        Eigen::Matrix3d F = K2_inv * E * K1_inv;

        // Jacobians of vec(E) w.r.t. rotation and translation
        Eigen::Matrix<double, 9, 3> dR_E, dt_E;
        deriv_essential_wrt_rotation(E, dR_E);
        deriv_essential_wrt_translation(R, dt_E);

        // Scale by K1_inv, K2_inv for F = K2_inv * E * K1_inv
        // F(i,j) = K2(i,i) * E(i,j) * K1(j,j)
        // vec(F) scaling: rows 0,1: x1, row 2: xf2, rows 3,4: x1, row 5: xf2, row 6: xf1, row 7: xf1, row 8: xf1*f2
        Eigen::Matrix<double, 9, 3> dR_F = dR_E, dt_F = dt_E;
        dR_F.row(2) *= f2;
        dR_F.row(5) *= f2;
        dR_F.row(6) *= f1;
        dR_F.row(7) *= f1;
        dR_F.row(8) *= f1 * f2;
        dt_F.row(2) *= f2;
        dt_F.row(5) *= f2;
        dt_F.row(6) *= f1;
        dt_F.row(7) *= f1;
        dt_F.row(8) *= f1 * f2;

        // Jacobian of vec(F) w.r.t. f1: d(K2_inv * E * K1_inv)/df1
        Eigen::Matrix<double, 9, 1> df1_F;
        df1_F << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, E(0, 2), E(1, 2), f2 * E(2, 2);
        // Jacobian of vec(F) w.r.t. f2: d(K2_inv * E * K1_inv)/df2
        Eigen::Matrix<double, 9, 1> df2_F;
        df2_F << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), 0.0, 0.0, f1 * E(2, 2);

        Eigen::Matrix<double, 2, 3> Jproj;
        Jproj.setZero();

        for (size_t i = 0; i < x1.size(); ++i) {
            // Reprojection error
            if (scale_reproj > 0.0) {
                const Eigen::Vector3d b1(x1[i](0) / f1, x1[i](1) / f1, 1.0);
                const Eigen::Vector3d b2(x2[i](0) / f2, x2[i](1) / f2, 1.0);
                const Eigen::Vector3d X1i = (d1[i] + shift_1) * b1;
                const Eigen::Vector3d Z1 = R * X1i + geometry.pose.t;
                const Eigen::Vector3d X2s = (d2[i] + shift_2) * b2;
                const Eigen::Vector3d X2i = scale * X2s;
                const Eigen::Vector3d Z2 = Rt * (X2i - geometry.pose.t);

                // Forward reprojection (cam1 -> cam2, project with f2)
                if (Z1(2) > 0) {
                    const double inv_z = 1.0 / Z1(2);
                    const Eigen::Vector2d xp_cal(Z1(0) * inv_z, Z1(1) * inv_z);
                    Eigen::Vector2d res = f2 * xp_cal - x2[i].head<2>();

                    Jproj(0, 0) = f2 * inv_z;
                    Jproj(1, 1) = f2 * inv_z;
                    Jproj(0, 2) = -f2 * Z1(0) * inv_z * inv_z;
                    Jproj(1, 2) = -f2 * Z1(1) * inv_z * inv_z;

                    Eigen::Matrix<double, 2, 9> J;
                    // Jacobian w.r.t. rotation
                    Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
                    J.col(0) = -X1i(2) * dZ.col(1) + X1i(1) * dZ.col(2);
                    J.col(1) = X1i(2) * dZ.col(0) - X1i(0) * dZ.col(2);
                    J.col(2) = -X1i(1) * dZ.col(0) + X1i(0) * dZ.col(1);
                    // Jacobian w.r.t. translation
                    J.block<2, 3>(0, 3) = Jproj;
                    // Jacobian w.r.t. scale
                    J.col(6).setZero();
                    // Jacobian w.r.t. f1 (affects unprojection of x1 only)
                    Eigen::Vector3d dX1_df1 = (d1[i] + shift_1) * Eigen::Vector3d(-x1[i](0) / (f1 * f1), -x1[i](1) / (f1 * f1), 0.0);
                    J.col(7) = Jproj * R * dX1_df1;
                    // Jacobian w.r.t. f2 (affects projection only)
                    J.col(8) = xp_cal;

                    res *= sr;
                    J *= sr;
                    acc.add_jacobian(res, J, weights[i]);
                }

                // Backward reprojection (cam2 -> cam1, project with f1)
                if (Z2(2) > 0) {
                    const double inv_z = 1.0 / Z2(2);
                    const Eigen::Vector2d xp_cal(Z2(0) * inv_z, Z2(1) * inv_z);
                    Eigen::Vector2d res = f1 * xp_cal - x1[i].head<2>();

                    Jproj(0, 0) = f1 * inv_z;
                    Jproj(1, 1) = f1 * inv_z;
                    Jproj(0, 2) = -f1 * Z2(0) * inv_z * inv_z;
                    Jproj(1, 2) = -f1 * Z2(1) * inv_z * inv_z;

                    Eigen::Matrix<double, 2, 9> J;
                    // Jacobian w.r.t. rotation
                    Eigen::Vector3d X2t = X2i - geometry.pose.t;
                    Eigen::Matrix3d dZdr;
                    dZdr.diagonal().setZero();
                    dZdr(1, 0) = X2t.dot(R.col(2));
                    dZdr(2, 0) = -X2t.dot(R.col(1));
                    dZdr(0, 1) = -X2t.dot(R.col(2));
                    dZdr(2, 1) = X2t.dot(R.col(0));
                    dZdr(0, 2) = X2t.dot(R.col(1));
                    dZdr(1, 2) = -X2t.dot(R.col(0));
                    J.block<2, 3>(0, 0) = Jproj * dZdr;
                    // Jacobian w.r.t. translation
                    J.block<2, 3>(0, 3) = -Jproj * Rt;
                    // Jacobian w.r.t. scale
                    J.col(6) = Jproj * Rt * X2s;
                    // Jacobian w.r.t. f1 (affects projection only)
                    J.col(7) = xp_cal;
                    // Jacobian w.r.t. f2 (affects unprojection of x2 only)
                    Eigen::Vector3d dX2_df2 = scale * (d2[i] + shift_2) * Eigen::Vector3d(-x2[i](0) / (f2 * f2), -x2[i](1) / (f2 * f2), 0.0);
                    J.col(8) = Jproj * Rt * dX2_df2;

                    res *= sr;
                    J *= sr;
                    acc.add_jacobian(res, J, weights[i]);
                }
            }

            // Sampson error (uses F = K2_inv * E * K1_inv)
            if (weight_sampson > 0.0) {
                Eigen::Matrix<double, 1, 9> dF_mat;
                const double r = compute_sampson_jacobian(x1[i], x2[i], F, dF_mat);

                Eigen::Matrix<double, 1, 9> J_sam;
                J_sam.block<1, 3>(0, 0) = dF_mat * dR_F;
                J_sam.block<1, 3>(0, 3) = dF_mat * dt_F;
                J_sam(0, 6) = 0.0; // Sampson doesn't depend on scale
                J_sam(0, 7) = (dF_mat * df1_F)(0, 0);
                J_sam(0, 8) = (dF_mat * df2_F)(0, 0);

                acc.add_jacobian(r, J_sam, weight_sampson * weights[i]);
            }
        }
    }

    MonoDepthImagePair step(const Eigen::VectorXd &dp, const MonoDepthImagePair &image_pair) const {
        MonoDepthImagePair result;
        result.geometry.pose.q = quat_step_post(image_pair.geometry.pose.q, dp.block<3, 1>(0, 0));
        result.geometry.pose.t = image_pair.geometry.pose.t + dp.block<3, 1>(3, 0);
        result.geometry.scale = image_pair.geometry.scale + dp(6);
        result.geometry.shift1 = image_pair.geometry.shift1;
        result.geometry.shift2 = image_pair.geometry.shift2;
        result.camera1 = Camera("SIMPLE_PINHOLE", {image_pair.camera1.focal() + dp(7), 0, 0}, -1, -1);
        result.camera2 = Camera("SIMPLE_PINHOLE", {image_pair.camera2.focal() + dp(8), 0, 0}, -1, -1);
        return result;
    }

    typedef MonoDepthImagePair param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;
    const double scale_reproj, weight_sampson;
    const ResidualWeightVector &weights;
};

} // namespace poselib

#endif
