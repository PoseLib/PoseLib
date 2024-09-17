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

#ifndef POSELIB_FUNDAMENTAL_H_
#define POSELIB_FUNDAMENTAL_H_

#include "../../types.h"
#include "optim_utils.h"
#include "refiner_base.h"

namespace poselib {

// Minimize Sampson error with pinhole camera model.
// NOTE: IT IS SUPER IMPORTANT TO NORMALIZE (RESCALE) YOUR INPUT!
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class PinholeFundamentalRefiner : public RefinerBase<FactorizedFundamentalMatrix, Accumulator> {
  public:
    PinholeFundamentalRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), weights(w) {
        this->num_params = 7;
    }

    double compute_residual(Accumulator &acc, const FactorizedFundamentalMatrix &FF) {
        Eigen::Matrix3d F = FF.F();

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const FactorizedFundamentalMatrix &FF) {
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

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), F.block<2, 3>(0, 0) * x1[k].homogeneous();
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

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

            // and then w.r.t. the fundamental matrix parameters
            Eigen::Matrix<double, 1, 7> J = dF * dF_dparams;
            acc.add_jacobian(r, J, weights[k]);
        }
    }

    FactorizedFundamentalMatrix step(const Eigen::VectorXd &dp, const FactorizedFundamentalMatrix &F) const {
        FactorizedFundamentalMatrix F_new;
        F_new.qU = quat_step_pre(F.qU, dp.block<3, 1>(0, 0));
        F_new.qV = quat_step_pre(F.qV, dp.block<3, 1>(3, 0));
        F_new.sigma = F.sigma + dp(6);
        return F_new;
    }

    typedef FactorizedFundamentalMatrix param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class RDFundamentalRefiner : public RefinerBase<FactorizedProjectiveImagePair, Accumulator> {
  public:
    RDFundamentalRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                         const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), weights(w) {
        this->num_params = 9;
    }

    double compute_residual(Accumulator &acc, const FactorizedProjectiveImagePair &projective_image_pair) {
        Eigen::Matrix3d F = projective_image_pair.FF.F();

        for (size_t i = 0; i < x1.size(); ++i) {
            Eigen::Matrix<double, 3, 1> xu1, xu2;
            Eigen::Matrix<double, 3, 2> J1, J2;
            projective_image_pair.camera1.unproject_with_jac(x1[i], &xu1, &J1);
            projective_image_pair.camera2.unproject_with_jac(x2[i], &xu2, &J2);

            double num = xu2.transpose() * (F * xu1);

            double den_sq =
                (xu2.transpose() * F * J1).squaredNorm() + (xu1.transpose() * F.transpose() * J2).squaredNorm();
            acc.add_residual(num / std::sqrt(den_sq), weights[i]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const FactorizedProjectiveImagePair &proj_image_pair) {
        FactorizedFundamentalMatrix FF = proj_image_pair.FF;
        // Using F directly from ProjectiveImagePair causes issues with U and V flipping signs in some columns
        const Eigen::Matrix3d F = FF.F();

        // Matrices contain the jacobians of F w.r.t. the factorized fundamental matrix (U,V,sigma)
        const Eigen::Matrix3d U = quat_to_rotmat(FF.qU);
        const Eigen::Matrix3d V = quat_to_rotmat(FF.qV);

        Eigen::Matrix3d d_sigma = U.col(1) * V.col(1).transpose();
        Eigen::Matrix<double, 9, 7> dF_dparams;
        dF_dparams << 0, F(2, 0), -F(1, 0), 0, F(0, 2), -F(0, 1), d_sigma(0, 0), -F(2, 0), 0, F(0, 0), 0, F(1, 2),
            -F(1, 1), d_sigma(1, 0), F(1, 0), -F(0, 0), 0, 0, F(2, 2), -F(2, 1), d_sigma(2, 0), 0, F(2, 1), -F(1, 1),
            -F(0, 2), 0, F(0, 0), d_sigma(0, 1), -F(2, 1), 0, F(0, 1), -F(1, 2), 0, F(1, 0), d_sigma(1, 1), F(1, 1),
            -F(0, 1), 0, -F(2, 2), 0, F(2, 0), d_sigma(2, 1), 0, F(2, 2), -F(1, 2), F(0, 1), -F(0, 0), 0, d_sigma(0, 2),
            -F(2, 2), 0, F(0, 2), F(1, 1), -F(1, 0), 0, d_sigma(1, 2), F(1, 2), -F(0, 2), 0, F(2, 1), -F(2, 0), 0,
            d_sigma(2, 2);

        double k1 = proj_image_pair.camera1.params[4];
        double k2 = proj_image_pair.camera2.params[4];

        for (size_t i = 0; i < x1.size(); ++i) {
            double x1_sq = x1[i].squaredNorm();
            double x2_sq = x2[i].squaredNorm();
            Eigen::Matrix<double, 3, 1> h1, h2;
            Eigen::Matrix<double, 3, 2> J1, J2;
            proj_image_pair.camera1.unproject_with_jac(x1[i], &h1, &J1);
            proj_image_pair.camera2.unproject_with_jac(x2[i], &h2, &J2);

            Eigen::Matrix<double, 3, 1> d1, d2;
            d1 << x1[i](0), x1[i](1), 1 + k1 * x1_sq;
            d2 << x2[i](0), x2[i](1), 1 + k2 * x2_sq;

            const double C = d2.transpose() * (F * d1);

            Eigen::Matrix<double, 4, 1> J_C;
            J_C.block<2, 1>(0, 0) = d2.transpose() * (F * J1);
            J_C.block<2, 1>(2, 0) = d1.transpose() * (F.transpose() * J2);
            const double nJ_C = J_C.norm();

            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;

            const double s = C * inv_nJ_C * inv_nJ_C;

            dF(0) = (J_C(0) * J1(0, 0) * d2(0));
            dF(1) = (J_C(0) * J1(0, 0) * d2(1));
            dF(2) = (J_C(0) * J1(0, 0) * d2(2));
            dF(3) = (J_C(0) * J1(1, 0) * d2(0));
            dF(4) = (J_C(0) * J1(1, 0) * d2(1));
            dF(5) = (J_C(0) * J1(1, 0) * d2(2));
            dF(6) = (J_C(0) * J1(2, 0) * d2(0));
            dF(7) = (J_C(0) * J1(2, 0) * d2(1));
            dF(8) = (J_C(0) * J1(2, 0) * d2(2));

            dF(0) += (J_C(1) * J1(0, 1) * d2(0));
            dF(1) += (J_C(1) * J1(0, 1) * d2(1));
            dF(2) += (J_C(1) * J1(0, 1) * d2(2));
            dF(3) += (J_C(1) * J1(1, 1) * d2(0));
            dF(4) += (J_C(1) * J1(1, 1) * d2(1));
            dF(5) += (J_C(1) * J1(1, 1) * d2(2));
            dF(6) += (J_C(1) * J1(2, 1) * d2(0));
            dF(7) += (J_C(1) * J1(2, 1) * d2(1));
            dF(8) += (J_C(1) * J1(2, 1) * d2(2));

            dF(0) += (J_C(2) * J2(0, 0) * d1(0));
            dF(1) += (J_C(2) * J2(1, 0) * d1(0));
            dF(2) += (J_C(2) * J2(2, 0) * d1(0));
            dF(3) += (J_C(2) * J2(0, 0) * d1(1));
            dF(4) += (J_C(2) * J2(1, 0) * d1(1));
            dF(5) += (J_C(2) * J2(2, 0) * d1(1));
            dF(6) += (J_C(2) * J2(0, 0) * d1(2));
            dF(7) += (J_C(2) * J2(1, 0) * d1(2));
            dF(8) += (J_C(2) * J2(2, 0) * d1(2));

            dF(0) += (J_C(3) * J2(0, 1) * d1(0));
            dF(1) += (J_C(3) * J2(1, 1) * d1(0));
            dF(2) += (J_C(3) * J2(2, 1) * d1(0));
            dF(3) += (J_C(3) * J2(0, 1) * d1(1));
            dF(4) += (J_C(3) * J2(1, 1) * d1(1));
            dF(5) += (J_C(3) * J2(2, 1) * d1(1));
            dF(6) += (J_C(3) * J2(0, 1) * d1(2));
            dF(7) += (J_C(3) * J2(1, 1) * d1(2));
            dF(8) += (J_C(3) * J2(2, 1) * d1(2));

            dF *= -s;

            dF(0) += d1(0) * d2(0);
            dF(1) += d1(0) * d2(1);
            dF(2) += d1(0) * d2(2);
            dF(3) += d1(1) * d2(0);
            dF(4) += d1(1) * d2(1);
            dF(5) += d1(1) * d2(2);
            dF(6) += d1(2) * d2(0);
            dF(7) += d1(2) * d2(1);
            dF(8) += d1(2) * d2(2);

            dF *= inv_nJ_C;

            // dLdk = dCdk * den / (den ** 2) + next
            double dLdk1 = inv_nJ_C * F.col(2).dot(d2) * x1_sq;
            double dLdk2 = inv_nJ_C * F.row(2).dot(d1) * x2_sq;

            double s1 = d1(2) * d1(2) + x1_sq;
            double s2 = d2(2) * d2(2) + x2_sq;
            Eigen::Matrix<double, 3, 2> dJ1dk, dJ2dk;
            double d10_sq = d1(0) * d1(0), d11_sq = d1(1) * d1(1);
            dJ1dk(0, 0) = (3 * d1(2) * x1_sq * (2 * d10_sq * d1(2) * k1 + d10_sq - s1) -
                           2 * s1 * (d10_sq * d1(2) + d10_sq * k1 * x1_sq - d1(2) * x1_sq));
            dJ1dk(0, 1) = d1(0) * d1(1) * (3 * d1(2) * x1_sq * (2 * d1(2) * k1 + 1) - 2 * s1 * (2 * k1 * x1_sq + 1));
            dJ1dk(1, 0) = dJ1dk(0, 1);
            dJ1dk(1, 1) = (3 * d1(2) * x1_sq * (2 * d11_sq * d1(2) * k1 + d11_sq - s1) -
                           2 * s1 * (d11_sq * d1(2) + d11_sq * k1 * x1_sq - d1(2) * x1_sq));
            dJ1dk(2, 0) = d1(0);
            dJ1dk(2, 1) = d1(1);
            dJ1dk.row(2) *= -x1_sq * (3 * d1(2) * (d1(2) - 2) - s1);
            dJ1dk /= std::pow(s1, 2.5);

            double d20_sq = d2(0) * d2(0), d21_sq = d2(1) * d2(1);
            dJ2dk(0, 0) = (3 * d2(2) * x2_sq * (2 * d20_sq * d2(2) * k2 + d20_sq - s2) -
                           2 * s2 * (d20_sq * d2(2) + d20_sq * k2 * x2_sq - d2(2) * x2_sq));
            dJ2dk(0, 1) = d2(0) * d2(1) * (3 * d2(2) * x2_sq * (2 * d2(2) * k2 + 1) - 2 * s2 * (2 * k2 * x2_sq + 1));
            dJ2dk(1, 0) = dJ2dk(0, 1);
            dJ2dk(1, 1) = (3 * d2(2) * x2_sq * (2 * d21_sq * d2(2) * k2 + d21_sq - s2) -
                           2 * s2 * (d21_sq * d2(2) + d21_sq * k2 * x2_sq - d2(2) * x2_sq));
            dJ2dk(2, 0) = d2(0);
            dJ2dk(2, 1) = d2(1);
            dJ2dk.row(2) *= -x2_sq * (3 * d2(2) * (d2(2) - 2) - s2);
            dJ2dk /= std::pow(s2, 2.5);

            Eigen::Matrix<double, 3, 1> dd1dk, dd2dk;
            dd1dk << 0.0, 0.0, x1_sq;
            dd2dk << 0.0, 0.0, x2_sq;

            dLdk1 -= C * inv_nJ_C * inv_nJ_C * inv_nJ_C *
                     (d2.transpose() * F * J1 * (d2.transpose() * F * dJ1dk).transpose() +
                      d1.transpose() * F.transpose() * J2 * (x1_sq * F.col(2).transpose() * J2).transpose())(0, 0);

            dLdk2 -= C * inv_nJ_C * inv_nJ_C * inv_nJ_C *
                     (d2.transpose() * F * J1 * (x2_sq * F.row(2) * J1).transpose() +
                      d1.transpose() * F.transpose() * J2 * (d1.transpose() * F.transpose() * dJ2dk).transpose())(0, 0);

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation + k)
            Eigen::Matrix<double, 1, 9> J;
            J.block<1, 7>(0, 0) = dF * dF_dparams;
            J(0, 7) = dLdk1;
            J(0, 8) = dLdk2;

            // and then w.r.t. the fundamental matrix parameters
            acc.add_jacobian(r, J, weights[i]);
        }
    }

    FactorizedProjectiveImagePair step(const Eigen::VectorXd &dp,
                                       const FactorizedProjectiveImagePair &proj_image_pair) const {
        FactorizedFundamentalMatrix F = proj_image_pair.FF;
        FactorizedFundamentalMatrix F_new;

        F_new.qU = quat_step_pre(F.qU, dp.block<3, 1>(0, 0));
        F_new.qV = quat_step_pre(F.qV, dp.block<3, 1>(3, 0));
        F_new.sigma = F.sigma + dp(6);

        Camera camera1_new = Camera(
            "DIVISION", std::vector<double>{1.0, 1.0, 0.0, 0.0, proj_image_pair.camera1.params[4] + dp(7)}, -1, -1);
        Camera camera2_new = Camera(
            "DIVISION", std::vector<double>{1.0, 1.0, 0.0, 0.0, proj_image_pair.camera2.params[4] + dp(8)}, -1, -1);

        return FactorizedProjectiveImagePair(F_new, camera1_new, camera2_new);
    }

    typedef FactorizedProjectiveImagePair param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

} // namespace poselib

#endif