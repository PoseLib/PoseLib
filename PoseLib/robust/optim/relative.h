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

#ifndef POSELIB_RELATIVE_H_
#define POSELIB_RELATIVE_H_

#include "../../misc/essential.h"
#include "../../types.h"
#include "optim_utils.h"
#include "refiner_base.h"

namespace poselib {

inline void deriv_essential_wrt_pose(const Eigen::Matrix3d &E, const Eigen::Matrix3d &R,
                                     const Eigen::Matrix<double, 3, 2> &tangent_basis, Eigen::Matrix<double, 9, 3> &dR,
                                     Eigen::Matrix<double, 9, 2> &dt) {
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
}

// Minimize Sampson error with pinhole camera model. Assumes image points are in the normalized image plane.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class PinholeRelativePoseRefiner : public RefinerBase<CameraPose, Accumulator> {
  public:
    PinholeRelativePoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                               const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), weights(w) {
        this->num_params = 5;
    }

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());
            double nJc_sq = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {
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
        deriv_essential_wrt_pose(E, R, tangent_basis, dR, dt);

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous(), E.block<2, 3>(0, 0) * x1[k].homogeneous();
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

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 5> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            acc.add_jacobian(r, J, weights[k]);
        }
    }

    CameraPose step(const Eigen::VectorXd &dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + tangent_basis * dp.block<2, 1>(3, 0);
        return pose_new;
    }

    typedef CameraPose param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

// Minimize Sampson error with pinhole camera model for relative pose and one unknown focal length shared by both cameras. 
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class SharedFocalRelativePoseRefiner : public RefinerBase<ImagePair, Accumulator> {
  public:
    SharedFocalRelativePoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                               const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), weights(w) {
        this->num_params = 6;
    }

    double compute_residual(Accumulator &acc, const ImagePair &image_pair) {
        Eigen::Matrix3d E, F;
        essential_from_motion(image_pair.pose, &E);
        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, image_pair.camera1.focal());
        F = K_inv * E * K_inv;

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = x2[k].homogeneous().dot(F * x1[k].homogeneous());
            double nJc_sq = (F.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() +
                            (F.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ImagePair &image_pair) {
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

        Eigen::Matrix3d E, F, R;
        R = image_pair.pose.R();
        essential_from_motion(image_pair.pose, &E);
        double focal = image_pair.camera1.focal();
        Eigen::DiagonalMatrix<double, 3> K_inv(1.0, 1.0, focal);
        F = K_inv * E * K_inv;

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;
        deriv_essential_wrt_pose(E, R, tangent_basis, dR, dt);

        dR.row(2) *= focal;
        dR.row(5) *= focal;
        dR.row(6) *= focal;
        dR.row(7) *= focal;
        dR.row(8) *= focal * focal;

        dt.row(2) *= focal;
        dt.row(5) *= focal;
        dt.row(6) *= focal;
        dt.row(7) *= focal;
        dt.row(8) *= focal * focal;

        Eigen::Matrix<double, 9, 1> df;
        df << 0.0, 0.0, E(2, 0), 0.0, 0.0, E(2, 1), E(0, 2), E(1, 2), 2 * E(2, 2) * focal;

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

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, 6> J;
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;
            J(5) = dF * df;

            acc.add_jacobian(r, J, weights[k]);
        }
    }

    ImagePair step(const Eigen::VectorXd &dp, const ImagePair &image_pair) const {
        CameraPose new_pose;
        new_pose.q = quat_step_post(image_pair.pose.q, dp.block<3, 1>(0, 0));
        new_pose.t = image_pair.pose.t + tangent_basis * dp.block<2, 1>(3, 0);

        Camera new_camera = Camera("SIMPLE_PINHOLE", {image_pair.camera1.focal() + dp(5), 0, 0}, -1, -1);

        return ImagePair(new_pose, new_camera, new_camera);
    }

    typedef ImagePair param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

} // namespace poselib

#endif