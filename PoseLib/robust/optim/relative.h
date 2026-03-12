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

inline void setup_tangent_basis(const Eigen::Vector3d &t, Eigen::Matrix<double, 3, 2> &tangent_basis) {
    // We start by setting up a basis for the updates in the translation (orthogonal to t)
    // We find the minimum element of t and cross product with the corresponding basis vector.
    // (this ensures that the first cross product is not close to the zero vector)
    if (std::abs(t.x()) < std::abs(t.y())) {
        // x < y
        if (std::abs(t.x()) < std::abs(t.z())) {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitX()).normalized();
        } else {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitZ()).normalized();
        }
    } else {
        // x > y
        if (std::abs(t.y()) < std::abs(t.z())) {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitY()).normalized();
        } else {
            tangent_basis.col(0) = t.cross(Eigen::Vector3d::UnitZ()).normalized();
        }
    }
    tangent_basis.col(1) = tangent_basis.col(0).cross(t).normalized();
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

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);
        setup_tangent_basis(pose.t, tangent_basis);

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

// Minimize Tangent Sampson error with any camera model. Assumes fixed camera intrinsics.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class FixCameraRelativePoseRefiner : public RefinerBase<CameraPose, Accumulator> {
  public:
    FixCameraRelativePoseRefiner(const std::vector<Point3D> &unproj_points2D_1,
                                 const std::vector<Point3D> &unproj_points2D_2,
                                 const std::vector<Eigen::Matrix<double, 3, 2>> &J1inv,
                                 const std::vector<Eigen::Matrix<double, 3, 2>> &J2inv,
                                 const ResidualWeightVector &w = ResidualWeightVector())
        : d1(unproj_points2D_1), d2(unproj_points2D_2), M1(J1inv), M2(J2inv), weights(w) {
        this->num_params = 5;
    }

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        Eigen::Matrix3d E;
        essential_from_motion(pose, &E);

        for (size_t k = 0; k < d1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);
            double nJc_sq = (M2[k].transpose() * E * d1[k]).squaredNorm() +
                            (M1[k].transpose() * E.transpose() * d2[k]).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {
        setup_tangent_basis(pose.t, tangent_basis);

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;
        deriv_essential_wrt_pose(E, R, tangent_basis, dR, dt);

        for (size_t k = 0; k < d1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << M1[k].transpose() * E.transpose() * d2[k], M2[k].transpose() * E * d1[k];
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << d1[k](0) * d2[k](0), d1[k](0) * d2[k](1), d1[k](0) * d2[k](2), d1[k](1) * d2[k](0),
                d1[k](1) * d2[k](1), d1[k](1) * d2[k](2), d1[k](2) * d2[k](0), d1[k](2) * d2[k](1), d1[k](2) * d2[k](2);
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(0) * M1[k](0, 0) * d2[k](0) + J_C(1) * M1[k](0, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](0) + J_C(3) * M2[k](0, 1) * d1[k](0));
            dF(1) -= s * (J_C(0) * M1[k](0, 0) * d2[k](1) + J_C(1) * M1[k](0, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](0) + J_C(3) * M2[k](1, 1) * d1[k](0));
            dF(2) -= s * (J_C(0) * M1[k](0, 0) * d2[k](2) + J_C(1) * M1[k](0, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](0) + J_C(3) * M2[k](2, 1) * d1[k](0));
            dF(3) -= s * (J_C(0) * M1[k](1, 0) * d2[k](0) + J_C(1) * M1[k](1, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](1) + J_C(3) * M2[k](0, 1) * d1[k](1));
            dF(4) -= s * (J_C(0) * M1[k](1, 0) * d2[k](1) + J_C(1) * M1[k](1, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](1) + J_C(3) * M2[k](1, 1) * d1[k](1));
            dF(5) -= s * (J_C(0) * M1[k](1, 0) * d2[k](2) + J_C(1) * M1[k](1, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](1) + J_C(3) * M2[k](2, 1) * d1[k](1));
            dF(6) -= s * (J_C(0) * M1[k](2, 0) * d2[k](0) + J_C(1) * M1[k](2, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](2) + J_C(3) * M2[k](0, 1) * d1[k](2));
            dF(7) -= s * (J_C(0) * M1[k](2, 0) * d2[k](1) + J_C(1) * M1[k](2, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](2) + J_C(3) * M2[k](1, 1) * d1[k](2));
            dF(8) -= s * (J_C(0) * M1[k](2, 0) * d2[k](2) + J_C(1) * M1[k](2, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](2) + J_C(3) * M2[k](2, 1) * d1[k](2));
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
    const std::vector<Point3D> &d1;
    const std::vector<Point3D> &d2;
    const std::vector<Eigen::Matrix<double, 3, 2>> &M1;
    const std::vector<Eigen::Matrix<double, 3, 2>> &M2;

    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;
};

// Minimize Tangent Sampson error with any camera model. Allows for optimization of camera intrinsics.
template <typename ResidualWeightVector = UniformWeightVector, typename Accumulator = NormalAccumulator>
class CameraRelativePoseRefiner : public RefinerBase<ImagePair, Accumulator> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  public:
    CameraRelativePoseRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                              const std::vector<size_t> &cam1_ref_idx, const std::vector<size_t> &cam2_ref_idx,
                              const bool shared_camera = false, // Shared intrinsics only use camera1
                              const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), camera1_refine_idx(cam1_ref_idx), camera2_refine_idx(cam2_ref_idx),
          shared_intrinsics(shared_camera), weights(w) {
        this->num_params = 5 + (shared_intrinsics ? cam1_ref_idx.size() : cam1_ref_idx.size() + cam2_ref_idx.size());
        d1.reserve(x1.size());
        d2.reserve(x2.size());
        M1.reserve(x1.size());
        M2.reserve(x2.size());
        d1_p.reserve(x1.size());
        d2_p.reserve(x2.size());
    }

    double compute_residual(Accumulator &acc, const ImagePair &pair) {
        Eigen::Matrix3d E;
        essential_from_motion(pair.pose, &E);

        const Camera &camera1 = pair.camera1;
        const Camera &camera2 = shared_intrinsics ? pair.camera1 : pair.camera2;

        camera1.unproject_with_jac(x1, &d1, &M1);
        camera2.unproject_with_jac(x2, &d2, &M2);

        for (size_t k = 0; k < d1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);
            double nJc_sq = (M2[k].transpose() * E * d1[k]).squaredNorm() +
                            (M1[k].transpose() * E.transpose() * d2[k]).squaredNorm();

            acc.add_residual(C / std::sqrt(nJc_sq), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const ImagePair &pair) {
        const CameraPose &pose = pair.pose;
        setup_tangent_basis(pose.t, tangent_basis);

        Eigen::Matrix3d E, R;
        R = pose.R();
        essential_from_motion(pose, &E);

        // Matrices contain the jacobians of E w.r.t. the rotation and translation parameters
        Eigen::Matrix<double, 9, 3> dR;
        Eigen::Matrix<double, 9, 2> dt;
        deriv_essential_wrt_pose(E, R, tangent_basis, dR, dt);

        const Camera &camera1 = pair.camera1;
        const Camera &camera2 = shared_intrinsics ? pair.camera1 : pair.camera2;

        if (camera1_refine_idx.size() + camera2_refine_idx.size() > 0) {
            camera1.unproject_with_jac(x1, &d1, &M1, &d1_p);
            camera2.unproject_with_jac(x2, &d2, &M2, &d2_p);
        } else {
            camera1.unproject_with_jac(x1, &d1, &M1);
            camera2.unproject_with_jac(x2, &d2, &M2);
        }

        for (size_t k = 0; k < x1.size(); ++k) {
            double C = d2[k].dot(E * d1[k]);

            // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
            Eigen::Vector4d J_C;
            J_C << M1[k].transpose() * E.transpose() * d2[k], M2[k].transpose() * E * d1[k];
            const double nJ_C = J_C.norm();
            const double inv_nJ_C = 1.0 / nJ_C;
            const double r = C * inv_nJ_C;

            // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
            Eigen::Matrix<double, 1, 9> dF;
            dF << d1[k](0) * d2[k](0), d1[k](0) * d2[k](1), d1[k](0) * d2[k](2), d1[k](1) * d2[k](0),
                d1[k](1) * d2[k](1), d1[k](1) * d2[k](2), d1[k](2) * d2[k](0), d1[k](2) * d2[k](1), d1[k](2) * d2[k](2);
            const double s = C * inv_nJ_C * inv_nJ_C;
            dF(0) -= s * (J_C(0) * M1[k](0, 0) * d2[k](0) + J_C(1) * M1[k](0, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](0) + J_C(3) * M2[k](0, 1) * d1[k](0));
            dF(1) -= s * (J_C(0) * M1[k](0, 0) * d2[k](1) + J_C(1) * M1[k](0, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](0) + J_C(3) * M2[k](1, 1) * d1[k](0));
            dF(2) -= s * (J_C(0) * M1[k](0, 0) * d2[k](2) + J_C(1) * M1[k](0, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](0) + J_C(3) * M2[k](2, 1) * d1[k](0));
            dF(3) -= s * (J_C(0) * M1[k](1, 0) * d2[k](0) + J_C(1) * M1[k](1, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](1) + J_C(3) * M2[k](0, 1) * d1[k](1));
            dF(4) -= s * (J_C(0) * M1[k](1, 0) * d2[k](1) + J_C(1) * M1[k](1, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](1) + J_C(3) * M2[k](1, 1) * d1[k](1));
            dF(5) -= s * (J_C(0) * M1[k](1, 0) * d2[k](2) + J_C(1) * M1[k](1, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](1) + J_C(3) * M2[k](2, 1) * d1[k](1));
            dF(6) -= s * (J_C(0) * M1[k](2, 0) * d2[k](0) + J_C(1) * M1[k](2, 1) * d2[k](0) +
                          J_C(2) * M2[k](0, 0) * d1[k](2) + J_C(3) * M2[k](0, 1) * d1[k](2));
            dF(7) -= s * (J_C(0) * M1[k](2, 0) * d2[k](1) + J_C(1) * M1[k](2, 1) * d2[k](1) +
                          J_C(2) * M2[k](1, 0) * d1[k](2) + J_C(3) * M2[k](1, 1) * d1[k](2));
            dF(8) -= s * (J_C(0) * M1[k](2, 0) * d2[k](2) + J_C(1) * M1[k](2, 1) * d2[k](2) +
                          J_C(2) * M2[k](2, 0) * d1[k](2) + J_C(3) * M2[k](2, 1) * d1[k](2));
            dF *= inv_nJ_C;

            // and then w.r.t. the pose parameters (rotation + tangent basis for translation)
            Eigen::Matrix<double, 1, Eigen::Dynamic> J(1, this->num_params);
            J.block<1, 3>(0, 0) = dF * dR;
            J.block<1, 2>(0, 3) = dF * dt;

            if (camera1_refine_idx.size() + camera2_refine_idx.size() > 0) {
                // Jacobian w.r.t. unprojected points
                Eigen::Matrix<double, 1, 3> J_d1, J_d2;
                J_d1 = (d2[k].transpose() * E -
                        C * inv_nJ_C * inv_nJ_C * (d1[k].transpose() * E.transpose() * M2[k] * M2[k].transpose() * E)) *
                       inv_nJ_C;
                J_d2 = (d1[k].transpose() * E.transpose() -
                        C * inv_nJ_C * inv_nJ_C * (d2[k].transpose() * E * M1[k] * M1[k].transpose() * E.transpose())) *
                       inv_nJ_C;

                // Jacobian w.r.t. inverse jacobians of unprojections
                Eigen::Matrix<double, 1, 3> J_M11, J_M12, J_M21, J_M22;
                J_M11 = -s * inv_nJ_C * M1[k].col(0).transpose() * E.transpose() * d2[k] * d2[k].transpose() * E;
                J_M12 = -s * inv_nJ_C * M1[k].col(1).transpose() * E.transpose() * d2[k] * d2[k].transpose() * E;
                J_M21 = -s * inv_nJ_C * M2[k].col(0).transpose() * E * d1[k] * d1[k].transpose() * E.transpose();
                J_M22 = -s * inv_nJ_C * M2[k].col(1).transpose() * E * d1[k] * d1[k].transpose() * E.transpose();

                // Since we don't have analytic second order mixed partial derivatives, we do a finite difference
                // approximation of the analytic jacobian w.r.t. the camera intrinsics
                const double eps = 1e-6;
                Eigen::Matrix<double, 3, Eigen::Dynamic> dxp, dp_e1, dp_e2;
                Eigen::Matrix<double, 3, 2> dummy0;
                Eigen::Vector3d dummy;
                Eigen::Vector2d x_e1, x_e2;

                // For first camera
                x_e1 << x1[k](0) + eps, x1[k](1);
                x_e2 << x1[k](0), x1[k](1) + eps;
                camera1.unproject_with_jac(x_e1, &dummy, &dummy0, &dp_e1);
                camera1.unproject_with_jac(x_e2, &dummy, &dummy0, &dp_e2);
                dp_e1 -= d1_p[k];
                dp_e1 /= eps;
                dp_e2 -= d1_p[k];
                dp_e2 /= eps;

                for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                    J(0, 5 + i) = J_d1.dot(d1_p[k].col(camera1_refine_idx[i])) +
                                  J_M11.dot(dp_e1.col(camera1_refine_idx[i])) +
                                  J_M12.dot(dp_e2.col(camera1_refine_idx[i]));
                }

                // For second camera
                x_e1 << x2[k](0) + eps, x2[k](1);
                x_e2 << x2[k](0), x2[k](1) + eps;
                camera2.unproject_with_jac(x_e1, &dummy, nullptr, &dp_e1);
                camera2.unproject_with_jac(x_e2, &dummy, nullptr, &dp_e2);
                dp_e1 -= d2_p[k];
                dp_e1 /= eps;
                dp_e2 -= d2_p[k];
                dp_e2 /= eps;

                if (shared_intrinsics) {
                    for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                        J(0, 5 + i) += J_d2.dot(d2_p[k].col(camera1_refine_idx[i])) +
                                       J_M21.dot(dp_e1.col(camera1_refine_idx[i])) +
                                       J_M22.dot(dp_e2.col(camera1_refine_idx[i]));
                    }
                } else {
                    for (size_t i = 0; i < camera2_refine_idx.size(); ++i) {
                        J(0, 5 + camera1_refine_idx.size() + i) = J_d2.dot(d2_p[k].col(camera2_refine_idx[i])) +
                                                                  J_M21.dot(dp_e1.col(camera2_refine_idx[i])) +
                                                                  J_M22.dot(dp_e2.col(camera2_refine_idx[i]));
                    }
                }
            }

            acc.add_jacobian(r, J, weights[k]);
        }
    }

    ImagePair step(const Eigen::VectorXd &dp, const ImagePair &pair) const {
        ImagePair image_pair_new;
        image_pair_new.camera1 = pair.camera1;
        image_pair_new.camera2 = pair.camera2;

        image_pair_new.pose.q = quat_step_post(pair.pose.q, dp.block<3, 1>(0, 0));
        image_pair_new.pose.t = pair.pose.t + tangent_basis * dp.block<2, 1>(3, 0);

        if (shared_intrinsics) {
            // We have shared intrinsics for both cameras
            for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                image_pair_new.camera1.params[camera1_refine_idx[i]] += dp(5 + i);
            }
            image_pair_new.camera2 = image_pair_new.camera1;
        } else {
            // Update intrinsics for first camera
            for (size_t i = 0; i < camera1_refine_idx.size(); ++i) {
                image_pair_new.camera1.params[camera1_refine_idx[i]] += dp(5 + i);
            }
            // and second camera
            for (size_t i = 0; i < camera2_refine_idx.size(); ++i) {
                image_pair_new.camera2.params[camera2_refine_idx[i]] += dp(5 + camera1_refine_idx.size() + i);
            }
        }
        return image_pair_new;
    }

    typedef ImagePair param_t;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<size_t> camera1_refine_idx;
    const std::vector<size_t> camera2_refine_idx;
    const bool shared_intrinsics;
    const ResidualWeightVector &weights;
    Eigen::Matrix<double, 3, 2> tangent_basis;

    // Pre-allocated vectors for undistortion
    std::vector<Point3D> d1, d2;
    std::vector<Eigen::Matrix<double, 3, 2>> M1, M2;
    std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> d1_p, d2_p;
};

// Minimize Sampson error with pinhole camera model for relative pose and one unknown focal length shared by both
// cameras.
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
        setup_tangent_basis(image_pair.pose.t, tangent_basis);

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