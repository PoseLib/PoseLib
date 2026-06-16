// Copyright (c) 2020, Viktor Larsson
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
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/re3q3.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {
namespace detail {

// Internal helper for PnPLf solvers: any mix of point and line correspondences with unknown focal length.
// Generalizes p4pf: 4 correspondences (each giving 2 constraints) → 8×12 system → 4D null space → re3q3.
//
// Point constraint: lambda * diag(1/fx, 1/fy, 1) * [xp; 1] = R * Xp + t
//   → two rows per point from cross-product elimination
// Line constraint: l^T * (R * (X + mu*V) + t) = 0
//   → l^T * R * V = 0 and l^T * (R*X + t) = 0, two rows per line
inline int pnplf_impl(const std::vector<Eigen::Vector2d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                      const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                      const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output,
                      std::vector<double> *output_fx, std::vector<double> *output_fy, bool filter_solutions) {

    const int n_pts = static_cast<int>(xp.size());
    const int n_lines = static_cast<int>(l.size());

    // Normalization: compute f0 from mean norm of 2D point coords
    double f0 = 1.0;
    if (n_pts > 0) {
        double sum_norm = 0.0;
        for (int i = 0; i < n_pts; ++i) {
            sum_norm += xp[i].norm();
        }
        f0 = sum_norm / n_pts;
    } else {
        // For pure line case, estimate f0 from distance of lines to image center.
        // For line l0*x + l1*y + l2 = 0, the distance to the origin is |l2| / ||(l0, l1)||.
        // This is scale-invariant (unlike ||(l0, l1)|| alone) and represents the typical
        // image coordinate magnitude, analogous to mean(||xp||) for points.
        double sum_dist = 0.0;
        for (int i = 0; i < n_lines; ++i) {
            double n12 = Eigen::Vector2d(l[i](0), l[i](1)).norm();
            if (n12 > 0.0)
                sum_dist += std::abs(l[i](2)) / n12;
        }
        if (n_lines > 0 && sum_dist > 0.0)
            f0 = sum_dist / n_lines;
    }

    // Normalized 2D points
    Eigen::Matrix<double, 2, Eigen::Dynamic> points2d(2, n_pts);
    for (int i = 0; i < n_pts; ++i) {
        points2d.col(i) = xp[i] / f0;
    }

    // Normalized line normals: l_hat = (l0, l1, l2/f0) (before unit normalization).
    // The line constraint (in pixel coords) is: l^T * diag(fx,fy,1) * (R*X + t) = 0
    // With P = diag(fx/f0, fy/f0, 1) * [R | t], we have (R*X+t) = diag(f0/fx, f0/fy, 1) * P * [X;1]
    // Substituting: (f0*l0, f0*l1, l2) * P * [X;1] = 0, i.e., (l0, l1, l2/f0) * P * [X;1] = 0
    std::vector<Eigen::Vector3d> l_norm(n_lines);
    for (int i = 0; i < n_lines; ++i) {
        l_norm[i] = Eigen::Vector3d(l[i](0), l[i](1), l[i](2) / f0);
        l_norm[i].normalize();
    }

    // Build 8×12 constraint matrix C where vec(P) is in row-major order:
    //   [P(0,0), P(0,1), P(0,2), P(0,3), P(1,0), P(1,1), P(1,2), P(1,3), P(2,0), P(2,1), P(2,2), P(2,3)]
    // P = diag(fx/f0, fy/f0, 1) * [R | t]
    Eigen::Matrix<double, 8, 12> C;
    C.setZero();
    int row = 0;

    // Point constraints: u*(P_row2*[X;1]) - P_row0*[X;1] = 0, v*(P_row2*[X;1]) - P_row1*[X;1] = 0
    for (int i = 0; i < n_pts; ++i) {
        double u = points2d(0, i);
        double v = points2d(1, i);
        double X0 = Xp[i](0), X1 = Xp[i](1), X2 = Xp[i](2);

        // Constraint 1: u * (P_row2 * [X;1]) - (P_row0 * [X;1]) = 0
        // -p1*X0 - p2*X1 - p3*X2 - p4 + u*p9*X0 + u*p10*X1 + u*p11*X2 + u*p12 = 0
        C(row, 0) = -X0;
        C(row, 1) = -X1;
        C(row, 2) = -X2;
        C(row, 3) = -1.0;
        C(row, 8) = u * X0;
        C(row, 9) = u * X1;
        C(row, 10) = u * X2;
        C(row, 11) = u;
        row++;

        // Constraint 2: v * (P_row2 * [X;1]) - (P_row1 * [X;1]) = 0
        // -p5*X0 - p6*X1 - p7*X2 - p8 + v*p9*X0 + v*p10*X1 + v*p11*X2 + v*p12 = 0
        C(row, 4) = -X0;
        C(row, 5) = -X1;
        C(row, 6) = -X2;
        C(row, 7) = -1.0;
        C(row, 8) = v * X0;
        C(row, 9) = v * X1;
        C(row, 10) = v * X2;
        C(row, 11) = v;
        row++;
    }

    // Line constraints: l^T * P * [X;1] = 0 and l^T * P * [V;0] = 0
    // l^T * P = [l0*p1+l1*p5+l2*p9, l0*p2+l1*p6+l2*p10, l0*p3+l1*p7+l2*p11, l0*p4+l1*p8+l2*p12]
    // Dot with [V;0]: (l0*p1+l1*p5+l2*p9)*V0 + (l0*p2+l1*p6+l2*p10)*V1 + (l0*p3+l1*p7+l2*p11)*V2 = 0
    // Dot with [X;1]: (l0*p1+l1*p5+l2*p9)*X0 + (l0*p2+l1*p6+l2*p10)*X1 + (l0*p3+l1*p7+l2*p11)*X2 + l0*p4+l1*p8+l2*p12 =
    // 0

    for (int i = 0; i < n_lines; ++i) {
        double l0 = l_norm[i](0), l1 = l_norm[i](1), l2 = l_norm[i](2);
        double V0 = V[i](0), V1 = V[i](1), V2 = V[i](2);
        double X0 = X[i](0), X1 = X[i](1), X2 = X[i](2);

        // Constraint: l^T * P * [V;0] = 0
        C(row, 0) = l0 * V0;
        C(row, 1) = l0 * V1;
        C(row, 2) = l0 * V2;
        C(row, 4) = l1 * V0;
        C(row, 5) = l1 * V1;
        C(row, 6) = l1 * V2;
        C(row, 8) = l2 * V0;
        C(row, 9) = l2 * V1;
        C(row, 10) = l2 * V2;
        row++;

        // Constraint: l^T * P * [X;1] = 0
        C(row, 0) = l0 * X0;
        C(row, 1) = l0 * X1;
        C(row, 2) = l0 * X2;
        C(row, 3) = l0;
        C(row, 4) = l1 * X0;
        C(row, 5) = l1 * X1;
        C(row, 6) = l1 * X2;
        C(row, 7) = l1;
        C(row, 8) = l2 * X0;
        C(row, 9) = l2 * X1;
        C(row, 10) = l2 * X2;
        C(row, 11) = l2;
        row++;
    }

    // Compute 4D null space of 8×12 matrix via QR
    // We need the last 4 columns of Q (or equivalently, SVD right null space)
    // Following p4pf approach but adapted for 8×12
    Eigen::Matrix<double, 12, 8> Ct = C.transpose();
    Eigen::Matrix<double, 12, 12> Q = Ct.householderQr().householderQ();
    Eigen::Matrix<double, 12, 4> N = Q.rightCols(4);

    // N parametrizes vec(P): [p1..p12] = N * [alpha0, alpha1, alpha2, 1]^T
    // P_row0 = [N(0,:); N(1,:); N(2,:); N(3,:)] * alpha  (entries 0-3)
    // P_row1 = [N(4,:); N(5,:); N(6,:); N(7,:)] * alpha  (entries 4-7)
    // P_row2 = [N(8,:); N(9,:); N(10,:); N(11,:)] * alpha (entries 8-11)

    // Build orthogonality constraints on the 3×3 leading submatrix of P.
    // ri = P_row_i 3D part, parametrized as ri(j) = coeffs(j,:) * [a0,a1,a2,1].
    // Constraints: r0.r1 = 0, r0.r2 = 0, r1.r2 = 0 → 3 quadratics in (a0,a1,a2) → re3q3.
    auto dot_product_coeffs = [](const Eigen::Matrix<double, 3, 4> &ra,
                                 const Eigen::Matrix<double, 3, 4> &rb) -> Eigen::Matrix<double, 1, 10> {
        Eigen::Matrix<double, 1, 10> c;
        c.setZero();
        for (int j = 0; j < 3; ++j) {
            // ra(j,:) = [a0_coeff, a1_coeff, a2_coeff, const_coeff]
            // rb(j,:) = [b0_coeff, b1_coeff, b2_coeff, const_coeff]
            double a0 = ra(j, 0), a1 = ra(j, 1), a2 = ra(j, 2), a3 = ra(j, 3);
            double b0 = rb(j, 0), b1 = rb(j, 1), b2 = rb(j, 2), b3 = rb(j, 3);

            // Product terms (ordered: a0^2, a0*a1, a0*a2, a1^2, a1*a2, a2^2, a0, a1, a2, 1)
            c(0) += a0 * b0;           // a0^2
            c(1) += a0 * b1 + a1 * b0; // a0*a1
            c(2) += a0 * b2 + a2 * b0; // a0*a2
            c(3) += a1 * b1;           // a1^2
            c(4) += a1 * b2 + a2 * b1; // a1*a2
            c(5) += a2 * b2;           // a2^2
            c(6) += a0 * b3 + a3 * b0; // a0
            c(7) += a1 * b3 + a3 * b1; // a1
            c(8) += a2 * b3 + a3 * b2; // a2
            c(9) += a3 * b3;           // 1
        }
        return c;
    };

    Eigen::Matrix<double, 3, 4> r0, r1, r2;
    r0 = N.block<3, 4>(0, 0); // rows 0,1,2 of N
    r1 = N.block<3, 4>(4, 0); // rows 4,5,6 of N
    r2 = N.block<3, 4>(8, 0); // rows 8,9,10 of N — directly from null space

    Eigen::Matrix<double, 3, 10> coeffs;
    coeffs.row(0) = dot_product_coeffs(r0, r1); // r0 . r1 = 0
    coeffs.row(1) = dot_product_coeffs(r0, r2); // r0 . r2 = 0
    coeffs.row(2) = dot_product_coeffs(r1, r2); // r1 . r2 = 0

    Eigen::Matrix<double, 3, 8> solutions;
    int n_sols = re3q3::re3q3(coeffs, &solutions);

    output->clear();
    output->reserve(n_sols);
    output_fx->clear();
    output_fx->reserve(n_sols);
    output_fy->clear();
    output_fy->reserve(n_sols);

    for (int i = 0; i < n_sols; ++i) {
        Eigen::Matrix<double, 3, 4> P;
        Eigen::Vector4d alpha;
        alpha << solutions.col(i), 1.0;

        // P_row0 = [N(0,:)*alpha, N(1,:)*alpha, N(2,:)*alpha, N(3,:)*alpha]
        for (int j = 0; j < 4; ++j)
            P(0, j) = N.row(j).dot(alpha);
        for (int j = 0; j < 4; ++j)
            P(1, j) = N.row(j + 4).dot(alpha);
        for (int j = 0; j < 4; ++j)
            P(2, j) = N.row(j + 8).dot(alpha);

        if (P.block<3, 3>(0, 0).determinant() < 0)
            P = -P;

        P = P / P.block<1, 3>(2, 0).norm();
        double fx = P.block<1, 3>(0, 0).norm();
        double fy = P.block<1, 3>(1, 0).norm();
        P.row(0) = P.row(0) / fx;
        P.row(1) = P.row(1) / fy;

        Eigen::Matrix3d R = P.block<3, 3>(0, 0);
        Eigen::Vector3d t = P.block<3, 1>(0, 3);
        fx *= f0;
        fy *= f0;

        CameraPose pose(R, t);

        if (filter_solutions) {
            if (fx < 0 || fy < 0)
                continue;

            // Check cheirality for points
            bool ok = true;
            for (int k = 0; k < n_pts; ++k) {
                if (R.row(2).dot(Xp[k]) + t(2) < 0.0) {
                    ok = false;
                    break;
                }
            }
            if (!ok)
                continue;
        }
        output->push_back(pose);
        output_fx->push_back(fx);
        output_fy->push_back(fy);
    }
    return output->size();
}

} // namespace detail
} // namespace poselib
