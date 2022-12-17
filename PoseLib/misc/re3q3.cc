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

#include "re3q3.h"

#include "PoseLib/misc/quaternion.h"
#include "PoseLib/misc/sturm.h"

#include <Eigen/Dense>

namespace poselib {
namespace re3q3 {

/* Homogeneous linear constraints on rotation matrix
     Rcoeffs*R(:) = 0
  converted into 3q3 problem. */
void rotation_to_3q3(const Eigen::Matrix<double, 3, 9> &Rcoeffs, Eigen::Matrix<double, 3, 10> *coeffs) {
    for (int k = 0; k < 3; k++) {
        (*coeffs)(k, 0) = Rcoeffs(k, 0) - Rcoeffs(k, 4) - Rcoeffs(k, 8);
        (*coeffs)(k, 1) = 2 * Rcoeffs(k, 1) + 2 * Rcoeffs(k, 3);
        (*coeffs)(k, 2) = 2 * Rcoeffs(k, 2) + 2 * Rcoeffs(k, 6);
        (*coeffs)(k, 3) = Rcoeffs(k, 4) - Rcoeffs(k, 0) - Rcoeffs(k, 8);
        (*coeffs)(k, 4) = 2 * Rcoeffs(k, 5) + 2 * Rcoeffs(k, 7);
        (*coeffs)(k, 5) = Rcoeffs(k, 8) - Rcoeffs(k, 4) - Rcoeffs(k, 0);
        (*coeffs)(k, 6) = 2 * Rcoeffs(k, 5) - 2 * Rcoeffs(k, 7);
        (*coeffs)(k, 7) = 2 * Rcoeffs(k, 6) - 2 * Rcoeffs(k, 2);
        (*coeffs)(k, 8) = 2 * Rcoeffs(k, 1) - 2 * Rcoeffs(k, 3);
        (*coeffs)(k, 9) = Rcoeffs(k, 0) + Rcoeffs(k, 4) + Rcoeffs(k, 8);
    }
}

/* Inhomogeneous linear constraints on rotation matrix
     Rcoeffs*[R(:);1] = 0
  converted into 3q3 problem. */
void rotation_to_3q3(const Eigen::Matrix<double, 3, 10> &Rcoeffs, Eigen::Matrix<double, 3, 10> *coeffs) {
    for (int k = 0; k < 3; k++) {
        (*coeffs)(k, 0) = Rcoeffs(k, 0) - Rcoeffs(k, 4) - Rcoeffs(k, 8) + Rcoeffs(k, 9);
        (*coeffs)(k, 1) = 2 * Rcoeffs(k, 1) + 2 * Rcoeffs(k, 3);
        (*coeffs)(k, 2) = 2 * Rcoeffs(k, 2) + 2 * Rcoeffs(k, 6);
        (*coeffs)(k, 3) = Rcoeffs(k, 4) - Rcoeffs(k, 0) - Rcoeffs(k, 8) + Rcoeffs(k, 9);
        (*coeffs)(k, 4) = 2 * Rcoeffs(k, 5) + 2 * Rcoeffs(k, 7);
        (*coeffs)(k, 5) = Rcoeffs(k, 8) - Rcoeffs(k, 4) - Rcoeffs(k, 0) + Rcoeffs(k, 9);
        (*coeffs)(k, 6) = 2 * Rcoeffs(k, 5) - 2 * Rcoeffs(k, 7);
        (*coeffs)(k, 7) = 2 * Rcoeffs(k, 6) - 2 * Rcoeffs(k, 2);
        (*coeffs)(k, 8) = 2 * Rcoeffs(k, 1) - 2 * Rcoeffs(k, 3);
        (*coeffs)(k, 9) = Rcoeffs(k, 0) + Rcoeffs(k, 4) + Rcoeffs(k, 8) + Rcoeffs(k, 9);
    }
}

void cayley_param(const Eigen::Matrix<double, 3, 1> &c, Eigen::Matrix<double, 3, 3> *R) {
    *R << c(0) * c(0) - c(1) * c(1) - c(2) * c(2) + 1, 2 * c(0) * c(1) - 2 * c(2), 2 * c(1) + 2 * c(0) * c(2),
        2 * c(2) + 2 * c(0) * c(1), c(1) * c(1) - c(0) * c(0) - c(2) * c(2) + 1, 2 * c(1) * c(2) - 2 * c(0),
        2 * c(0) * c(2) - 2 * c(1), 2 * c(0) + 2 * c(1) * c(2), c(2) * c(2) - c(1) * c(1) - c(0) * c(0) + 1;
    *R /= 1 + c(0) * c(0) + c(1) * c(1) + c(2) * c(2);
}

inline void refine_3q3(const Eigen::Matrix<double, 3, 10> &coeffs, Eigen::Matrix<double, 3, 8> *solutions, int n_sols) {
    Eigen::Matrix3d J;
    Eigen::Vector3d r;
    Eigen::Vector3d dx;
    double x, y, z;

    for (int i = 0; i < n_sols; ++i) {
        x = (*solutions)(0, i);
        y = (*solutions)(1, i);
        z = (*solutions)(2, i);

        // [x^2, x*y, x*z, y^2, y*z, z^2, x, y, z, 1.0]
        for (int iter = 0; iter < 5; ++iter) {
            r = coeffs.col(0) * x * x + coeffs.col(1) * x * y + coeffs.col(2) * x * z + coeffs.col(3) * y * y +
                coeffs.col(4) * y * z + coeffs.col(5) * z * z + coeffs.col(6) * x + coeffs.col(7) * y +
                coeffs.col(8) * z + coeffs.col(9);

            if (r.cwiseAbs().maxCoeff() < 1e-8)
                break;

            J.col(0) = 2.0 * coeffs.col(0) * x + coeffs.col(1) * y + coeffs.col(2) * z + coeffs.col(6);
            J.col(1) = coeffs.col(1) * x + 2.0 * coeffs.col(3) * y + coeffs.col(4) * z + coeffs.col(7);
            J.col(2) = coeffs.col(2) * x + coeffs.col(4) * y + 2.0 * coeffs.col(5) * z + coeffs.col(8);

            dx = J.inverse() * r;

            x -= dx(0);
            y -= dx(1);
            z -= dx(2);
        }

        (*solutions)(0, i) = x;
        (*solutions)(1, i) = y;
        (*solutions)(2, i) = z;
    }
}

/*
 * Order of coefficients is:  x^2, xy, xz, y^2, yz, z^2, x, y, z, 1.0;
 *
 */
int re3q3(const Eigen::Matrix<double, 3, 10> &coeffs, Eigen::Matrix<double, 3, 8> *solutions,
          bool try_random_var_change) {

    Eigen::Matrix<double, 3, 3> Ax, Ay, Az;
    Ax << coeffs.col(3), coeffs.col(5), coeffs.col(4); // y^2, z^2, yz
    Ay << coeffs.col(0), coeffs.col(5), coeffs.col(2); // x^2, z^2, xz
    Az << coeffs.col(3), coeffs.col(0), coeffs.col(1); // y^2, x^2, yx

    // We check det(A) as a cheaper proxy for condition number
    int elim_var = 0;
    double detx = std::abs(Ax.determinant());
    double dety = std::abs(Ay.determinant());
    double detz = std::abs(Az.determinant());
    double det = detx;
    if (det < dety) {
        det = dety;
        elim_var = 1;
    }
    if (det < detz) {
        det = detz;
        elim_var = 2;
    }

    if (try_random_var_change && det < 1e-10) {
        Eigen::Matrix<double, 3, 4> A;
        A.block<3, 3>(0, 0) = Eigen::Quaternion<double>::UnitRandom().toRotationMatrix();
        A.block<3, 1>(0, 3).setRandom().normalize();

        Eigen::Matrix<double, 10, 10> B;
        B << A(0, 0) * A(0, 0), 2 * A(0, 0) * A(0, 1), 2 * A(0, 0) * A(0, 2), A(0, 1) * A(0, 1), 2 * A(0, 1) * A(0, 2),
            A(0, 2) * A(0, 2), 2 * A(0, 0) * A(0, 3), 2 * A(0, 1) * A(0, 3), 2 * A(0, 2) * A(0, 3), A(0, 3) * A(0, 3),
            A(0, 0) * A(1, 0), A(0, 0) * A(1, 1) + A(0, 1) * A(1, 0), A(0, 0) * A(1, 2) + A(0, 2) * A(1, 0),
            A(0, 1) * A(1, 1), A(0, 1) * A(1, 2) + A(0, 2) * A(1, 1), A(0, 2) * A(1, 2),
            A(0, 0) * A(1, 3) + A(0, 3) * A(1, 0), A(0, 1) * A(1, 3) + A(0, 3) * A(1, 1),
            A(0, 2) * A(1, 3) + A(0, 3) * A(1, 2), A(0, 3) * A(1, 3), A(0, 0) * A(2, 0),
            A(0, 0) * A(2, 1) + A(0, 1) * A(2, 0), A(0, 0) * A(2, 2) + A(0, 2) * A(2, 0), A(0, 1) * A(2, 1),
            A(0, 1) * A(2, 2) + A(0, 2) * A(2, 1), A(0, 2) * A(2, 2), A(0, 0) * A(2, 3) + A(0, 3) * A(2, 0),
            A(0, 1) * A(2, 3) + A(0, 3) * A(2, 1), A(0, 2) * A(2, 3) + A(0, 3) * A(2, 2), A(0, 3) * A(2, 3),
            A(1, 0) * A(1, 0), 2 * A(1, 0) * A(1, 1), 2 * A(1, 0) * A(1, 2), A(1, 1) * A(1, 1), 2 * A(1, 1) * A(1, 2),
            A(1, 2) * A(1, 2), 2 * A(1, 0) * A(1, 3), 2 * A(1, 1) * A(1, 3), 2 * A(1, 2) * A(1, 3), A(1, 3) * A(1, 3),
            A(1, 0) * A(2, 0), A(1, 0) * A(2, 1) + A(1, 1) * A(2, 0), A(1, 0) * A(2, 2) + A(1, 2) * A(2, 0),
            A(1, 1) * A(2, 1), A(1, 1) * A(2, 2) + A(1, 2) * A(2, 1), A(1, 2) * A(2, 2),
            A(1, 0) * A(2, 3) + A(1, 3) * A(2, 0), A(1, 1) * A(2, 3) + A(1, 3) * A(2, 1),
            A(1, 2) * A(2, 3) + A(1, 3) * A(2, 2), A(1, 3) * A(2, 3), A(2, 0) * A(2, 0), 2 * A(2, 0) * A(2, 1),
            2 * A(2, 0) * A(2, 2), A(2, 1) * A(2, 1), 2 * A(2, 1) * A(2, 2), A(2, 2) * A(2, 2), 2 * A(2, 0) * A(2, 3),
            2 * A(2, 1) * A(2, 3), 2 * A(2, 2) * A(2, 3), A(2, 3) * A(2, 3), 0, 0, 0, 0, 0, 0, A(0, 0), A(0, 1),
            A(0, 2), A(0, 3), 0, 0, 0, 0, 0, 0, A(1, 0), A(1, 1), A(1, 2), A(1, 3), 0, 0, 0, 0, 0, 0, A(2, 0), A(2, 1),
            A(2, 2), A(2, 3), 0, 0, 0, 0, 0, 0, 0, 0, 0, 1;
        Eigen::Matrix<double, 3, 10> coeffsB = coeffs * B;

        int n_sols = re3q3(coeffsB, solutions, false);

        // Revert change of variables
        for (int k = 0; k < n_sols; k++) {
            solutions->col(k) = A.block<3, 3>(0, 0) * solutions->col(k) + A.col(3);
        }

        // In some cases the numerics are quite poor after the change of variables, so we do some newton steps with the
        // original coefficients.
        refine_3q3(coeffs, solutions, n_sols);

        return n_sols;
    }

    Eigen::Matrix<double, 3, 7> P;

    if (elim_var == 0) {
        // re-order columns to eliminate x (target: y^2 z^2 yz x^2 xy xz x y z 1)
        P << coeffs.col(0), coeffs.col(1), coeffs.col(2), coeffs.col(6), coeffs.col(7), coeffs.col(8), coeffs.col(9);
        P = -Ax.inverse() * P;
    } else if (elim_var == 1) {
        // re-order columns to eliminate y (target: x^2 z^2 xz y^2 xy yz y x z 1)
        P << coeffs.col(3), coeffs.col(1), coeffs.col(4), coeffs.col(7), coeffs.col(6), coeffs.col(8), coeffs.col(9);
        P = -Ay.inverse() * P;
    } else if (elim_var == 2) {
        // re-order columns to eliminate z (target: y^2 x^2 yx z^2 zy z y x 1)
        P << coeffs.col(5), coeffs.col(4), coeffs.col(2), coeffs.col(8), coeffs.col(7), coeffs.col(6), coeffs.col(9);
        P = -Az.inverse() * P;
    }

    double a11 = P(0, 1) * P(2, 1) + P(0, 2) * P(1, 1) - P(2, 1) * P(0, 1) - P(2, 2) * P(2, 1) - P(2, 0);
    double a12 = P(0, 1) * P(2, 4) + P(0, 4) * P(2, 1) + P(0, 2) * P(1, 4) + P(0, 5) * P(1, 1) - P(2, 1) * P(0, 4) -
                 P(2, 4) * P(0, 1) - P(2, 2) * P(2, 4) - P(2, 5) * P(2, 1) - P(2, 3);
    double a13 = P(0, 4) * P(2, 4) + P(0, 5) * P(1, 4) - P(2, 4) * P(0, 4) - P(2, 5) * P(2, 4) - P(2, 6);
    double a14 = P(0, 1) * P(2, 2) + P(0, 2) * P(1, 2) - P(2, 1) * P(0, 2) - P(2, 2) * P(2, 2) + P(0, 0);
    double a15 = P(0, 1) * P(2, 5) + P(0, 4) * P(2, 2) + P(0, 2) * P(1, 5) + P(0, 5) * P(1, 2) - P(2, 1) * P(0, 5) -
                 P(2, 4) * P(0, 2) - P(2, 2) * P(2, 5) - P(2, 5) * P(2, 2) + P(0, 3);
    double a16 = P(0, 4) * P(2, 5) + P(0, 5) * P(1, 5) - P(2, 4) * P(0, 5) - P(2, 5) * P(2, 5) + P(0, 6);
    double a17 = P(0, 1) * P(2, 0) + P(0, 2) * P(1, 0) - P(2, 1) * P(0, 0) - P(2, 2) * P(2, 0);
    double a18 = P(0, 1) * P(2, 3) + P(0, 4) * P(2, 0) + P(0, 2) * P(1, 3) + P(0, 5) * P(1, 0) - P(2, 1) * P(0, 3) -
                 P(2, 4) * P(0, 0) - P(2, 2) * P(2, 3) - P(2, 5) * P(2, 0);
    double a19 = P(0, 1) * P(2, 6) + P(0, 4) * P(2, 3) + P(0, 2) * P(1, 6) + P(0, 5) * P(1, 3) - P(2, 1) * P(0, 6) -
                 P(2, 4) * P(0, 3) - P(2, 2) * P(2, 6) - P(2, 5) * P(2, 3);
    double a110 = P(0, 4) * P(2, 6) + P(0, 5) * P(1, 6) - P(2, 4) * P(0, 6) - P(2, 5) * P(2, 6);

    double a21 = P(2, 1) * P(2, 1) + P(2, 2) * P(1, 1) - P(1, 1) * P(0, 1) - P(1, 2) * P(2, 1) - P(1, 0);
    double a22 = P(2, 1) * P(2, 4) + P(2, 4) * P(2, 1) + P(2, 2) * P(1, 4) + P(2, 5) * P(1, 1) - P(1, 1) * P(0, 4) -
                 P(1, 4) * P(0, 1) - P(1, 2) * P(2, 4) - P(1, 5) * P(2, 1) - P(1, 3);
    double a23 = P(2, 4) * P(2, 4) + P(2, 5) * P(1, 4) - P(1, 4) * P(0, 4) - P(1, 5) * P(2, 4) - P(1, 6);
    double a24 = P(2, 1) * P(2, 2) + P(2, 2) * P(1, 2) - P(1, 1) * P(0, 2) - P(1, 2) * P(2, 2) + P(2, 0);
    double a25 = P(2, 1) * P(2, 5) + P(2, 4) * P(2, 2) + P(2, 2) * P(1, 5) + P(2, 5) * P(1, 2) - P(1, 1) * P(0, 5) -
                 P(1, 4) * P(0, 2) - P(1, 2) * P(2, 5) - P(1, 5) * P(2, 2) + P(2, 3);
    double a26 = P(2, 4) * P(2, 5) + P(2, 5) * P(1, 5) - P(1, 4) * P(0, 5) - P(1, 5) * P(2, 5) + P(2, 6);
    double a27 = P(2, 1) * P(2, 0) + P(2, 2) * P(1, 0) - P(1, 1) * P(0, 0) - P(1, 2) * P(2, 0);
    double a28 = P(2, 1) * P(2, 3) + P(2, 4) * P(2, 0) + P(2, 2) * P(1, 3) + P(2, 5) * P(1, 0) - P(1, 1) * P(0, 3) -
                 P(1, 4) * P(0, 0) - P(1, 2) * P(2, 3) - P(1, 5) * P(2, 0);
    double a29 = P(2, 1) * P(2, 6) + P(2, 4) * P(2, 3) + P(2, 2) * P(1, 6) + P(2, 5) * P(1, 3) - P(1, 1) * P(0, 6) -
                 P(1, 4) * P(0, 3) - P(1, 2) * P(2, 6) - P(1, 5) * P(2, 3);
    double a210 = P(2, 4) * P(2, 6) + P(2, 5) * P(1, 6) - P(1, 4) * P(0, 6) - P(1, 5) * P(2, 6);

    double t2 = P(2, 1) * P(2, 1);
    double t3 = P(2, 2) * P(2, 2);
    double t4 = P(0, 1) * P(1, 4);
    double t5 = P(0, 4) * P(1, 1);
    double t6 = t4 + t5;
    double t7 = P(0, 2) * P(1, 5);
    double t8 = P(0, 5) * P(1, 2);
    double t9 = t7 + t8;
    double t10 = P(0, 1) * P(1, 5);
    double t11 = P(0, 4) * P(1, 2);
    double t12 = t10 + t11;
    double t13 = P(0, 2) * P(1, 4);
    double t14 = P(0, 5) * P(1, 1);
    double t15 = t13 + t14;
    double t16 = P(2, 1) * P(2, 5);
    double t17 = P(2, 2) * P(2, 4);
    double t18 = t16 + t17;
    double t19 = P(2, 4) * P(2, 4);
    double t20 = P(2, 5) * P(2, 5);
    double a31 = P(0, 0) * P(1, 1) + P(0, 1) * P(1, 0) - P(2, 0) * P(2, 1) * 2.0 - P(0, 1) * t2 - P(1, 1) * t3 -
                 P(2, 2) * t2 * 2.0 + (P(0, 1) * P(0, 1)) * P(1, 1) + P(0, 2) * P(1, 1) * P(1, 2) +
                 P(0, 1) * P(1, 2) * P(2, 1) + P(0, 2) * P(1, 1) * P(2, 1);
    double a32 = P(0, 0) * P(1, 4) + P(0, 1) * P(1, 3) + P(0, 3) * P(1, 1) + P(0, 4) * P(1, 0) -
                 P(2, 0) * P(2, 4) * 2.0 - P(2, 1) * P(2, 3) * 2.0 - P(0, 4) * t2 + P(0, 1) * t6 - P(1, 4) * t3 +
                 P(1, 1) * t9 + P(2, 1) * t12 + P(2, 1) * t15 - P(2, 1) * t18 * 2.0 + P(0, 1) * P(0, 4) * P(1, 1) +
                 P(0, 2) * P(1, 2) * P(1, 4) + P(0, 1) * P(1, 2) * P(2, 4) + P(0, 2) * P(1, 1) * P(2, 4) -
                 P(0, 1) * P(2, 1) * P(2, 4) * 2.0 - P(1, 1) * P(2, 2) * P(2, 5) * 2.0 -
                 P(2, 1) * P(2, 2) * P(2, 4) * 2.0;
    double a33 = P(0, 1) * P(1, 6) + P(0, 3) * P(1, 4) + P(0, 4) * P(1, 3) + P(0, 6) * P(1, 1) -
                 P(2, 1) * P(2, 6) * 2.0 - P(2, 3) * P(2, 4) * 2.0 + P(0, 4) * t6 - P(0, 1) * t19 + P(1, 4) * t9 -
                 P(1, 1) * t20 + P(2, 4) * t12 + P(2, 4) * t15 - P(2, 4) * t18 * 2.0 + P(0, 1) * P(0, 4) * P(1, 4) +
                 P(0, 5) * P(1, 1) * P(1, 5) + P(0, 4) * P(1, 5) * P(2, 1) + P(0, 5) * P(1, 4) * P(2, 1) -
                 P(0, 4) * P(2, 1) * P(2, 4) * 2.0 - P(1, 4) * P(2, 2) * P(2, 5) * 2.0 -
                 P(2, 1) * P(2, 4) * P(2, 5) * 2.0;
    double a34 = P(0, 4) * P(1, 6) + P(0, 6) * P(1, 4) - P(2, 4) * P(2, 6) * 2.0 - P(0, 4) * t19 - P(1, 4) * t20 -
                 P(2, 5) * t19 * 2.0 + (P(0, 4) * P(0, 4)) * P(1, 4) + P(0, 5) * P(1, 4) * P(1, 5) +
                 P(0, 4) * P(1, 5) * P(2, 4) + P(0, 5) * P(1, 4) * P(2, 4);
    double a35 = P(0, 0) * P(1, 2) + P(0, 2) * P(1, 0) - P(2, 0) * P(2, 2) * 2.0 - P(0, 2) * t2 - P(1, 2) * t3 -
                 P(2, 1) * t3 * 2.0 + P(0, 2) * (P(1, 2) * P(1, 2)) + P(0, 1) * P(0, 2) * P(1, 1) +
                 P(0, 1) * P(1, 2) * P(2, 2) + P(0, 2) * P(1, 1) * P(2, 2);
    double a36 = P(0, 0) * P(1, 5) + P(0, 2) * P(1, 3) + P(0, 3) * P(1, 2) + P(0, 5) * P(1, 0) -
                 P(2, 0) * P(2, 5) * 2.0 - P(2, 2) * P(2, 3) * 2.0 - P(0, 5) * t2 + P(0, 2) * t6 - P(1, 5) * t3 +
                 P(1, 2) * t9 + P(2, 2) * t12 + P(2, 2) * t15 - P(2, 2) * t18 * 2.0 + P(0, 1) * P(0, 5) * P(1, 1) +
                 P(0, 2) * P(1, 2) * P(1, 5) + P(0, 1) * P(1, 2) * P(2, 5) + P(0, 2) * P(1, 1) * P(2, 5) -
                 P(0, 2) * P(2, 1) * P(2, 4) * 2.0 - P(1, 2) * P(2, 2) * P(2, 5) * 2.0 -
                 P(2, 1) * P(2, 2) * P(2, 5) * 2.0;
    double a37 = P(0, 2) * P(1, 6) + P(0, 3) * P(1, 5) + P(0, 5) * P(1, 3) + P(0, 6) * P(1, 2) -
                 P(2, 2) * P(2, 6) * 2.0 - P(2, 3) * P(2, 5) * 2.0 + P(0, 5) * t6 - P(0, 2) * t19 + P(1, 5) * t9 -
                 P(1, 2) * t20 + P(2, 5) * t12 + P(2, 5) * t15 - P(2, 5) * t18 * 2.0 + P(0, 2) * P(0, 4) * P(1, 4) +
                 P(0, 5) * P(1, 2) * P(1, 5) + P(0, 4) * P(1, 5) * P(2, 2) + P(0, 5) * P(1, 4) * P(2, 2) -
                 P(0, 5) * P(2, 1) * P(2, 4) * 2.0 - P(1, 5) * P(2, 2) * P(2, 5) * 2.0 -
                 P(2, 2) * P(2, 4) * P(2, 5) * 2.0;
    double a38 = P(0, 5) * P(1, 6) + P(0, 6) * P(1, 5) - P(2, 5) * P(2, 6) * 2.0 - P(0, 5) * t19 - P(1, 5) * t20 -
                 P(2, 4) * t20 * 2.0 + P(0, 5) * (P(1, 5) * P(1, 5)) + P(0, 4) * P(0, 5) * P(1, 4) +
                 P(0, 4) * P(1, 5) * P(2, 5) + P(0, 5) * P(1, 4) * P(2, 5);
    double a39 = P(0, 0) * P(1, 0) - P(0, 0) * t2 - P(1, 0) * t3 - P(2, 0) * P(2, 0) + P(0, 0) * P(0, 1) * P(1, 1) +
                 P(0, 2) * P(1, 0) * P(1, 2) + P(0, 1) * P(1, 2) * P(2, 0) + P(0, 2) * P(1, 1) * P(2, 0) -
                 P(2, 0) * P(2, 1) * P(2, 2) * 2.0;
    double a310 = P(0, 0) * P(1, 3) + P(0, 3) * P(1, 0) - P(2, 0) * P(2, 3) * 2.0 - P(0, 3) * t2 + P(0, 0) * t6 -
                  P(1, 3) * t3 + P(1, 0) * t9 + P(2, 0) * t12 + P(2, 0) * t15 - P(2, 0) * t18 * 2.0 +
                  P(0, 1) * P(0, 3) * P(1, 1) + P(0, 2) * P(1, 2) * P(1, 3) + P(0, 1) * P(1, 2) * P(2, 3) +
                  P(0, 2) * P(1, 1) * P(2, 3) - P(0, 0) * P(2, 1) * P(2, 4) * 2.0 - P(1, 0) * P(2, 2) * P(2, 5) * 2.0 -
                  P(2, 1) * P(2, 2) * P(2, 3) * 2.0;
    double a311 = P(0, 0) * P(1, 6) + P(0, 3) * P(1, 3) + P(0, 6) * P(1, 0) - P(2, 0) * P(2, 6) * 2.0 - P(0, 6) * t2 +
                  P(0, 3) * t6 - P(0, 0) * t19 - P(1, 6) * t3 + P(1, 3) * t9 - P(1, 0) * t20 + P(2, 3) * t12 +
                  P(2, 3) * t15 - P(2, 3) * t18 * 2.0 - P(2, 3) * P(2, 3) + P(0, 0) * P(0, 4) * P(1, 4) +
                  P(0, 1) * P(0, 6) * P(1, 1) + P(0, 2) * P(1, 2) * P(1, 6) + P(0, 5) * P(1, 0) * P(1, 5) +
                  P(0, 1) * P(1, 2) * P(2, 6) + P(0, 2) * P(1, 1) * P(2, 6) + P(0, 4) * P(1, 5) * P(2, 0) +
                  P(0, 5) * P(1, 4) * P(2, 0) - P(0, 3) * P(2, 1) * P(2, 4) * 2.0 - P(1, 3) * P(2, 2) * P(2, 5) * 2.0 -
                  P(2, 0) * P(2, 4) * P(2, 5) * 2.0 - P(2, 1) * P(2, 2) * P(2, 6) * 2.0;
    double a312 = P(0, 3) * P(1, 6) + P(0, 6) * P(1, 3) - P(2, 3) * P(2, 6) * 2.0 + P(0, 6) * t6 - P(0, 3) * t19 +
                  P(1, 6) * t9 - P(1, 3) * t20 + P(2, 6) * t12 + P(2, 6) * t15 - P(2, 6) * t18 * 2.0 +
                  P(0, 3) * P(0, 4) * P(1, 4) + P(0, 5) * P(1, 3) * P(1, 5) + P(0, 4) * P(1, 5) * P(2, 3) +
                  P(0, 5) * P(1, 4) * P(2, 3) - P(0, 6) * P(2, 1) * P(2, 4) * 2.0 - P(1, 6) * P(2, 2) * P(2, 5) * 2.0 -
                  P(2, 3) * P(2, 4) * P(2, 5) * 2.0;
    double a313 = P(0, 6) * P(1, 6) - P(0, 6) * t19 - P(1, 6) * t20 - P(2, 6) * P(2, 6) + P(0, 4) * P(0, 6) * P(1, 4) +
                  P(0, 5) * P(1, 5) * P(1, 6) + P(0, 4) * P(1, 5) * P(2, 6) + P(0, 5) * P(1, 4) * P(2, 6) -
                  P(2, 4) * P(2, 5) * P(2, 6) * 2.0;

    // det(M(x))
    double c[9];
    c[8] = a14 * a27 * a31 - a17 * a24 * a31 - a11 * a27 * a35 + a17 * a21 * a35 + a11 * a24 * a39 - a14 * a21 * a39;
    c[7] = a14 * a27 * a32 + a14 * a28 * a31 + a15 * a27 * a31 - a17 * a24 * a32 - a17 * a25 * a31 - a18 * a24 * a31 -
           a11 * a27 * a36 - a11 * a28 * a35 - a12 * a27 * a35 + a17 * a21 * a36 + a17 * a22 * a35 + a18 * a21 * a35 +
           a11 * a25 * a39 + a12 * a24 * a39 - a14 * a22 * a39 - a15 * a21 * a39 + a11 * a24 * a310 - a14 * a21 * a310;
    c[6] = a14 * a27 * a33 + a14 * a28 * a32 + a14 * a29 * a31 + a15 * a27 * a32 + a15 * a28 * a31 + a16 * a27 * a31 -
           a17 * a24 * a33 - a17 * a25 * a32 - a17 * a26 * a31 - a18 * a24 * a32 - a18 * a25 * a31 - a19 * a24 * a31 -
           a11 * a27 * a37 - a11 * a28 * a36 - a11 * a29 * a35 - a12 * a27 * a36 - a12 * a28 * a35 - a13 * a27 * a35 +
           a17 * a21 * a37 + a17 * a22 * a36 + a17 * a23 * a35 + a18 * a21 * a36 + a18 * a22 * a35 + a19 * a21 * a35 +
           a11 * a26 * a39 + a12 * a25 * a39 + a13 * a24 * a39 - a14 * a23 * a39 - a15 * a22 * a39 - a16 * a21 * a39 +
           a11 * a24 * a311 + a11 * a25 * a310 + a12 * a24 * a310 - a14 * a21 * a311 - a14 * a22 * a310 -
           a15 * a21 * a310;
    c[5] = a14 * a27 * a34 + a14 * a28 * a33 + a14 * a29 * a32 + a15 * a27 * a33 + a15 * a28 * a32 + a15 * a29 * a31 +
           a16 * a27 * a32 + a16 * a28 * a31 - a17 * a24 * a34 - a17 * a25 * a33 - a17 * a26 * a32 - a18 * a24 * a33 -
           a18 * a25 * a32 - a18 * a26 * a31 - a19 * a24 * a32 - a19 * a25 * a31 - a11 * a27 * a38 - a11 * a28 * a37 -
           a11 * a29 * a36 - a12 * a27 * a37 - a12 * a28 * a36 - a12 * a29 * a35 - a13 * a27 * a36 - a13 * a28 * a35 +
           a17 * a21 * a38 + a17 * a22 * a37 + a17 * a23 * a36 + a18 * a21 * a37 + a18 * a22 * a36 + a18 * a23 * a35 +
           a19 * a21 * a36 + a19 * a22 * a35 + a12 * a26 * a39 + a13 * a25 * a39 - a15 * a23 * a39 - a16 * a22 * a39 -
           a24 * a31 * a110 + a21 * a35 * a110 + a14 * a31 * a210 - a11 * a35 * a210 + a11 * a24 * a312 +
           a11 * a25 * a311 + a11 * a26 * a310 + a12 * a24 * a311 + a12 * a25 * a310 + a13 * a24 * a310 -
           a14 * a21 * a312 - a14 * a22 * a311 - a14 * a23 * a310 - a15 * a21 * a311 - a15 * a22 * a310 -
           a16 * a21 * a310;
    c[4] = a14 * a28 * a34 + a14 * a29 * a33 + a15 * a27 * a34 + a15 * a28 * a33 + a15 * a29 * a32 + a16 * a27 * a33 +
           a16 * a28 * a32 + a16 * a29 * a31 - a17 * a25 * a34 - a17 * a26 * a33 - a18 * a24 * a34 - a18 * a25 * a33 -
           a18 * a26 * a32 - a19 * a24 * a33 - a19 * a25 * a32 - a19 * a26 * a31 - a11 * a28 * a38 - a11 * a29 * a37 -
           a12 * a27 * a38 - a12 * a28 * a37 - a12 * a29 * a36 - a13 * a27 * a37 - a13 * a28 * a36 - a13 * a29 * a35 +
           a17 * a22 * a38 + a17 * a23 * a37 + a18 * a21 * a38 + a18 * a22 * a37 + a18 * a23 * a36 + a19 * a21 * a37 +
           a19 * a22 * a36 + a19 * a23 * a35 + a13 * a26 * a39 - a16 * a23 * a39 - a24 * a32 * a110 - a25 * a31 * a110 +
           a21 * a36 * a110 + a22 * a35 * a110 + a14 * a32 * a210 + a15 * a31 * a210 - a11 * a36 * a210 -
           a12 * a35 * a210 + a11 * a24 * a313 + a11 * a25 * a312 + a11 * a26 * a311 + a12 * a24 * a312 +
           a12 * a25 * a311 + a12 * a26 * a310 + a13 * a24 * a311 + a13 * a25 * a310 - a14 * a21 * a313 -
           a14 * a22 * a312 - a14 * a23 * a311 - a15 * a21 * a312 - a15 * a22 * a311 - a15 * a23 * a310 -
           a16 * a21 * a311 - a16 * a22 * a310;
    c[3] = a14 * a29 * a34 + a15 * a28 * a34 + a15 * a29 * a33 + a16 * a27 * a34 + a16 * a28 * a33 + a16 * a29 * a32 -
           a17 * a26 * a34 - a18 * a25 * a34 - a18 * a26 * a33 - a19 * a24 * a34 - a19 * a25 * a33 - a19 * a26 * a32 -
           a11 * a29 * a38 - a12 * a28 * a38 - a12 * a29 * a37 - a13 * a27 * a38 - a13 * a28 * a37 - a13 * a29 * a36 +
           a17 * a23 * a38 + a18 * a22 * a38 + a18 * a23 * a37 + a19 * a21 * a38 + a19 * a22 * a37 + a19 * a23 * a36 -
           a24 * a33 * a110 - a25 * a32 * a110 - a26 * a31 * a110 + a21 * a37 * a110 + a22 * a36 * a110 +
           a23 * a35 * a110 + a14 * a33 * a210 + a15 * a32 * a210 + a16 * a31 * a210 - a11 * a37 * a210 -
           a12 * a36 * a210 - a13 * a35 * a210 + a11 * a25 * a313 + a11 * a26 * a312 + a12 * a24 * a313 +
           a12 * a25 * a312 + a12 * a26 * a311 + a13 * a24 * a312 + a13 * a25 * a311 + a13 * a26 * a310 -
           a14 * a22 * a313 - a14 * a23 * a312 - a15 * a21 * a313 - a15 * a22 * a312 - a15 * a23 * a311 -
           a16 * a21 * a312 - a16 * a22 * a311 - a16 * a23 * a310;
    c[2] = a15 * a29 * a34 + a16 * a28 * a34 + a16 * a29 * a33 - a18 * a26 * a34 - a19 * a25 * a34 - a19 * a26 * a33 -
           a12 * a29 * a38 - a13 * a28 * a38 - a13 * a29 * a37 + a18 * a23 * a38 + a19 * a22 * a38 + a19 * a23 * a37 -
           a24 * a34 * a110 - a25 * a33 * a110 - a26 * a32 * a110 + a21 * a38 * a110 + a22 * a37 * a110 +
           a23 * a36 * a110 + a14 * a34 * a210 + a15 * a33 * a210 + a16 * a32 * a210 - a11 * a38 * a210 -
           a12 * a37 * a210 - a13 * a36 * a210 + a11 * a26 * a313 + a12 * a25 * a313 + a12 * a26 * a312 +
           a13 * a24 * a313 + a13 * a25 * a312 + a13 * a26 * a311 - a14 * a23 * a313 - a15 * a22 * a313 -
           a15 * a23 * a312 - a16 * a21 * a313 - a16 * a22 * a312 - a16 * a23 * a311;
    c[1] = a16 * a29 * a34 - a19 * a26 * a34 - a13 * a29 * a38 + a19 * a23 * a38 - a25 * a34 * a110 - a26 * a33 * a110 +
           a22 * a38 * a110 + a23 * a37 * a110 + a15 * a34 * a210 + a16 * a33 * a210 - a12 * a38 * a210 -
           a13 * a37 * a210 + a12 * a26 * a313 + a13 * a25 * a313 + a13 * a26 * a312 - a15 * a23 * a313 -
           a16 * a22 * a313 - a16 * a23 * a312;
    c[0] = -a26 * a34 * a110 + a23 * a38 * a110 + a16 * a34 * a210 - a13 * a38 * a210 + a13 * a26 * a313 -
           a16 * a23 * a313;

    double roots[8];

    int n_roots = sturm::bisect_sturm<8>(c, roots);

    Eigen::Matrix<double, 3, 3> A;
    for (int i = 0; i < n_roots; ++i) {
        double xs1 = roots[i];
        double xs2 = xs1 * xs1;
        double xs3 = xs1 * xs2;
        double xs4 = xs1 * xs3;

        A << a11 * xs2 + a12 * xs1 + a13, a14 * xs2 + a15 * xs1 + a16, a17 * xs3 + a18 * xs2 + a19 * xs1 + a110,
            a21 * xs2 + a22 * xs1 + a23, a24 * xs2 + a25 * xs1 + a26, a27 * xs3 + a28 * xs2 + a29 * xs1 + a210,
            a31 * xs3 + a32 * xs2 + a33 * xs1 + a34, a35 * xs3 + a36 * xs2 + a37 * xs1 + a38,
            a39 * xs4 + a310 * xs3 + a311 * xs2 + a312 * xs1 + a313;

        (*solutions)(0, i) = xs1;
        (*solutions)(1, i) = (A(1, 2) * A(0, 1) - A(0, 2) * A(1, 1)) / (A(0, 0) * A(1, 1) - A(1, 0) * A(0, 1));
        (*solutions)(2, i) = (A(1, 2) * A(0, 0) - A(0, 2) * A(1, 0)) / (A(0, 1) * A(1, 0) - A(1, 1) * A(0, 0));
    }
    if (elim_var == 1) {
        solutions->row(0).swap(solutions->row(1));
    } else if (elim_var == 2) {
        solutions->row(0).swap(solutions->row(2));
    }

    refine_3q3(coeffs, solutions, n_roots);

    return n_roots;
}

inline int re3q3_rotation_impl(Eigen::Matrix<double, 3, 10> &Rcoeffs, Eigen::Matrix<double, 4, 8> *solutions,
                               bool try_random_var_change) {
    Eigen::Vector4d q0 = Eigen::Quaterniond::UnitRandom().coeffs();
    Eigen::Matrix3d R0 = quat_to_rotmat(q0);
    Rcoeffs.block<3, 3>(0, 0) = Rcoeffs.block<3, 3>(0, 0) * R0;
    Rcoeffs.block<3, 3>(0, 3) = Rcoeffs.block<3, 3>(0, 3) * R0;
    Rcoeffs.block<3, 3>(0, 6) = Rcoeffs.block<3, 3>(0, 6) * R0;

    Eigen::Matrix<double, 3, 10> coeffs;
    rotation_to_3q3(Rcoeffs, &coeffs);

    Eigen::Matrix<double, 3, 8> solutions_cayley;
    int n_sols = re3q3(coeffs, &solutions_cayley, try_random_var_change);

    for (int i = 0; i < n_sols; ++i) {
        Eigen::Vector4d q{1.0, solutions_cayley(0, i), solutions_cayley(1, i), solutions_cayley(2, i)};
        q.normalize();
        solutions->col(i) = quat_multiply(q0, q);
    }
    return n_sols;
}

int re3q3_rotation(const Eigen::Matrix<double, 3, 9> &Rcoeffs, Eigen::Matrix<double, 4, 8> *solutions,
                   bool try_random_var_change) {
    Eigen::Matrix<double, 3, 10> Rcoeffs_copy;
    Rcoeffs_copy.block<3, 9>(0, 0) = Rcoeffs;
    Rcoeffs_copy.block<3, 1>(0, 9).setZero();
    return re3q3_rotation_impl(Rcoeffs_copy, solutions, try_random_var_change);
}
int re3q3_rotation(const Eigen::Matrix<double, 3, 10> &Rcoeffs, Eigen::Matrix<double, 4, 8> *solutions,
                   bool try_random_var_change) {
    Eigen::Matrix<double, 3, 10> Rcoeffs_copy = Rcoeffs;
    return re3q3_rotation_impl(Rcoeffs_copy, solutions, try_random_var_change);
}

} // namespace re3q3
} // namespace poselib
