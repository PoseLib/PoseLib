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

#include "p5lp_radial.h"

#include "PoseLib/misc/univariate.h"

namespace poselib {

int p5lp_radial(const std::vector<Vector3> &l, const std::vector<Vector3> &X, std::vector<CameraPose> *output) {
    std::vector<Vector2> x(5);
    for (size_t i = 0; i < 5; ++i) {
        x[i] << l[i](1), -l[i](0);
    }
    // Note: Assumes that l[i](2) = 0 !
    return p5lp_radial(x, X, output);
}

int p5lp_radial(const std::vector<Vector2> &x, const std::vector<Vector3> &X, std::vector<CameraPose> *output) {

    // Setup nullspace
    Eigen::Matrix<real, 8, 5> cc;
    for (int i = 0; i < 5; i++) {
        cc(0, i) = -x[i](1) * X[i](0);
        cc(1, i) = -x[i](1) * X[i](1);
        cc(2, i) = -x[i](1) * X[i](2);
        cc(3, i) = -x[i](1);
        cc(4, i) = x[i](0) * X[i](0);
        cc(5, i) = x[i](0) * X[i](1);
        cc(6, i) = x[i](0) * X[i](2);
        cc(7, i) = x[i](0);
    }

    Eigen::Matrix<real, 8, 8> Q = cc.householderQr().householderQ();
    Eigen::Matrix<real, 8, 3> N = Q.rightCols(3);

    // Compute coefficients for sylvester resultant
    real c11_1 = N(0, 1) * N(4, 1) + N(1, 1) * N(5, 1) + N(2, 1) * N(6, 1);
    real c12_1 = N(0, 1) * N(4, 2) + N(0, 2) * N(4, 1) + N(1, 1) * N(5, 2) + N(1, 2) * N(5, 1) + N(2, 1) * N(6, 2) +
                 N(2, 2) * N(6, 1);
    real c12_2 = N(0, 0) * N(4, 1) + N(0, 1) * N(4, 0) + N(1, 0) * N(5, 1) + N(1, 1) * N(5, 0) + N(2, 0) * N(6, 1) +
                 N(2, 1) * N(6, 0);
    real c13_1 = N(0, 2) * N(4, 2) + N(1, 2) * N(5, 2) + N(2, 2) * N(6, 2);
    real c13_2 = N(0, 0) * N(4, 2) + N(0, 2) * N(4, 0) + N(1, 0) * N(5, 2) + N(1, 2) * N(5, 0) + N(2, 0) * N(6, 2) +
                 N(2, 2) * N(6, 0);
    real c13_3 = N(0, 0) * N(4, 0) + N(1, 0) * N(5, 0) + N(2, 0) * N(6, 0);
    real c21_1 = N(0, 1) * N(0, 1) + N(1, 1) * N(1, 1) + N(2, 1) * N(2, 1) - N(4, 1) * N(4, 1) - N(5, 1) * N(5, 1) -
                 N(6, 1) * N(6, 1);
    real c22_1 = 2 * N(0, 1) * N(0, 2) + 2 * N(1, 1) * N(1, 2) + 2 * N(2, 1) * N(2, 2) - 2 * N(4, 1) * N(4, 2) -
                 2 * N(5, 1) * N(5, 2) - 2 * N(6, 1) * N(6, 2);
    real c22_2 = 2 * N(0, 0) * N(0, 1) + 2 * N(1, 0) * N(1, 1) + 2 * N(2, 0) * N(2, 1) - 2 * N(4, 0) * N(4, 1) -
                 2 * N(5, 0) * N(5, 1) - 2 * N(6, 0) * N(6, 1);
    real c23_1 = N(0, 2) * N(0, 2) + N(1, 2) * N(1, 2) + N(2, 2) * N(2, 2) - N(4, 2) * N(4, 2) - N(5, 2) * N(5, 2) -
                 N(6, 2) * N(6, 2);
    real c23_2 = 2 * N(0, 0) * N(0, 2) + 2 * N(1, 0) * N(1, 2) + 2 * N(2, 0) * N(2, 2) - 2 * N(4, 0) * N(4, 2) -
                 2 * N(5, 0) * N(5, 2) - 2 * N(6, 0) * N(6, 2);
    real c23_3 = N(0, 0) * N(0, 0) + N(1, 0) * N(1, 0) + N(2, 0) * N(2, 0) - N(4, 0) * N(4, 0) - N(5, 0) * N(5, 0) -
                 N(6, 0) * N(6, 0);

    real a4 = c11_1 * c11_1 * c23_3 * c23_3 - c11_1 * c12_2 * c22_2 * c23_3 - 2 * c11_1 * c13_3 * c21_1 * c23_3 +
              c11_1 * c13_3 * c22_2 * c22_2 + c12_2 * c12_2 * c21_1 * c23_3 - c12_2 * c13_3 * c21_1 * c22_2 +
              c13_3 * c13_3 * c21_1 * c21_1;
    real a3 = c11_1 * c13_2 * c22_2 * c22_2 + 2 * c13_2 * c13_3 * c21_1 * c21_1 + c12_2 * c12_2 * c21_1 * c23_2 +
              2 * c11_1 * c11_1 * c23_2 * c23_3 - c11_1 * c12_1 * c22_2 * c23_3 - c11_1 * c12_2 * c22_1 * c23_3 -
              c11_1 * c12_2 * c22_2 * c23_2 - 2 * c11_1 * c13_2 * c21_1 * c23_3 - 2 * c11_1 * c13_3 * c21_1 * c23_2 +
              2 * c11_1 * c13_3 * c22_1 * c22_2 + 2 * c12_1 * c12_2 * c21_1 * c23_3 - c12_1 * c13_3 * c21_1 * c22_2 -
              c12_2 * c13_2 * c21_1 * c22_2 - c12_2 * c13_3 * c21_1 * c22_1;
    real a2 = c11_1 * c11_1 * c23_2 * c23_2 + c13_2 * c13_2 * c21_1 * c21_1 + c11_1 * c13_1 * c22_2 * c22_2 +
              c11_1 * c13_3 * c22_1 * c22_1 + 2 * c13_1 * c13_3 * c21_1 * c21_1 + c12_2 * c12_2 * c21_1 * c23_1 +
              c12_1 * c12_1 * c21_1 * c23_3 + 2 * c11_1 * c11_1 * c23_1 * c23_3 - c11_1 * c12_1 * c22_1 * c23_3 -
              c11_1 * c12_1 * c22_2 * c23_2 - c11_1 * c12_2 * c22_1 * c23_2 - c11_1 * c12_2 * c22_2 * c23_1 -
              2 * c11_1 * c13_1 * c21_1 * c23_3 - 2 * c11_1 * c13_2 * c21_1 * c23_2 +
              2 * c11_1 * c13_2 * c22_1 * c22_2 - 2 * c11_1 * c13_3 * c21_1 * c23_1 +
              2 * c12_1 * c12_2 * c21_1 * c23_2 - c12_1 * c13_2 * c21_1 * c22_2 - c12_1 * c13_3 * c21_1 * c22_1 -
              c12_2 * c13_1 * c21_1 * c22_2 - c12_2 * c13_2 * c21_1 * c22_1;
    real a1 = c11_1 * c13_2 * c22_1 * c22_1 + 2 * c13_1 * c13_2 * c21_1 * c21_1 + c12_1 * c12_1 * c21_1 * c23_2 +
              2 * c11_1 * c11_1 * c23_1 * c23_2 - c11_1 * c12_1 * c22_1 * c23_2 - c11_1 * c12_1 * c22_2 * c23_1 -
              c11_1 * c12_2 * c22_1 * c23_1 - 2 * c11_1 * c13_1 * c21_1 * c23_2 + 2 * c11_1 * c13_1 * c22_1 * c22_2 -
              2 * c11_1 * c13_2 * c21_1 * c23_1 + 2 * c12_1 * c12_2 * c21_1 * c23_1 - c12_1 * c13_1 * c21_1 * c22_2 -
              c12_1 * c13_2 * c21_1 * c22_1 - c12_2 * c13_1 * c21_1 * c22_1;
    real a0 = c11_1 * c11_1 * c23_1 * c23_1 - c11_1 * c12_1 * c22_1 * c23_1 - 2 * c11_1 * c13_1 * c21_1 * c23_1 +
              c11_1 * c13_1 * c22_1 * c22_1 + c12_1 * c12_1 * c21_1 * c23_1 - c12_1 * c13_1 * c21_1 * c22_1 +
              c13_1 * c13_1 * c21_1 * c21_1;

    a4 = 1.0 / a4;
    a3 *= a4;
    a2 *= a4;
    a1 *= a4;
    a0 *= a4;

    real roots[4];

    int n_roots = univariate::solve_quartic_real(a3, a2, a1, a0, roots);

    CameraPose pose;
    output->clear();
    for (int i = 0; i < n_roots; i++) {
        // We have two quadratic polynomials in y after substituting x
        real a = roots[i];
        real c1a = c11_1;
        real c1b = c12_1 + c12_2 * a;
        real c1c = c13_1 + c13_2 * a + c13_3 * a * a;

        real c2a = c21_1;
        real c2b = c22_1 + c22_2 * a;
        real c2c = c23_1 + c23_2 * a + c23_3 * a * a;

        // we solve the first one
        real bb[2];
        if (!univariate::solve_quadratic_real(c1a, c1b, c1c, bb))
            continue;

        // and check the residuals of the other
        real res1 = c2a * bb[0] * bb[0] + c2b * bb[0] + c2c;
        real res2;

        // For data where X(3,:) = 0 there is only a single solution
        // In this case the second solution will be NaN
        if (std::isnan(bb[1]))
            res2 = std::numeric_limits<real>::max();
        else
            res2 = c2a * bb[1] * bb[1] + c2b * bb[1] + c2c;

        real b = (std::abs(res1) > std::abs(res2)) ? bb[1] : bb[0];

        Eigen::Matrix<real, 8, 1> p = N.col(0) * a + N.col(1) * b + N.col(2);

        Matrix3x3 R;
        R.row(0) << p(0), p(1), p(2);
        R.row(1) << p(4), p(5), p(6);
        Vector3 t(p(3), p(7), 0.0);

        real scale = R.row(0).norm();
        R.row(0) /= scale;
        R.row(1) /= scale;
        t /= scale;
        R.row(2) = R.row(0).cross(R.row(1));

        // Select sign using first point
        if ((R * X[0] + t).topRows<2>().dot(x[0]) < 0) {
            R.block<2, 3>(0, 0) = -R.block<2, 3>(0, 0);
            t = -t;
        }

        output->emplace_back(R, t);
    }
    return n_roots;
}

} // namespace poselib