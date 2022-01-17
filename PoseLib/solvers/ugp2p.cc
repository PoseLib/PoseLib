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

#include "ugp2p.h"

#include "PoseLib/misc/univariate.h"

int poselib::ugp2p(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                   const std::vector<Eigen::Vector3d> &X, poselib::CameraPoseVector *output) {
    Eigen::Matrix<double, 4, 4> A;
    Eigen::Matrix<double, 4, 2> b;

    A << -x[0](2), 0, x[0](0), x[0](2) * (X[0](0) + p[0](0)) - x[0](0) * (X[0](2) + p[0](2)), 0, -x[0](2), x[0](1),
        -x[0](2) * (X[0](1) - p[0](1)) - x[0](1) * (X[0](2) + p[0](2)), -x[1](2), 0, x[1](0),
        x[1](2) * (X[1](0) + p[1](0)) - x[1](0) * (X[1](2) + p[1](2)), 0, -x[1](2), x[1](1),
        -x[1](2) * (X[1](1) - p[1](1)) - x[1](1) * (X[1](2) + p[1](2));
    b << -2 * X[0](0) * x[0](0) - 2 * X[0](2) * x[0](2), x[0](0) * (X[0](2) - p[0](2)) - x[0](2) * (X[0](0) - p[0](0)),
        -2 * X[0](0) * x[0](1), x[0](1) * (X[0](2) - p[0](2)) - x[0](2) * (X[0](1) - p[0](1)),
        -2 * X[1](0) * x[1](0) - 2 * X[1](2) * x[1](2), x[1](0) * (X[1](2) - p[1](2)) - x[1](2) * (X[1](0) - p[1](0)),
        -2 * X[1](0) * x[1](1), x[1](1) * (X[1](2) - p[1](2)) - x[1](2) * (X[1](1) - p[1](1));

    // b = A.partialPivLu().solve(b);
    b = A.inverse() * b;

    const double c2 = b(3, 0);
    const double c3 = b(3, 1);

    double qq[2];
    const int sols = univariate::solve_quadratic_real(1.0, c2, c3, qq);

    output->clear();
    for (int i = 0; i < sols; ++i) {
        CameraPose pose;

        const double q = qq[i];
        const double q2 = q * q;
        const double inv_norm = 1.0 / (1 + q2);
        const double cq = (1 - q2) * inv_norm;
        const double sq = 2 * q * inv_norm;

        Eigen::Matrix3d R;
        R.setIdentity();
        R(0, 0) = cq;
        R(0, 2) = sq;
        R(2, 0) = -sq;
        R(2, 2) = cq;

        Eigen::Vector3d t;
        t = b.block<3, 1>(0, 0) * q + b.block<3, 1>(0, 1);
        t *= -inv_norm;

        output->emplace_back(R, t);
    }
    return sols;
}
