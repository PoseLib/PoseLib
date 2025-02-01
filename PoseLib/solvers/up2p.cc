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

#include "up2p.h"

#include "PoseLib/misc/univariate.h"

int poselib::up2p(const std::vector<Vector3> &x, const std::vector<Vector3> &X, poselib::CameraPoseVector *output) {
    Eigen::Matrix<Real, 4, 4> A;
    Eigen::Matrix<Real, 4, 2> b;

    A << -x[0](2), 0, x[0](0), X[0](0) * x[0](2) - X[0](2) * x[0](0), 0, -x[0](2), x[0](1),
        -X[0](1) * x[0](2) - X[0](2) * x[0](1), -x[1](2), 0, x[1](0), X[1](0) * x[1](2) - X[1](2) * x[1](0), 0,
        -x[1](2), x[1](1), -X[1](1) * x[1](2) - X[1](2) * x[1](1);
    b << -2 * X[0](0) * x[0](0) - 2 * X[0](2) * x[0](2), X[0](2) * x[0](0) - X[0](0) * x[0](2), -2 * X[0](0) * x[0](1),
        X[0](2) * x[0](1) - X[0](1) * x[0](2), -2 * X[1](0) * x[1](0) - 2 * X[1](2) * x[1](2),
        X[1](2) * x[1](0) - X[1](0) * x[1](2), -2 * X[1](0) * x[1](1), X[1](2) * x[1](1) - X[1](1) * x[1](2);

    // b = A.partialPivLu().solve(b);
    b = A.inverse() * b;

    const Real c2 = b(3, 0);
    const Real c3 = b(3, 1);

    Real qq[2];
    const int sols = univariate::solve_quadratic_real(1.0, c2, c3, qq);

    output->clear();
    for (int i = 0; i < sols; ++i) {
        const Real q = qq[i];
        const Real q2 = q * q;
        const Real inv_norm = 1.0 / (1 + q2);
        const Real cq = (1 - q2) * inv_norm;
        const Real sq = 2 * q * inv_norm;

        Matrix3x3 R;
        R.setIdentity();
        R(0, 0) = cq;
        R(0, 2) = sq;
        R(2, 0) = -sq;
        R(2, 2) = cq;

        Vector3 t;
        t = b.block<3, 1>(0, 0) * q + b.block<3, 1>(0, 1);
        t *= -inv_norm;

        output->emplace_back(R, t);
    }
    return sols;
}

int poselib::up2p(const std::vector<Vector3> &x, const std::vector<Vector3> &X, const Vector3 &g_cam,
                  const Vector3 &g_world, CameraPoseVector *output) {

    // Rotate camera world coordinate system
    Matrix3x3 Rc = Quaternion::FromTwoVectors(g_cam, Vector3::UnitY()).toRotationMatrix();
    Matrix3x3 Rw = Quaternion::FromTwoVectors(g_world, Vector3::UnitY()).toRotationMatrix();

    std::vector<Vector3> x_upright = x;
    std::vector<Vector3> X_upright = X;

    for (int i = 0; i < 2; ++i) {
        x_upright[i] = Rc * x[i];
        X_upright[i] = Rw * X[i];
    }

    int n_sols = up2p(x_upright, X_upright, output);

    // De-rotate coordinate systems
    for (int i = 0; i < n_sols; ++i) {
        Matrix3x3 R = (*output)[i].R();
        Vector3 t = (*output)[i].t;
        t = Rc.transpose() * t;
        R = Rc.transpose() * R * Rw;
        (*output)[i] = CameraPose(R, t);
    }
    return n_sols;
}
