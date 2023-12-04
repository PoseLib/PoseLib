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

#include "up1p1ll.h"

#include "PoseLib/misc/univariate.h"

int poselib::up1p1ll(const Eigen::Vector3d &xp, const Eigen::Vector3d &Xp, const Eigen::Vector3d &l,
                     const Eigen::Vector3d &X, const Eigen::Vector3d &V, CameraPoseVector *output) {

    const double c2 = V[1] * l[1] - V[0] * l[0] - V[2] * l[2];
    const double c1 = 2 * V[2] * l[0] - 2 * V[0] * l[2];
    const double c0 = V[0] * l[0] + V[1] * l[1] + V[2] * l[2];
    double qq[2];
    const int sols = univariate::solve_quadratic_real(c2, c1, c0, qq);

    Eigen::Matrix3d A;
    A.row(0) << xp(2), 0.0, -xp(0);
    A.row(1) << 0.0, xp(2), -xp(1);
    A.row(2) << l(0), l(1), l(2);

    Eigen::Matrix3d Ainv = A.inverse();

    output->clear();
    for (int i = 0; i < sols; ++i) {
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

        Eigen::Vector3d RXp = R * Xp;
        Eigen::Vector3d RX = R * X;
        Eigen::Vector3d b;
        b << A.row(0).dot(RXp), A.row(1).dot(RXp), A.row(2).dot(RX);
        Eigen::Vector3d t = -Ainv * b;

        output->emplace_back(R, t);
    }
    return sols;
}

int poselib::up1p1ll(const Eigen::Vector3d &xp, const Eigen::Vector3d &Xp, const Eigen::Vector3d &l,
                     const Eigen::Vector3d &X, const Eigen::Vector3d &V, const Eigen::Vector3d &g_cam,
                     const Eigen::Vector3d &g_world, CameraPoseVector *output) {
    // Rotate camera world coordinate system
    Eigen::Matrix3d Rc = Eigen::Quaterniond::FromTwoVectors(g_cam, Eigen::Vector3d::UnitY()).toRotationMatrix();
    Eigen::Matrix3d Rw = Eigen::Quaterniond::FromTwoVectors(g_world, Eigen::Vector3d::UnitY()).toRotationMatrix();

    Eigen::Vector3d xp_upright = Rc * xp;
    Eigen::Vector3d Xp_upright = Rw * Xp;

    Eigen::Vector3d l_upright = Rc * l;
    Eigen::Vector3d X_upright = Rw * X;
    Eigen::Vector3d V_upright = Rw * V;

    int n_sols = up1p1ll(xp_upright, Xp_upright, l_upright, X_upright, V_upright, output);

    // De-rotate coordinate systems
    for (int i = 0; i < n_sols; ++i) {
        Eigen::Matrix3d R = (*output)[i].R();
        Eigen::Vector3d t = (*output)[i].t;
        t = Rc.transpose() * t;
        R = Rc.transpose() * R * Rw;
        (*output)[i] = CameraPose(R, t);
    }
    return n_sols;
}
