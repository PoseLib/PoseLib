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

int poselib::up1p1ll(const Vector3 &xp, const Vector3 &Xp, const Vector3 &l, const Vector3 &X, const Vector3 &V,
                     CameraPoseVector *output) {

    const Real c2 = V[1] * l[1] - V[0] * l[0] - V[2] * l[2];
    const Real c1 = 2 * V[2] * l[0] - 2 * V[0] * l[2];
    const Real c0 = V[0] * l[0] + V[1] * l[1] + V[2] * l[2];
    Real qq[2];
    const int sols = univariate::solve_quadratic_real(c2, c1, c0, qq);

    Matrix3x3 A;
    A.row(0) << xp(2), 0.0, -xp(0);
    A.row(1) << 0.0, xp(2), -xp(1);
    A.row(2) << l(0), l(1), l(2);

    Matrix3x3 Ainv = A.inverse();

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

        Vector3 RXp = R * Xp;
        Vector3 RX = R * X;
        Vector3 b;
        b << A.row(0).dot(RXp), A.row(1).dot(RXp), A.row(2).dot(RX);
        Vector3 t = -Ainv * b;

        output->emplace_back(R, t);
    }
    return sols;
}

int poselib::up1p1ll(const Vector3 &xp, const Vector3 &Xp, const Vector3 &l, const Vector3 &X, const Vector3 &V,
                     const Vector3 &g_cam, const Vector3 &g_world, CameraPoseVector *output) {
    // Rotate camera world coordinate system
    Matrix3x3 Rc = Quaternion::FromTwoVectors(g_cam, Vector3::UnitY()).toRotationMatrix();
    Matrix3x3 Rw = Quaternion::FromTwoVectors(g_world, Vector3::UnitY()).toRotationMatrix();

    Vector3 xp_upright = Rc * xp;
    Vector3 Xp_upright = Rw * Xp;

    Vector3 l_upright = Rc * l;
    Vector3 X_upright = Rw * X;
    Vector3 V_upright = Rw * V;

    int n_sols = up1p1ll(xp_upright, Xp_upright, l_upright, X_upright, V_upright, output);

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
