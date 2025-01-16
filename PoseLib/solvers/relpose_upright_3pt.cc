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

// Author: Yaqing Ding
//         Mark Shachkov

#include "relpose_upright_3pt.h"

#include "PoseLib/misc/univariate.h"
#include "p3p_common.h"

namespace poselib {

int relpose_upright_3pt(const std::vector<Vector3> &x1, const std::vector<Vector3> &x2, CameraPoseVector *output) {

    Vector3 u1 = x1[0].cross(x1[1]);
    Vector3 v1 = x2[1].cross(x2[0]);
    Vector3 u2 = x1[0].cross(x1[2]);
    Vector3 v2 = x2[2].cross(x2[0]);

    real a[12];
    a[0] = u1(0);
    a[1] = u1(1);
    a[2] = u1(2);
    a[3] = x2[1](0);
    a[4] = x2[1](1);
    a[5] = x2[1](2);
    a[6] = v1(0);
    a[7] = v1(1);
    a[8] = v1(2);
    a[9] = x1[1](0);
    a[10] = x1[1](1);
    a[11] = x1[1](2);

    real b[12];
    b[0] = u2(0);
    b[1] = u2(1);
    b[2] = u2(2);
    b[3] = x2[2](0);
    b[4] = x2[2](1);
    b[5] = x2[2](2);
    b[6] = v2(0);
    b[7] = v2(1);
    b[8] = v2(2);
    b[9] = x1[2](0);
    b[10] = x1[2](1);
    b[11] = x1[2](2);

    real m[12];
    m[0] = a[1] * a[4];
    m[1] = a[0] * a[3] + a[2] * a[5];
    m[2] = a[2] * a[3] - a[0] * a[5];
    m[5] = a[8] * a[9] - a[6] * a[11];
    m[4] = -a[6] * a[9] - a[8] * a[11];
    m[3] = -a[7] * a[10];
    m[8] = b[2] * b[3] - b[0] * b[5];
    m[7] = b[0] * b[3] + b[2] * b[5];
    m[6] = b[1] * b[4];
    m[11] = b[8] * b[9] - b[6] * b[11];
    m[10] = -b[6] * b[9] - b[8] * b[11];
    m[9] = -b[7] * b[10];

    Matrix3x3 D1, D2;
    // first conic
    D1 << m[0] * m[9] - m[3] * m[6], (m[0] * m[10] + m[1] * m[9] - m[3] * m[7] - m[4] * m[6]) * 0.5,
        (m[0] * m[11] + m[2] * m[9] - m[3] * m[8] - m[5] * m[6]) * 0.5,
        (m[0] * m[10] + m[1] * m[9] - m[3] * m[7] - m[4] * m[6]) * 0.5, m[1] * m[10] - m[4] * m[7],
        (m[1] * m[11] + m[2] * m[10] - m[4] * m[8] - m[5] * m[7]) * 0.5,
        (m[0] * m[11] + m[2] * m[9] - m[3] * m[8] - m[5] * m[6]) * 0.5,
        (m[1] * m[11] + m[2] * m[10] - m[4] * m[8] - m[5] * m[7]) * 0.5, m[2] * m[11] - m[5] * m[8];
    // circle
    D2 << -1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0;

    Matrix3x3 DX1, DX2;
    DX1 << D1.col(1).cross(D1.col(2)), D1.col(2).cross(D1.col(0)), D1.col(0).cross(D1.col(1));
    DX2 << D2.col(1).cross(D2.col(2)), D2.col(2).cross(D2.col(0)), D2.col(0).cross(D2.col(1));

    real k3 = D2.col(0).dot(DX2.col(0));
    real k2 = (D1.array() * DX2.array()).sum();
    real k1 = (D2.array() * DX1.array()).sum();
    real k0 = D1.col(0).dot(DX1.col(0));

    real k3_inv = 1.0 / k3;
    k2 *= k3_inv;
    k1 *= k3_inv;
    k0 *= k3_inv;

    real s;
    bool G = univariate::solve_cubic_single_real(k2, k1, k0, s);

    Matrix3x3 C = D1 + s * D2;
    std::array<Vector3, 2> pq = compute_pq(C);

    int n_sols = 0;
    for (int i = 0; i < 2; ++i) {
        // [p1 p2 p3] * [1; cos; sin] = 0
        real p1 = pq[i](0);
        real p2 = pq[i](1);
        real p3 = pq[i](2);

        bool switch_23 = std::abs(p3) <= std::abs(p2);

        if (switch_23) {
            real w0 = -p1 / p2;
            real w1 = -p3 / p2;
            // find intersections between line [p1 p2 p3] * [1; cos; sin] = 0 and circle sin^2+cos^2=1
            real ca = 1.0 / (w1 * w1 + 1.0);
            real cb = 2.0 * w0 * w1 * ca;
            real cc = (w0 * w0 - 1.0) * ca;
            real taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (real sq : taus) {
                real cq = w0 + w1 * sq;
                real lambda = -(m[0] + m[1] * cq + m[2] * sq) / (m[3] + m[4] * cq + m[5] * sq);
                if (lambda < 0)
                    continue;

                Matrix3x3 R;
                R.setIdentity();
                R(0, 0) = cq;
                R(0, 2) = sq;
                R(2, 0) = -sq;
                R(2, 2) = cq;

                Vector3 trans;
                trans = lambda * x2[0] - R * x1[0];
                trans.normalize();
                CameraPose pose(R, trans);
                output->push_back(pose);
                ++n_sols;
            }
        } else {
            real w0 = -p1 / p3;
            real w1 = -p2 / p3;
            real ca = 1.0 / (w1 * w1 + 1);
            real cb = 2.0 * w0 * w1 * ca;
            real cc = (w0 * w0 - 1.0) * ca;

            real taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (real cq : taus) {
                real sq = w0 + w1 * cq;
                real lambda = -(m[0] + m[1] * cq + m[2] * sq) / (m[3] + m[4] * cq + m[5] * sq);
                if (lambda < 0)
                    continue;

                Matrix3x3 R;
                R.setIdentity();
                R(0, 0) = cq;
                R(0, 2) = sq;
                R(2, 0) = -sq;
                R(2, 2) = cq;

                Vector3 trans;
                trans = lambda * x2[0] - R * x1[0];
                trans.normalize();
                CameraPose pose(R, trans);
                output->push_back(pose);
                ++n_sols;
            }
        }

        if (n_sols > 0 && G)
            break;
    }

    return output->size();
}

int relpose_upright_3pt(const std::vector<Vector3> &x1, const std::vector<Vector3> &x2, const Vector3 &g_cam1,
                        const Vector3 &g_cam2, CameraPoseVector *output) {
    // Rotate camera world coordinate system
    Matrix3x3 Rc1 = Quaternion::FromTwoVectors(g_cam1, Vector3::UnitY()).toRotationMatrix();
    Matrix3x3 Rc2 = Quaternion::FromTwoVectors(g_cam2, Vector3::UnitY()).toRotationMatrix();

    std::vector<Vector3> x1_upright = x1;
    std::vector<Vector3> x2_upright = x2;

    for (int i = 0; i < 3; ++i) {
        x1_upright[i] = Rc1 * x1[i];
        x2_upright[i] = Rc2 * x2[i];
    }

    int n_sols = relpose_upright_3pt(x1_upright, x2_upright, output);

    // De-rotate coordinate systems
    for (int i = 0; i < n_sols; ++i) {
        Matrix3x3 R = (*output)[i].R();
        Vector3 t = (*output)[i].t;
        t = Rc2.transpose() * t;
        R = Rc2.transpose() * R * Rc1;
        (*output)[i] = CameraPose(R, t);
    }
    return n_sols;
}

} // namespace poselib
