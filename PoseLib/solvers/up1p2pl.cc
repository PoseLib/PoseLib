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

#include "up1p2pl.h"

#include "PoseLib/misc/univariate.h"
#include "p3p_common.h"

namespace poselib {

int up1p2pl(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
            const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X0,
            const std::vector<Eigen::Vector3d> &V, CameraPoseVector *output) {

    Eigen::Matrix<double, 3, 2> X;
    X << X0[0] - Xp[0], X0[1] - Xp[0];

    Eigen::Vector3d z1 = xp[0].cross(x[0]);
    Eigen::Vector3d z2 = xp[0].cross(x[1]);

    Eigen::Vector3d z3 = V[0].cross(X.col(0));
    Eigen::Vector3d z4 = V[1].cross(X.col(1));

    double a[12];
    a[0] = V[0](0) * z1(0) + V[0](2) * z1(2);
    a[1] = V[0](2) * z1(0) - V[0](0) * z1(2);
    a[2] = V[0](1) * z1(1);

    a[3] = x[1](0) * z4(0) + x[1](2) * z4(2);
    a[4] = x[1](0) * z4(2) - x[1](2) * z4(0);
    a[5] = x[1](1) * z4(1);

    a[6] = V[1](0) * z2(0) + V[1](2) * z2(2);
    a[7] = V[1](2) * z2(0) - V[1](0) * z2(2);
    a[8] = V[1](1) * z2(1);

    a[9] = x[0](0) * z3(0) + x[0](2) * z3(2);
    a[10] = -x[0](2) * z3(0) + x[0](0) * z3(2);
    a[11] = x[0](1) * z3(1);

    double b[6];
    b[0] = a[0] * a[3] - a[6] * a[9];
    b[1] = a[0] * a[4] + a[1] * a[3] - a[6] * a[10] - a[7] * a[9];
    b[2] = a[0] * a[5] + a[2] * a[3] - a[6] * a[11] - a[8] * a[9];
    b[3] = a[1] * a[4] - a[7] * a[10];
    b[4] = a[1] * a[5] + a[2] * a[4] - a[7] * a[11] - a[8] * a[10];
    b[5] = a[2] * a[5] - a[8] * a[11];

    Eigen::Matrix3d D1, D2;
    // first conic
    D1 << b[5], b[2] / 2, b[4] / 2, b[2] / 2, b[0], b[1] / 2, b[4] / 2, b[1] / 2, b[3];
    // circle
    D2 << -1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0;

    Eigen::Matrix3d DX1, DX2;
    DX1 << D1.col(1).cross(D1.col(2)), D1.col(2).cross(D1.col(0)), D1.col(0).cross(D1.col(1));
    DX2 << D2.col(1).cross(D2.col(2)), D2.col(2).cross(D2.col(0)), D2.col(0).cross(D2.col(1));

    double k3 = D2.col(0).dot(DX2.col(0));
    double k2 = (D1.array() * DX2.array()).sum();
    double k1 = (D2.array() * DX1.array()).sum();
    double k0 = D1.col(0).dot(DX1.col(0));

    double k3_inv = 1.0 / k3;
    k2 *= k3_inv;
    k1 *= k3_inv;
    k0 *= k3_inv;

    double s;
    bool G = univariate::solve_cubic_single_real(k2, k1, k0, s);

    Eigen::Matrix3d C = D1 + s * D2;
    std::array<Eigen::Vector3d, 2> pq = compute_pq(C);

    output->clear();
    int n_sols = 0;
    for (int i = 0; i < 2; ++i) {
        // [p1 p2 p3] * [1; cos; sin] = 0
        double p1 = pq[i](0);
        double p2 = pq[i](1);
        double p3 = pq[i](2);

        bool switch_23 = std::abs(p3) <= std::abs(p2);

        if (switch_23) {
            double w0 = -p1 / p2;
            double w1 = -p3 / p2;
            // find intersections between line [p1 p2 p3] * [1; cos; sin] = 0 and circle sin^2+cos^2=1
            double ca = 1.0 / (w1 * w1 + 1.0);
            double cb = 2.0 * w0 * w1 * ca;
            double cc = (w0 * w0 - 1.0) * ca;
            double taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (double sq : taus) {
                double cq = w0 + w1 * sq;
                Eigen::Matrix3d R;
                R.setIdentity();
                R(0, 0) = cq;
                R(0, 2) = sq;
                R(2, 0) = -sq;
                R(2, 2) = cq;

                Eigen::Vector3d a = x[0].cross(R * V[0]);
                Eigen::Vector3d b = R * X.col(0);

                double alpha = -a.dot(b) / a.dot(xp[0]);
                Eigen::Vector3d t = alpha * xp[0] - R * Xp[0];
                output->emplace_back(R, t);
                ++n_sols;
            }
        } else {
            double w0 = -p1 / p3;
            double w1 = -p2 / p3;
            double ca = 1.0 / (w1 * w1 + 1);
            double cb = 2.0 * w0 * w1 * ca;
            double cc = (w0 * w0 - 1.0) * ca;

            double taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (double cq : taus) {
                double sq = w0 + w1 * cq;
                Eigen::Matrix3d R;
                R.setIdentity();
                R(0, 0) = cq;
                R(0, 2) = sq;
                R(2, 0) = -sq;
                R(2, 2) = cq;

                Eigen::Vector3d a = x[0].cross(R * V[0]);
                Eigen::Vector3d b = R * X.col(0);

                double alpha = -a.dot(b) / a.dot(xp[0]);
                Eigen::Vector3d t = alpha * xp[0] - R * Xp[0];
                output->emplace_back(R, t);
                ++n_sols;
            }
        }

        if (G && n_sols > 0)
            break;
    }
    return output->size();
}

} // namespace poselib