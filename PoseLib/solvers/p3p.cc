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

#include "p3p.h"

#include "PoseLib/misc/univariate.h"
#include "p3p_common.h"

namespace poselib {

int p3p(const std::vector<Vector3> &x_copy, const std::vector<Vector3> &X_copy, std::vector<CameraPose> *output) {
    if (output == nullptr) {
        return 0;
    }
    output->clear();
    output->reserve(4);

    Vector3 X01 = X_copy[0] - X_copy[1];
    Vector3 X02 = X_copy[0] - X_copy[2];
    Vector3 X12 = X_copy[1] - X_copy[2];

    real a01 = X01.squaredNorm();
    real a02 = X02.squaredNorm();
    real a12 = X12.squaredNorm();

    std::array<Vector3, 3> X = {X_copy[0], X_copy[1], X_copy[2]};
    std::array<Vector3, 3> x = {x_copy[0], x_copy[1], x_copy[2]};

    // Switch X,x so that BC is the largest distance among {X01, X02, X12}
    if (a01 > a02) {
        if (a01 > a12) {
            std::swap(x[0], x[2]);
            std::swap(X[0], X[2]);
            std::swap(a01, a12);
            X01 = -X12;
            X02 = -X02;
        }
    } else if (a02 > a12) {
        std::swap(x[0], x[1]);
        std::swap(X[0], X[1]);
        std::swap(a02, a12);
        X01 = -X01;
        X02 = X12;
    }

    const real a12d = 1.0 / a12;
    const real a = a01 * a12d;
    const real b = a02 * a12d;

    const real m01 = x[0].dot(x[1]);
    const real m02 = x[0].dot(x[2]);
    const real m12 = x[1].dot(x[2]);

    // Ugly parameters to simplify the calculation
    const real m12sq = -m12 * m12 + 1.0;
    const real m02sq = -1.0 + m02 * m02;
    const real m01sq = -1.0 + m01 * m01;
    const real ab = a * b;
    const real bsq = b * b;
    const real asq = a * a;
    const real m013 = -2.0 + 2.0 * m01 * m02 * m12;
    const real bsqm12sq = bsq * m12sq;
    const real asqm12sq = asq * m12sq;
    const real abm12sq = 2.0 * ab * m12sq;

    const real k3_inv = 1.0 / (bsqm12sq + b * m02sq);
    const real k2 = k3_inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
    const real k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
    const real k0 = k3_inv * (asqm12sq + a * m01sq);

    real s;
    bool G = univariate::solve_cubic_single_real(k2, k1, k0, s);

    Matrix3x3 C;
    C(0, 0) = -a + s * (1 - b);
    C(0, 1) = -m02 * s;
    C(0, 2) = a * m12 + b * m12 * s;
    C(1, 0) = C(0, 1);
    C(1, 1) = s + 1;
    C(1, 2) = -m01;
    C(2, 0) = C(0, 2);
    C(2, 1) = C(1, 2);
    C(2, 2) = -a - b * s + 1;

    std::array<Vector3, 2> pq = compute_pq(C);

    real d0, d1, d2;
    CameraPose pose;
    output->clear();
    Matrix3x3 XX;

    XX << X01, X02, X01.cross(X02);
    XX = XX.inverse().eval();

    Vector3 v1, v2;
    Matrix3x3 YY;

    int n_sols = 0;

    for (int i = 0; i < 2; ++i) {
        // [p0 p1 p2] * [1; x; y] = 0, or [p0 p1 p2] * [d2; d0; d1] = 0
        real p0 = pq[i](0);
        real p1 = pq[i](1);
        real p2 = pq[i](2);
        // here we run into trouble if p0 is zero,
        // so depending on which is larger, we solve for either d0 or d1
        // The case p0 = p1 = 0 is degenerate and can be ignored
        bool switch_12 = std::abs(p0) <= std::abs(p1);

        if (switch_12) {
            // eliminate d0
            real w0 = -p0 / p1;
            real w1 = -p2 / p1;
            real ca = 1.0 / (w1 * w1 - b);
            real cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
            real cc = (w0 * w0 - 2 * m02 * w0 - b + 1.0) * ca;
            real taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (real tau : taus) {
                if (tau <= 0)
                    continue;
                // positive only
                d2 = std::sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0));
                d1 = tau * d2;
                d0 = (w0 * d2 + w1 * d1);
                if (d0 < 0)
                    continue;

                refine_lambda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
                v1 = d0 * x[0] - d1 * x[1];
                v2 = d0 * x[0] - d2 * x[2];
                YY << v1, v2, v1.cross(v2);
                Matrix3x3 R = YY * XX;
                output->emplace_back(R, d0 * x[0] - R * X[0]);
                ++n_sols;
            }
        } else {
            real w0 = -p1 / p0;
            real w1 = -p2 / p0;
            real ca = 1.0 / (-a * w1 * w1 + 2 * a * m12 * w1 - a + 1);
            real cb = 2 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
            real cc = (1 - a * w0 * w0) * ca;

            real taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (real tau : taus) {
                if (tau <= 0)
                    continue;
                d0 = std::sqrt(a01 / (tau * (tau - 2.0 * m01) + 1.0));
                d1 = tau * d0;
                d2 = w0 * d0 + w1 * d1;

                if (d2 < 0)
                    continue;

                refine_lambda(d0, d1, d2, a01, a02, a12, m01, m02, m12);
                v1 = d0 * x[0] - d1 * x[1];
                v2 = d0 * x[0] - d2 * x[2];
                YY << v1, v2, v1.cross(v2);
                Matrix3x3 R = YY * XX;
                output->emplace_back(R, d0 * x[0] - R * X[0]);
                ++n_sols;
            }
        }

        if (n_sols > 0 && G)
            break;
    }

    return output->size();
}

} // namespace poselib
