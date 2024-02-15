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

// Author: Yaqing Ding (yaq.ding@gmail.com).
// Some of the scripts are based on Mark Shachkov (mark.shachkov@gmail.com) and the Lambda-twist P3P implementation

#include "p3p_ding.h"

#include "p3p_common.h"

namespace poselib {

int p3p_ding(const std::vector<Eigen::Vector3d> &x_copy, const std::vector<Eigen::Vector3d> &X_copy,
             std::vector<CameraPose> *output) {
    if (output == nullptr) {
        return 0;
    }
    output->clear();
    output->reserve(4);

    Eigen::Vector3d X01 = X_copy[0] - X_copy[1];
    Eigen::Vector3d X02 = X_copy[0] - X_copy[2];
    Eigen::Vector3d X12 = X_copy[1] - X_copy[2];

    const double a12_copy = X01.squaredNorm();
    const double a13_copy = X02.squaredNorm();
    const double a23_copy = X12.squaredNorm();

    std::vector<Eigen::Vector3d> X = X_copy;
    std::vector<Eigen::Vector3d> x = x_copy;

    double a12;
    double a13;
    double a23;

    // Switch X,x so that BC is the largest distance among {X01, X02, X12}
    if (a12_copy > a13_copy) {
        if (a12_copy > a23_copy) {
            x[0] = x_copy[2];
            x[2] = x_copy[0];
            X[0] = X_copy[2];
            X[2] = X_copy[0];
            a12 = a23_copy;
            a13 = a13_copy;
            a23 = a12_copy;
            X01 = -X12;
            X02 = -X02;
        } else {
            a12 = a12_copy;
            a13 = a13_copy;
            a23 = a23_copy;
        }
    } else if (a13_copy > a23_copy) {
        x[0] = x_copy[1];
        x[1] = x_copy[0];
        X[0] = X_copy[1];
        X[1] = X_copy[0];
        a12 = a12_copy;
        a13 = a23_copy;
        a23 = a13_copy;
        X01 = -X01;
        X02 = X12;
    } else {
        a12 = a12_copy;
        a13 = a13_copy;
        a23 = a23_copy;
    }

    const double a23d = 1.0 / a23;
    const double a = a12 * a23d;
    const double b = a13 * a23d;

    const double m12 = x[0].dot(x[1]);
    const double m13 = x[0].dot(x[2]);
    const double m23 = x[1].dot(x[2]);

    // Ugly parameters to simplify the calculation
    const double m23sq = -m23 * m23 + 1.0;
    const double m13sq = -1.0 + m13 * m13;
    const double m12sq = -1.0 + m12 * m12;
    const double ab = a * b;
    const double bsq = b * b;
    const double asq = a * a;
    const double m123 = -2.0 + 2.0 * m12 * m13 * m23;
    const double bsqm23sq = bsq * m23sq;
    const double asqm23sq = asq * m23sq;
    const double abm23sq = 2.0 * ab * m23sq;

    const double k3_inv = 1.0 / (bsqm23sq + b * m13sq);
    const double k2 = k3_inv * ((-1.0 + a) * m13sq + abm23sq + bsqm23sq + b * m123);
    const double k1 = k3_inv * (asqm23sq + abm23sq + a * m123 + (-1.0 + b) * m12sq);
    const double k0 = k3_inv * (asqm23sq + a * m12sq);
    const double k22 = k2 * k2;
    const double alpha = k1 - 1.0 / 3.0 * k22;
    const double beta = k0 - 1.0 / 3.0 * k1 * k2 + (2.0 / 27.0) * k22 * k2;
    const double alpha3 = alpha * alpha * alpha / 27.0;
    const double G = beta * beta / 4.0 + alpha3;

    double s;
    if (G != 0) {
        if (G < 0) {
            s = cubic_trigonometric_solution(alpha, beta, k2, alpha3);
        } else {
            s = cubic_cardano_solution(beta, G, k2);
        }
    } else {
        s = -k2 / 3.0 + (alpha != 0 ? (3.0 * beta / alpha) : 0);
    }
    Eigen::Matrix3d C;
    C(0, 0) = -a + s * (1 - b);
    C(0, 1) = -m13 * s;
    C(0, 2) = a * m23 + b * m23 * s;
    C(1, 0) = C(0, 1);
    C(1, 1) = s + 1;
    C(1, 2) = -m12;
    C(2, 0) = C(0, 2);
    C(2, 1) = C(1, 2);
    C(2, 2) = -a - b * s + 1;

    std::array<Eigen::Vector3d, 2> pq = compute_pq(C);

    double d1, d2, d3;
    CameraPose pose;
    output->clear();
    Eigen::Matrix3d XX;

    XX << X01, X02, X01.cross(X02);
    XX = XX.inverse().eval();

    Eigen::Vector3d v1, v2;
    Eigen::Matrix3d YY;

    int n_sols = 0;

    for (int i = 0; i < 2; ++i) {
        // [p1 p2 p3] * [1; x; y] = 0, or [p1 p2 p3] * [d3; d1; d2] = 0
        double p1 = pq[i](0);
        double p2 = pq[i](1);
        double p3 = pq[i](2);
        // here we run into trouble if p1 is zero,
        // so depending on which is larger, we solve for either d1 or d2
        // The case p1 = p2 = 0 is degenerate and can be ignored
        bool switch_12 = std::abs(p1) <= std::abs(p2);

        if (switch_12) {
            // eliminate d1
            double w0 = -p1 / p2;
            double w1 = -p3 / p2;
            double ca = 1.0 / (w1 * w1 - b);
            double cb = 2.0 * (b * m23 - m13 * w1 + w0 * w1) * ca;
            double cc = (w0 * w0 - 2 * m13 * w0 - b + 1.0) * ca;
            double taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (double tau : taus) {
                if (tau <= 0)
                    continue;
                // positive only
                d3 = std::sqrt(a23 / (tau * (tau - 2.0 * m23) + 1.0));
                d2 = tau * d3;
                d1 = (w0 * d3 + w1 * d2);
                if (d1 < 0)
                    continue;

                refine_lambda(d1, d2, d3, a12, a13, a23, m12, m13, m23);
                v1 = d1 * x[0] - d2 * x[1];
                v2 = d1 * x[0] - d3 * x[2];
                YY << v1, v2, v1.cross(v2);
                Eigen::Matrix3d R = YY * XX;
                output->emplace_back(R, d1 * x[0] - R * X[0]);
                ++n_sols;
            }
        } else {
            double w0 = -p2 / p1;
            double w1 = -p3 / p1;
            double ca = 1.0 / (-a * w1 * w1 + 2 * a * m23 * w1 - a + 1);
            double cb = 2 * (a * m23 * w0 - m12 - a * w0 * w1) * ca;
            double cc = (1 - a * w0 * w0) * ca;

            double taus[2];
            if (!root2real(cb, cc, taus[0], taus[1]))
                continue;
            for (double tau : taus) {
                if (tau <= 0)
                    continue;
                d1 = std::sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0));
                d2 = tau * d1;
                d3 = w0 * d1 + w1 * d2;

                if (d3 < 0)
                    continue;

                refine_lambda(d1, d2, d3, a12, a13, a23, m12, m13, m23);
                v1 = d1 * x[0] - d2 * x[1];
                v2 = d1 * x[0] - d3 * x[2];
                YY << v1, v2, v1.cross(v2);
                Eigen::Matrix3d R = YY * XX;
                output->emplace_back(R, d1 * x[0] - R * X[0]);
                ++n_sols;
            }
        }

        if ((n_sols > 0 && G > 0))
            break;
    }

    return output->size();
}

} // namespace poselib
