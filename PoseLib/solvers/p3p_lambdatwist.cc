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

#include "p3p_lambdatwist.h"

#include "p3p_common.h"

namespace poselib {

// Computes the eigen decomposition of a 3x3 matrix given that one eigenvalue is zero.
void compute_eig3x3known0(const Eigen::Matrix3d &M, Eigen::Matrix3d &E, double &sig1, double &sig2) {

    // In the original paper there is a missing minus sign here (for M(0,0))
    double p1 = -M(0, 0) - M(1, 1) - M(2, 2);
    double p0 =
        -M(0, 1) * M(0, 1) - M(0, 2) * M(0, 2) - M(1, 2) * M(1, 2) + M(0, 0) * (M(1, 1) + M(2, 2)) + M(1, 1) * M(2, 2);

    double disc = std::sqrt(p1 * p1 / 4.0 - p0);
    double tmp = -p1 / 2.0;
    sig1 = tmp + disc;
    sig2 = tmp - disc;

    if (std::abs(sig1) < std::abs(sig2))
        std::swap(sig1, sig2);

    double c = sig1 * sig1 + M(0, 0) * M(1, 1) - sig1 * (M(0, 0) + M(1, 1)) - M(0, 1) * M(0, 1);
    double a1 = (sig1 * M(0, 2) + M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) / c;
    double a2 = (sig1 * M(1, 2) + M(0, 1) * M(0, 2) - M(0, 0) * M(1, 2)) / c;
    double n = 1.0 / std::sqrt(1 + a1 * a1 + a2 * a2);
    E.col(0) << a1 * n, a2 * n, n;

    c = sig2 * sig2 + M(0, 0) * M(1, 1) - sig2 * (M(0, 0) + M(1, 1)) - M(0, 1) * M(0, 1);
    a1 = (sig2 * M(0, 2) + M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) / c;
    a2 = (sig2 * M(1, 2) + M(0, 1) * M(0, 2) - M(0, 0) * M(1, 2)) / c;
    n = 1.0 / std::sqrt(1 + a1 * a1 + a2 * a2);
    E.col(1) << a1 * n, a2 * n, n;

    // This is never used so we don't compute it
    // E.col(2) = M.col(1).cross(M.col(2)).normalized();
}

// Solves for camera pose such that: lambda*x = R*X+t  with positive lambda.
int p3p_lambdatwist(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
                    std::vector<CameraPose> *output) {

    Eigen::Vector3d dX12 = X[0] - X[1];
    Eigen::Vector3d dX13 = X[0] - X[2];
    Eigen::Vector3d dX23 = X[1] - X[2];

    double a12 = dX12.squaredNorm();
    double b12 = x[0].dot(x[1]);

    double a13 = dX13.squaredNorm();
    double b13 = x[0].dot(x[2]);

    double a23 = dX23.squaredNorm();
    double b23 = x[1].dot(x[2]);

    double a23b12 = a23 * b12;
    double a12b23 = a12 * b23;
    double a23b13 = a23 * b13;
    double a13b23 = a13 * b23;

    Eigen::Matrix3d D1, D2;

    D1 << a23, -a23b12, 0.0, -a23b12, a23 - a12, a12b23, 0.0, a12b23, -a12;
    D2 << a23, 0.0, -a23b13, 0.0, -a13, a13b23, -a23b13, a13b23, a23 - a13;

    Eigen::Matrix3d DX1, DX2;
    DX1 << D1.col(1).cross(D1.col(2)), D1.col(2).cross(D1.col(0)), D1.col(0).cross(D1.col(1));
    DX2 << D2.col(1).cross(D2.col(2)), D2.col(2).cross(D2.col(0)), D2.col(0).cross(D2.col(1));

    // Coefficients of p(gamma) = det(D1 + gamma*D2)
    // In the original paper c2 and c1 are switched.
    double c3 = D2.col(0).dot(DX2.col(0));
    double c2 = (D1.array() * DX2.array()).sum();
    double c1 = (D2.array() * DX1.array()).sum();
    double c0 = D1.col(0).dot(DX1.col(0));

    // closed root solver for cubic root
    const double c3inv = 1.0 / c3;
    c2 *= c3inv;
    c1 *= c3inv;
    c0 *= c3inv;

    double a = c1 - c2 * c2 / 3.0;
    double b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    double c = b * b / 4.0 + a * a * a / 27.0;
    double gamma;
    if (c > 0) {
        c = std::sqrt(c);
        b *= -0.5;
        gamma = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
    } else {
        c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
        gamma = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
    }

    // We do a single newton step on the cubic equation
    double f = gamma * gamma * gamma + c2 * gamma * gamma + c1 * gamma + c0;
    double df = 3.0 * gamma * gamma + 2.0 * c2 * gamma + c1;
    gamma = gamma - f / df;

    Eigen::Matrix3d D0 = D1 + gamma * D2;

    Eigen::Matrix3d E;
    double sig1, sig2;

    compute_eig3x3known0(D0, E, sig1, sig2);

    double s = std::sqrt(-sig2 / sig1);
    double lambda1, lambda2, lambda3;
    CameraPose pose;
    output->clear();
    Eigen::Matrix3d XX;

    XX << dX12, dX13, dX12.cross(dX13);
    XX = XX.inverse().eval();

    Eigen::Vector3d v1, v2;
    Eigen::Matrix3d YY;

    const double TOL_DOUBLE_ROOT = 1e-12;

    for (int s_flip = 0; s_flip < 2; ++s_flip, s = -s) {
        // [u1 u2 u3] * [lambda1; lambda2; lambda3] = 0
        double u1 = E(0, 0) - s * E(0, 1);
        double u2 = E(1, 0) - s * E(1, 1);
        double u3 = E(2, 0) - s * E(2, 1);

        // here we run into trouble if u1 is zero,
        // so depending on which is larger, we solve for either lambda1 or lambda2
        // The case u1 = u2 = 0 is degenerate and can be ignored
        bool switch_12 = std::abs(u1) < std::abs(u2);

        double a, b, c, w0, w1;

        if (switch_12) {
            // solve for lambda2
            w0 = -u1 / u2;
            w1 = -u3 / u2;
            a = -a13 * w1 * w1 + 2 * a13b23 * w1 - a13 + a23;
            b = 2 * a13b23 * w0 - 2 * a23b13 - 2 * a13 * w0 * w1;
            c = -a13 * w0 * w0 + a23;

            double b2m4ac = b * b - 4.0 * a * c;

            // if b2m4ac is zero we have a double root
            // to handle this case we allow slightly negative discriminants here
            if (b2m4ac < -TOL_DOUBLE_ROOT)
                continue;
            // clip to zero here in case we have double root
            double sq = std::sqrt(std::max(0.0, b2m4ac));

            // first root of tau
            double tau = (b > 0) ? (2.0 * c) / (-b - sq) : (2.0 * c) / (-b + sq);

            for (int tau_flip = 0; tau_flip < 2; ++tau_flip, tau = c / (a * tau)) {
                if (tau > 0) {
                    lambda1 = std::sqrt(a13 / (tau * (tau - 2.0 * b13) + 1.0));
                    lambda3 = tau * lambda1;
                    lambda2 = w0 * lambda1 + w1 * lambda3;
                    // since tau > 0 and lambda1 > 0 we only need to check lambda2 here
                    if (lambda2 < 0)
                        continue;

                    refine_lambda(lambda1, lambda2, lambda3, a12, a13, a23, b12, b13, b23);
                    v1 = lambda1 * x[0] - lambda2 * x[1];
                    v2 = lambda1 * x[0] - lambda3 * x[2];
                    YY << v1, v2, v1.cross(v2);
                    Eigen::Matrix3d R = YY * XX;
                    output->emplace_back(R, lambda1 * x[0] - R * X[0]);
                }

                if (b2m4ac < TOL_DOUBLE_ROOT) {
                    // double root we can skip the second tau
                    break;
                }
            }

        } else {
            // Same as except we solve for lambda1 as a combination of lambda2 and lambda3
            // (default case in the paper)
            w0 = -u2 / u1;
            w1 = -u3 / u1;
            a = (a13 - a12) * w1 * w1 + 2.0 * a12 * b13 * w1 - a12;
            b = -2.0 * a13 * b12 * w1 + 2.0 * a12 * b13 * w0 - 2.0 * w0 * w1 * (a12 - a13);
            c = (a13 - a12) * w0 * w0 - 2.0 * a13 * b12 * w0 + a13;
            double b2m4ac = b * b - 4.0 * a * c;
            if (b2m4ac < -TOL_DOUBLE_ROOT)
                continue;
            double sq = std::sqrt(std::max(0.0, b2m4ac));
            double tau = (b > 0) ? (2.0 * c) / (-b - sq) : (2.0 * c) / (-b + sq);
            for (int tau_flip = 0; tau_flip < 2; ++tau_flip, tau = c / (a * tau)) {
                if (tau > 0) {
                    lambda2 = std::sqrt(a23 / (tau * (tau - 2.0 * b23) + 1.0));
                    lambda3 = tau * lambda2;
                    lambda1 = w0 * lambda2 + w1 * lambda3;

                    if (lambda1 < 0)
                        continue;
                    refine_lambda(lambda1, lambda2, lambda3, a12, a13, a23, b12, b13, b23);
                    v1 = lambda1 * x[0] - lambda2 * x[1];
                    v2 = lambda1 * x[0] - lambda3 * x[2];
                    YY << v1, v2, v1.cross(v2);
                    Eigen::Matrix3d R = YY * XX;
                    output->emplace_back(R, lambda1 * x[0] - R * X[0]);
                }
                if (b2m4ac < TOL_DOUBLE_ROOT) {
                    break;
                }
            }
        }
    }

    return output->size();
}

} // namespace poselib
