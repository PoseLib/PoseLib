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

#include "relpose_upright_planar_2pt.h"

#include "PoseLib/misc/essential.h"

inline bool recover_a_b(const Eigen::Matrix<double, 2, 2> &C, double cos2phi, double sin2phi, Eigen::Vector2d &a,
                        Eigen::Vector2d &b) {

    if (std::abs(cos2phi) >= 1.0)
        return false;

    const double inv_sq2 = 1.0 / std::sqrt(2.0);
    a << std::sqrt(1 + cos2phi) * inv_sq2, std::sqrt(1 - cos2phi) * inv_sq2;

    if (sin2phi < 0)
        a(1) = -a(1);

    b = C * a;

    return true;
}

int poselib::relpose_upright_planar_2pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                                        CameraPoseVector *output) {

    Eigen::Matrix<double, 2, 2> A, B, C;
    Eigen::Vector2d a, b;

    A << x2[0](1) * x1[0](0), -x2[0](1) * x1[0](2), x2[1](1) * x1[1](0), -x2[1](1) * x1[1](2);
    B << x2[0](0) * x1[0](1), x2[0](2) * x1[0](1), x2[1](0) * x1[1](1), x2[1](2) * x1[1](1);
    C = B.inverse() * A;

    // There is a bug in the paper here where the factor 2 is missing from beta;
    const double alpha = C.col(0).dot(C.col(0));
    const double beta = 2.0 * C.col(0).dot(C.col(1));
    const double gamma = C.col(1).dot(C.col(1));
    const double alphap = alpha - gamma;
    const double gammap = alpha + gamma - 2.0;
    double inv_norm = 1.0 / (alphap * alphap + beta * beta);
    const double disc2 = alphap * alphap + beta * beta - gammap * gammap;

    output->clear();
    if (disc2 < 0) {
        // Degenerate situation. In this case we return the closest non-degen solution
        // See equation (27) in the paper
        inv_norm = std::sqrt(inv_norm);
        if (gammap < 0)
            inv_norm = -inv_norm;

        if (recover_a_b(C, -alphap * inv_norm, -beta * inv_norm, a, b)) {
            b.normalize();
            motion_from_essential_planar(b(0), b(1), -a(0), a(1), x1, x2, output);
        }
        return output->size();
    }

    const double disc = std::sqrt(disc2);

    // First set of solutions
    if (recover_a_b(C, (-alphap * gammap + beta * disc) * inv_norm, (-beta * gammap - alphap * disc) * inv_norm, a,
                    b)) {
        motion_from_essential_planar(b(0), b(1), -a(0), a(1), x1, x2, output);
    }

    // Second set of solutions
    if (recover_a_b(C, (-alphap * gammap - beta * disc) * inv_norm, (-beta * gammap + alphap * disc) * inv_norm, a,
                    b)) {
        motion_from_essential_planar(b(0), b(1), -a(0), a(1), x1, x2, output);
    }

    return output->size();
}
