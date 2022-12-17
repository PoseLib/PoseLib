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

#include "ugp4pl.h"

#include "PoseLib/misc/qep.h"

int poselib::ugp4pl(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                    const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector3d> &V,
                    CameraPoseVector *output) {

    Eigen::Matrix<double, 4, 4> M, C, K;
    Eigen::Matrix<double, 3, 4> VX;

    for (int i = 0; i < 4; ++i)
        VX.col(i) = V[i].cross(X[i]);

    M(0, 0) = -V[0](1) * x[0](2) - V[0](2) * x[0](1);
    M(0, 1) = V[0](2) * x[0](0) - V[0](0) * x[0](2);
    M(0, 2) = V[0](0) * x[0](1) + V[0](1) * x[0](0);
    M(0, 3) = VX(1, 0) * x[0](1) - VX(0, 0) * x[0](0) - VX(2, 0) * x[0](2) + V[0](0) * p[0](1) * x[0](2) -
              V[0](0) * p[0](2) * x[0](1) + V[0](1) * p[0](0) * x[0](2) - V[0](1) * p[0](2) * x[0](0) +
              V[0](2) * p[0](0) * x[0](1) - V[0](2) * p[0](1) * x[0](0);
    M(1, 0) = -V[1](1) * x[1](2) - V[1](2) * x[1](1);
    M(1, 1) = V[1](2) * x[1](0) - V[1](0) * x[1](2);
    M(1, 2) = V[1](0) * x[1](1) + V[1](1) * x[1](0);
    M(1, 3) = VX(1, 1) * x[1](1) - VX(0, 1) * x[1](0) - VX(2, 1) * x[1](2) + V[1](0) * p[1](1) * x[1](2) -
              V[1](0) * p[1](2) * x[1](1) + V[1](1) * p[1](0) * x[1](2) - V[1](1) * p[1](2) * x[1](0) +
              V[1](2) * p[1](0) * x[1](1) - V[1](2) * p[1](1) * x[1](0);
    M(2, 0) = -V[2](1) * x[2](2) - V[2](2) * x[2](1);
    M(2, 1) = V[2](2) * x[2](0) - V[2](0) * x[2](2);
    M(2, 2) = V[2](0) * x[2](1) + V[2](1) * x[2](0);
    M(2, 3) = VX(1, 2) * x[2](1) - VX(0, 2) * x[2](0) - VX(2, 2) * x[2](2) + V[2](0) * p[2](1) * x[2](2) -
              V[2](0) * p[2](2) * x[2](1) + V[2](1) * p[2](0) * x[2](2) - V[2](1) * p[2](2) * x[2](0) +
              V[2](2) * p[2](0) * x[2](1) - V[2](2) * p[2](1) * x[2](0);
    M(3, 0) = -V[3](1) * x[3](2) - V[3](2) * x[3](1);
    M(3, 1) = V[3](2) * x[3](0) - V[3](0) * x[3](2);
    M(3, 2) = V[3](0) * x[3](1) + V[3](1) * x[3](0);
    M(3, 3) = VX(1, 3) * x[3](1) - VX(0, 3) * x[3](0) - VX(2, 3) * x[3](2) + V[3](0) * p[3](1) * x[3](2) -
              V[3](0) * p[3](2) * x[3](1) + V[3](1) * p[3](0) * x[3](2) - V[3](1) * p[3](2) * x[3](0) +
              V[3](2) * p[3](0) * x[3](1) - V[3](2) * p[3](1) * x[3](0);

    C(0, 0) = -2 * V[0](0) * x[0](1);
    C(0, 1) = 2 * V[0](0) * x[0](0) + 2 * V[0](2) * x[0](2);
    C(0, 2) = -2 * V[0](2) * x[0](1);
    C(0, 3) = 2 * VX(2, 0) * x[0](0) - 2 * VX(0, 0) * x[0](2) + 2 * V[0](0) * p[0](0) * x[0](1) -
              2 * V[0](0) * p[0](1) * x[0](0) - 2 * V[0](2) * p[0](1) * x[0](2) + 2 * V[0](2) * p[0](2) * x[0](1);
    C(1, 0) = -2 * V[1](0) * x[1](1);
    C(1, 1) = 2 * V[1](0) * x[1](0) + 2 * V[1](2) * x[1](2);
    C(1, 2) = -2 * V[1](2) * x[1](1);
    C(1, 3) = 2 * VX(2, 1) * x[1](0) - 2 * VX(0, 1) * x[1](2) + 2 * V[1](0) * p[1](0) * x[1](1) -
              2 * V[1](0) * p[1](1) * x[1](0) - 2 * V[1](2) * p[1](1) * x[1](2) + 2 * V[1](2) * p[1](2) * x[1](1);
    C(2, 0) = -2 * V[2](0) * x[2](1);
    C(2, 1) = 2 * V[2](0) * x[2](0) + 2 * V[2](2) * x[2](2);
    C(2, 2) = -2 * V[2](2) * x[2](1);
    C(2, 3) = 2 * VX(2, 2) * x[2](0) - 2 * VX(0, 2) * x[2](2) + 2 * V[2](0) * p[2](0) * x[2](1) -
              2 * V[2](0) * p[2](1) * x[2](0) - 2 * V[2](2) * p[2](1) * x[2](2) + 2 * V[2](2) * p[2](2) * x[2](1);
    C(3, 0) = -2 * V[3](0) * x[3](1);
    C(3, 1) = 2 * V[3](0) * x[3](0) + 2 * V[3](2) * x[3](2);
    C(3, 2) = -2 * V[3](2) * x[3](1);
    C(3, 3) = 2 * VX(2, 3) * x[3](0) - 2 * VX(0, 3) * x[3](2) + 2 * V[3](0) * p[3](0) * x[3](1) -
              2 * V[3](0) * p[3](1) * x[3](0) - 2 * V[3](2) * p[3](1) * x[3](2) + 2 * V[3](2) * p[3](2) * x[3](1);

    K(0, 0) = V[0](2) * x[0](1) - V[0](1) * x[0](2);
    K(0, 1) = V[0](0) * x[0](2) - V[0](2) * x[0](0);
    K(0, 2) = V[0](1) * x[0](0) - V[0](0) * x[0](1);
    K(0, 3) = VX(0, 0) * x[0](0) + VX(1, 0) * x[0](1) + VX(2, 0) * x[0](2) - V[0](0) * p[0](1) * x[0](2) +
              V[0](0) * p[0](2) * x[0](1) + V[0](1) * p[0](0) * x[0](2) - V[0](1) * p[0](2) * x[0](0) -
              V[0](2) * p[0](0) * x[0](1) + V[0](2) * p[0](1) * x[0](0);
    K(1, 0) = V[1](2) * x[1](1) - V[1](1) * x[1](2);
    K(1, 1) = V[1](0) * x[1](2) - V[1](2) * x[1](0);
    K(1, 2) = V[1](1) * x[1](0) - V[1](0) * x[1](1);
    K(1, 3) = VX(0, 1) * x[1](0) + VX(1, 1) * x[1](1) + VX(2, 1) * x[1](2) - V[1](0) * p[1](1) * x[1](2) +
              V[1](0) * p[1](2) * x[1](1) + V[1](1) * p[1](0) * x[1](2) - V[1](1) * p[1](2) * x[1](0) -
              V[1](2) * p[1](0) * x[1](1) + V[1](2) * p[1](1) * x[1](0);
    K(2, 0) = V[2](2) * x[2](1) - V[2](1) * x[2](2);
    K(2, 1) = V[2](0) * x[2](2) - V[2](2) * x[2](0);
    K(2, 2) = V[2](1) * x[2](0) - V[2](0) * x[2](1);
    K(2, 3) = VX(0, 2) * x[2](0) + VX(1, 2) * x[2](1) + VX(2, 2) * x[2](2) - V[2](0) * p[2](1) * x[2](2) +
              V[2](0) * p[2](2) * x[2](1) + V[2](1) * p[2](0) * x[2](2) - V[2](1) * p[2](2) * x[2](0) -
              V[2](2) * p[2](0) * x[2](1) + V[2](2) * p[2](1) * x[2](0);
    K(3, 0) = V[3](2) * x[3](1) - V[3](1) * x[3](2);
    K(3, 1) = V[3](0) * x[3](2) - V[3](2) * x[3](0);
    K(3, 2) = V[3](1) * x[3](0) - V[3](0) * x[3](1);
    K(3, 3) = VX(0, 3) * x[3](0) + VX(1, 3) * x[3](1) + VX(2, 3) * x[3](2) - V[3](0) * p[3](1) * x[3](2) +
              V[3](0) * p[3](2) * x[3](1) + V[3](1) * p[3](0) * x[3](2) - V[3](1) * p[3](2) * x[3](0) -
              V[3](2) * p[3](0) * x[3](1) + V[3](2) * p[3](1) * x[3](0);

    /*
    Eigen::Matrix<double, 3, 8> eig_vecs;
    double eig_vals[8];
    const int n_roots = qep::qep_sturm(M, C, K, eig_vals, &eig_vecs);
    */
    // We know that (1+q^2) is a factor. Dividing by this gives degree 6 poly.
    Eigen::Matrix<double, 3, 6> eig_vecs;
    double eig_vals[6];
    const int n_roots = qep::qep_sturm_div_1_q2(M, C, K, eig_vals, &eig_vecs);

    output->clear();
    for (int i = 0; i < n_roots; ++i) {
        poselib::CameraPose pose;
        const double q = eig_vals[i];
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

        output->emplace_back(R, eig_vecs.col(i));
    }
    return n_roots;
}
