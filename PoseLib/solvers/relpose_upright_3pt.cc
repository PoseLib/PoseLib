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

#include "relpose_upright_3pt.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/misc/qep.h"

namespace poselib {

int relpose_upright_3pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                        CameraPoseVector *output) {

    Eigen::Matrix<double, 3, 3> M, C, K;

    M(0, 0) = x2[0](1) * x1[0](2) + x2[0](2) * x1[0](1);
    M(0, 1) = x2[0](2) * x1[0](0) - x2[0](0) * x1[0](2);
    M(0, 2) = -x2[0](0) * x1[0](1) - x2[0](1) * x1[0](0);
    M(1, 0) = x2[1](1) * x1[1](2) + x2[1](2) * x1[1](1);
    M(1, 1) = x2[1](2) * x1[1](0) - x2[1](0) * x1[1](2);
    M(1, 2) = -x2[1](0) * x1[1](1) - x2[1](1) * x1[1](0);
    M(2, 0) = x2[2](1) * x1[2](2) + x2[2](2) * x1[2](1);
    M(2, 1) = x2[2](2) * x1[2](0) - x2[2](0) * x1[2](2);
    M(2, 2) = -x2[2](0) * x1[2](1) - x2[2](1) * x1[2](0);

    C(0, 0) = 2 * x2[0](1) * x1[0](0);
    C(0, 1) = -2 * x2[0](0) * x1[0](0) - 2 * x2[0](2) * x1[0](2);
    C(0, 2) = 2 * x2[0](1) * x1[0](2);
    C(1, 0) = 2 * x2[1](1) * x1[1](0);
    C(1, 1) = -2 * x2[1](0) * x1[1](0) - 2 * x2[1](2) * x1[1](2);
    C(1, 2) = 2 * x2[1](1) * x1[1](2);
    C(2, 0) = 2 * x2[2](1) * x1[2](0);
    C(2, 1) = -2 * x2[2](0) * x1[2](0) - 2 * x2[2](2) * x1[2](2);
    C(2, 2) = 2 * x2[2](1) * x1[2](2);

    K(0, 0) = x2[0](2) * x1[0](1) - x2[0](1) * x1[0](2);
    K(0, 1) = x2[0](0) * x1[0](2) - x2[0](2) * x1[0](0);
    K(0, 2) = x2[0](1) * x1[0](0) - x2[0](0) * x1[0](1);
    K(1, 0) = x2[1](2) * x1[1](1) - x2[1](1) * x1[1](2);
    K(1, 1) = x2[1](0) * x1[1](2) - x2[1](2) * x1[1](0);
    K(1, 2) = x2[1](1) * x1[1](0) - x2[1](0) * x1[1](1);
    K(2, 0) = x2[2](2) * x1[2](1) - x2[2](1) * x1[2](2);
    K(2, 1) = x2[2](0) * x1[2](2) - x2[2](2) * x1[2](0);
    K(2, 2) = x2[2](1) * x1[2](0) - x2[2](0) * x1[2](1);

    /*
    Eigen::Matrix<double, 3, 6> eig_vecs;
    double eig_vals[6];
    const int n_roots = qep::qep_sturm(M, C, K, eig_vals, &eig_vecs);
    */

    // We know that (1+q^2) is a factor. Dividing by this gives degree 6 poly.
    Eigen::Matrix<double, 3, 4> eig_vecs;
    double eig_vals[4];
    const int n_roots = qep::qep_div_1_q2(M, C, K, eig_vals, &eig_vecs);

    output->clear();
    for (int i = 0; i < n_roots; ++i) {
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
        CameraPose pose(R, eig_vecs.col(i));

        if (check_cheirality(pose, x1[0], x2[0])) {
            output->push_back(pose);
        }

        // Solution with opposite sign for t
        pose.t = -pose.t;
        if (check_cheirality(pose, x1[0], x2[0])) {
            output->push_back(pose);
        }
    }
    return output->size();
}

} // namespace poselib