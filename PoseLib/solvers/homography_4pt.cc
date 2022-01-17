// Copyright (c) 2021, Viktor Larsson
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

#include "homography_4pt.h"

namespace poselib {

int homography_4pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2, Eigen::Matrix3d *H,
                   bool check_cheirality) {
    if (check_cheirality) {
        Eigen::Vector3d p = x1[0].cross(x1[1]);
        Eigen::Vector3d q = x2[0].cross(x2[1]);

        if (p.dot(x1[2]) * q.dot(x2[2]) < 0)
            return 0;

        if (p.dot(x1[3]) * q.dot(x2[3]) < 0)
            return 0;

        p = x1[2].cross(x1[3]);
        q = x2[2].cross(x2[3]);

        if (p.dot(x1[0]) * q.dot(x2[0]) < 0)
            return 0;
        if (p.dot(x1[1]) * q.dot(x2[1]) < 0)
            return 0;
    }

    Eigen::Matrix<double, 8, 9> M;
    for (size_t i = 0; i < 4; ++i) {
        M.block<1, 3>(2 * i, 0) = x2[i].z() * x1[i].transpose();
        M.block<1, 3>(2 * i, 3).setZero();
        M.block<1, 3>(2 * i, 6) = -x2[i].x() * x1[i].transpose();

        M.block<1, 3>(2 * i + 1, 0).setZero();
        M.block<1, 3>(2 * i + 1, 3) = x2[i].z() * x1[i].transpose();
        M.block<1, 3>(2 * i + 1, 6) = -x2[i].y() * x1[i].transpose();
    }

#if 0
    // Find left nullspace to M using QR (slower)
    Eigen::Matrix<double, 9, 9> Q = M.transpose().householderQr().householderQ();
    Eigen::Matrix<double, 9, 1> h = Q.col(8);
#else
    // Find left nullspace using LU (faster but has degeneracies)
    Eigen::Matrix<double, 9, 1> h = M.block<8, 8>(0, 0).partialPivLu().solve(-M.block<8, 1>(0, 8)).homogeneous();
#endif
    *H = Eigen::Map<const Eigen::Matrix3d>(h.data()).transpose();

    // Check for degenerate homography
    H->normalize();
    double det = H->determinant();
    if (std::abs(det) < 1e-8) {
        return 0;
    }

    return 1;
}

} // namespace poselib