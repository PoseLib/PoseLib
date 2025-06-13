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
//
// S. Cai, Z. Wu, L. Guo, J. Wang, S. Zhang, J. Yan, S. Shen, Fast and interpretable 2d homography decomposition:
// Similarity-kernel-similarity and affine-core-affine transformations. PAMI 2025.

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

    std::array<Eigen::Vector3d, 4> xa;
    std::array<Eigen::Vector3d, 4> xb;
    for (int i = 0; i < 4; i++) {
        xa[i] = x1[i] / x1[i].z();
        xb[i] = x2[i] / x2[i].z();
    }

    double M1N1_x = xa[1].x() - xa[0].x(), M1P1_x = xa[2].x() - xa[0].x(), M1Q1_x = xa[3].x() - xa[0].x();
    double M1N1_y = xa[1].y() - xa[0].y(), M1P1_y = xa[2].y() - xa[0].y(), M1Q1_y = xa[3].y() - xa[0].y();
    double fA1 = M1N1_x * M1P1_y - M1N1_y * M1P1_x;
    double Q3_x = M1P1_y * M1Q1_x - M1P1_x * M1Q1_y;
    double Q3_y = M1N1_x * M1Q1_y - M1N1_y * M1Q1_x;
    //        [  M1P1_y   -M1P1_x          ]   [1        -src[0] ]
    // H_A1 = [ -M1N1_y    M1N1_x          ] * [     1   -src[1] ]
    //        [                       f_A1 ]   [             1   ]

    // compute H_A2 and other variables on target plane
    double M2N2_x = xb[1].x() - xb[0].x(), M2P2_x = xb[2].x() - xb[0].x(), M2Q2_x = xb[3].x() - xb[0].x();
    double M2N2_y = xb[1].y() - xb[0].y(), M2P2_y = xb[2].y() - xb[0].y(), M2Q2_y = xb[3].y() - xb[0].y();
    double fA2 = M2N2_x * M2P2_y - M2N2_y * M2P2_x;
    double Q4_x = M2P2_y * M2Q2_x - M2P2_x * M2Q2_y;
    double Q4_y = M2N2_x * M2Q2_y - M2N2_y * M2Q2_x;
    //            [  M2N2_x   M2P2_x    tar[0] ]
    // H_A2_inv = [  M2N2_y   M2P2_y    tar[1] ]
    //            [                        1   ]

    // obtain the core transformation H_C
    double tt1 = fA1 - Q3_x - Q3_y;
    double C11 = Q3_y * Q4_x * tt1;
    double C22 = Q3_x * Q4_y * tt1;
    double C33 = Q3_x * Q3_y * (fA2 - Q4_x - Q4_y);
    double C31 = C11 - C33;
    double C32 = C22 - C33;
    //       [   C11          0       0  ]
    // H_C = [    0          C22      0  ]
    //       [ C11-C33     C22-C33   C33 ]

    // obtain some intermediate variables
    double tt3 = xb[0].x() * C33;
    double tt4 = xb[0].y() * C33;
    double H1_11 = xb[1].x() * C11 - tt3;
    double H1_12 = xb[2].x() * C22 - tt3;
    double H1_21 = xb[1].y() * C11 - tt4;
    double H1_22 = xb[2].y() * C22 - tt4;
    // H1 = H_A2_inv * H_C
    // only compute the 2*2 upper-left elements. C_33*M2 are repeated twice. no need to compute the last row
    //      [ C11*tar[2]-C33*tar[0]     C22*tar[4]-C33*tar[0]     C33*tar[0]  ]
    // H1 = [ C11*tar[3]-C33*tar[1]     C22*tar[5]-C33*tar[1]     C33*tar[1]  ]
    //      [          C31                       C32                 C33      ]
    Eigen::Matrix<double, 9, 1> h;
    // obtain H
    h[0] = H1_11 * M1P1_y - H1_12 * M1N1_y;
    h[1] = H1_12 * M1N1_x - H1_11 * M1P1_x;
    h[3] = H1_21 * M1P1_y - H1_22 * M1N1_y;
    h[4] = H1_22 * M1N1_x - H1_21 * M1P1_x;
    h[6] = C31 * M1P1_y - C32 * M1N1_y;
    h[7] = C32 * M1N1_x - C31 * M1P1_x;
    h[2] = tt3 * fA1 - h[0] * xa[0].x() - h[1] * xa[0].y();
    h[5] = tt4 * fA1 - h[3] * xa[0].x() - h[4] * xa[0].y();
    h[8] = C33 * fA1 - h[6] * xa[0].x() - h[7] * xa[0].y();

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