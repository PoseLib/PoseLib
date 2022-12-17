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

#include "p4pf.h"

#include "PoseLib/misc/re3q3.h"

namespace poselib {

int p4pf(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output,
         std::vector<double> *output_focal, bool filter_solutions) {

    Eigen::Matrix<double, 2, 4> points2d;
    for (int i = 0; i < 4; ++i) {
        points2d.col(i) = x[i].hnormalized();
    }
    double f0 = points2d.colwise().norm().mean();
    points2d /= f0;

    double d[48];
    // Setup nullspace
    Eigen::Matrix<double, 8, 4> M;
    Eigen::Matrix<double, 4, 4> A;
    Eigen::Map<Eigen::Matrix<double, 8, 4>> N(d);
    Eigen::Map<Eigen::Matrix<double, 4, 4>> B(d + 32);

    for (int i = 0; i < 4; i++) {
        M(0, i) = -points2d(1, i) * X[i](0);
        M(2, i) = -points2d(1, i) * X[i](1);
        M(4, i) = -points2d(1, i) * X[i](2);
        M(6, i) = -points2d(1, i);
        M(1, i) = points2d(0, i) * X[i](0);
        M(3, i) = points2d(0, i) * X[i](1);
        M(5, i) = points2d(0, i) * X[i](2);
        M(7, i) = points2d(0, i);
    }

    // Compute nullspace using QR
    Eigen::Matrix<double, 8, 8> Q = M.householderQr().householderQ();
    N = Q.rightCols(4);

    // Setup matrices A and B (see paper for definition)
    for (int i = 0; i < 4; ++i) {
        if (std::abs(points2d(0, i)) < std::abs(points2d(1, i))) {
            A(i, 0) = points2d(1, i) * X[i](0);
            A(i, 1) = points2d(1, i) * X[i](1);
            A(i, 2) = points2d(1, i) * X[i](2);
            A(i, 3) = points2d(1, i);

            B(i, 0) = X[i](0) * N(1, 0) + X[i](1) * N(3, 0) + X[i](2) * N(5, 0) + N(7, 0); // alpha1
            B(i, 1) = X[i](0) * N(1, 1) + X[i](1) * N(3, 1) + X[i](2) * N(5, 1) + N(7, 1); // alpha2
            B(i, 2) = X[i](0) * N(1, 2) + X[i](1) * N(3, 2) + X[i](2) * N(5, 2) + N(7, 2); // alpha3
            B(i, 3) = X[i](0) * N(1, 3) + X[i](1) * N(3, 3) + X[i](2) * N(5, 3) + N(7, 3); // 1
        } else {
            A(i, 0) = points2d(0, i) * X[i](0);
            A(i, 1) = points2d(0, i) * X[i](1);
            A(i, 2) = points2d(0, i) * X[i](2);
            A(i, 3) = points2d(0, i);

            B(i, 0) = X[i](0) * N(0, 0) + X[i](1) * N(2, 0) + X[i](2) * N(4, 0) + N(6, 0); // alpha1
            B(i, 1) = X[i](0) * N(0, 1) + X[i](1) * N(2, 1) + X[i](2) * N(4, 1) + N(6, 1); // alpha2
            B(i, 2) = X[i](0) * N(0, 2) + X[i](1) * N(2, 2) + X[i](2) * N(4, 2) + N(6, 2); // alpha3
            B(i, 3) = X[i](0) * N(0, 3) + X[i](1) * N(2, 3) + X[i](2) * N(4, 3) + N(6, 3); // 1
        }
    }

    // [p31,p32,p33,p34] = B * [alpha; 1]
    B = A.inverse() * B;

    Eigen::Matrix<double, 3, 10> coeffs;
    Eigen::Matrix<double, 3, 8> solutions;

    // Orthogonality constraints
    coeffs.row(0) << d[0] * d[1] + d[2] * d[3] + d[4] * d[5],
        d[0] * d[9] + d[1] * d[8] + d[2] * d[11] + d[3] * d[10] + d[4] * d[13] + d[5] * d[12],
        d[0] * d[17] + d[1] * d[16] + d[2] * d[19] + d[3] * d[18] + d[4] * d[21] + d[5] * d[20],
        d[8] * d[9] + d[10] * d[11] + d[12] * d[13],
        d[8] * d[17] + d[9] * d[16] + d[10] * d[19] + d[11] * d[18] + d[12] * d[21] + d[13] * d[20],
        d[16] * d[17] + d[18] * d[19] + d[20] * d[21],
        d[0] * d[25] + d[1] * d[24] + d[2] * d[27] + d[3] * d[26] + d[4] * d[29] + d[5] * d[28],
        d[8] * d[25] + d[9] * d[24] + d[10] * d[27] + d[11] * d[26] + d[12] * d[29] + d[13] * d[28],
        d[16] * d[25] + d[17] * d[24] + d[18] * d[27] + d[19] * d[26] + d[20] * d[29] + d[21] * d[28],
        d[24] * d[25] + d[26] * d[27] + d[28] * d[29];
    coeffs.row(1) << d[0] * d[32] + d[2] * d[33] + d[4] * d[34],
        d[0] * d[36] + d[2] * d[37] + d[8] * d[32] + d[4] * d[38] + d[10] * d[33] + d[12] * d[34],
        d[0] * d[40] + d[2] * d[41] + d[4] * d[42] + d[16] * d[32] + d[18] * d[33] + d[20] * d[34],
        d[8] * d[36] + d[10] * d[37] + d[12] * d[38],
        d[8] * d[40] + d[10] * d[41] + d[16] * d[36] + d[12] * d[42] + d[18] * d[37] + d[20] * d[38],
        d[16] * d[40] + d[18] * d[41] + d[20] * d[42],
        d[0] * d[44] + d[2] * d[45] + d[4] * d[46] + d[24] * d[32] + d[26] * d[33] + d[28] * d[34],
        d[8] * d[44] + d[10] * d[45] + d[12] * d[46] + d[24] * d[36] + d[26] * d[37] + d[28] * d[38],
        d[16] * d[44] + d[18] * d[45] + d[24] * d[40] + d[20] * d[46] + d[26] * d[41] + d[28] * d[42],
        d[24] * d[44] + d[26] * d[45] + d[28] * d[46];
    coeffs.row(2) << d[1] * d[32] + d[3] * d[33] + d[5] * d[34],
        d[1] * d[36] + d[3] * d[37] + d[9] * d[32] + d[5] * d[38] + d[11] * d[33] + d[13] * d[34],
        d[1] * d[40] + d[3] * d[41] + d[5] * d[42] + d[17] * d[32] + d[19] * d[33] + d[21] * d[34],
        d[9] * d[36] + d[11] * d[37] + d[13] * d[38],
        d[9] * d[40] + d[11] * d[41] + d[17] * d[36] + d[13] * d[42] + d[19] * d[37] + d[21] * d[38],
        d[17] * d[40] + d[19] * d[41] + d[21] * d[42],
        d[1] * d[44] + d[3] * d[45] + d[5] * d[46] + d[25] * d[32] + d[27] * d[33] + d[29] * d[34],
        d[9] * d[44] + d[11] * d[45] + d[13] * d[46] + d[25] * d[36] + d[27] * d[37] + d[29] * d[38],
        d[17] * d[44] + d[19] * d[45] + d[25] * d[40] + d[21] * d[46] + d[27] * d[41] + d[29] * d[42],
        d[25] * d[44] + d[27] * d[45] + d[29] * d[46];

    // The fourth unused constraint (norms of two first rows equal)
    //	d[0] * d[0] - d[1] * d[1] + d[2] * d[2] - d[3] * d[3] + d[4] * d[4] - d[5] * d[5], 2 * d[0] * d[8] - 2 * d[1] *
    // d[9] + 2 * d[2] * d[10] - 2 * d[3] * d[11] + 2 * d[4] * d[12] - 2 * d[5] * d[13], 2 * d[0] * d[16] - 2 * d[1] *
    // d[17] + 2 * d[2] * d[18] - 2 * d[3] * d[19] + 2 * d[4] * d[20] - 2 * d[5] * d[21], d[8] * d[8] - d[9] * d[9] +
    // d[10] * d[10] - d[11] * d[11] + d[12] * d[12] - d[13] * d[13], 2 * d[8] * d[16] - 2 * d[9] * d[17] + 2 * d[10] *
    // d[18] - 2 * d[11] * d[19] + 2 * d[12] * d[20] - 2 * d[13] * d[21], d[16] * d[16] - d[17] * d[17] + d[18] * d[18]
    // - d[19] * d[19] + d[20] * d[20] - d[21] * d[21], 2 * d[0] * d[24] - 2 * d[1] * d[25] + 2 * d[2] * d[26] - 2 *
    // d[3]
    // * d[27] + 2 * d[4] * d[28] - 2 * d[5] * d[29], 2 * d[8] * d[24] - 2 * d[9] * d[25] + 2 * d[10] * d[26] - 2 *
    // d[11]
    // * d[27] + 2 * d[12] * d[28] - 2 * d[13] * d[29], 2 * d[16] * d[24] - 2 * d[17] * d[25] + 2 * d[18] * d[26] - 2 *
    // d[19] * d[27] + 2 * d[20] * d[28] - 2 * d[21] * d[29], d[24] * d[24] - d[25] * d[25] + d[26] * d[26] - d[27] *
    // d[27] + d[28] * d[28] - d[29] * d[29];

    int n_sols = re3q3::re3q3(coeffs, &solutions);

    CameraPose best_pose;
    double best_focal = 1.0;
    double best_res = 1.0; // we are probably not interested in solutions above this res anyway
    output->clear();
    for (int i = 0; i < n_sols; ++i) {
        Eigen::Matrix<double, 3, 4> P;
        Eigen::Vector4d alpha;
        alpha << solutions.col(i), 1.0;
        Eigen::Matrix<double, 8, 1> P12 = N * alpha;
        P.block<2, 4>(0, 0) = Eigen::Map<Eigen::Matrix<double, 2, 4>>(P12.data());
        P.row(2) = B * alpha;

        if (P.block<3, 3>(0, 0).determinant() < 0)
            P = -P;

        P = P / P.block<1, 3>(2, 0).norm();
        double focal = (P.block<1, 3>(0, 0).norm() + P.block<1, 3>(1, 0).norm()) / 2;
        P.row(0) = P.row(0) / focal;
        P.row(1) = P.row(1) / focal;

        // TODO:  Project to rotations?

        Eigen::Matrix3d R = P.block<3, 3>(0, 0);
        Eigen::Vector3d t = P.block<3, 1>(0, 3);
        focal *= f0;

        CameraPose pose(R, t);

        if (filter_solutions) {
            double res = std::abs(R.row(0).squaredNorm() - 1.0) + std::abs(R.row(1).squaredNorm() - 1.0);
            if (res < best_res) {
                best_pose = pose;
                best_focal = focal;
                best_res = res;
            }
        } else {
            output->push_back(pose);
            output_focal->push_back(focal);
        }
    }

    if (filter_solutions && best_res < 1.0) {
        output->push_back(best_pose);
        output_focal->push_back(best_focal);
    }

    return output->size();
}

} // namespace poselib