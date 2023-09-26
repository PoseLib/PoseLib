// Copyright (c) 2020, Pierre Moulon
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

#include "relpose_8pt.h"

#include "PoseLib/misc/essential.h"

#include <array>
#include <cassert>

/**
 * Build a 9 x n matrix from bearing vector matches, where each row is equivalent to the
 * equation x'T*F*x = 0 for a single correspondence pair (x', x).
 *
 * Note that this does not resize the matrix A; it is expected to have the
 * appropriate size already (n x 9).
 */
void encode_epipolar_equation(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                              Eigen::Matrix<double, Eigen::Dynamic, 9> *A) {
    assert(x1.size() == x2.size());
    assert(A->cols() == 9);
    assert(static_cast<size_t>(A->rows()) == x1.size());
    for (size_t i = 0; i < x1.size(); ++i) {
        A->row(i) << x2[i].x() * x1[i].transpose(), x2[i].y() * x1[i].transpose(), x2[i].z() * x1[i].transpose();
    }
}

void poselib::essential_matrix_8pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                                   Eigen::Matrix3d *essential_matrix) {
    assert(8 <= x1.size());

    using MatX9 = Eigen::Matrix<double, Eigen::Dynamic, 9>;
    MatX9 epipolar_constraint(x1.size(), 9);
    encode_epipolar_equation(x1, x2, &epipolar_constraint);

    using RMat3 = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>;
    Eigen::Matrix3d E;
    if (x1.size() == 8) {
        // In the case where we have exactly 8 correspondences, there is no need to compute the SVD
        Eigen::Matrix<double, 9, 9> Q = epipolar_constraint.transpose().householderQr().householderQ();
        Eigen::Matrix<double, 9, 1> e = Q.col(8);
        E = Eigen::Map<const RMat3>(e.data());
    } else {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 9, 9>> solver(epipolar_constraint.transpose() *
                                                                          epipolar_constraint);
        E = Eigen::Map<const RMat3>(solver.eigenvectors().leftCols<1>().data());
    }

    // Find the closest essential matrix to E in frobenius norm
    // E = UD'VT
    Eigen::JacobiSVD<Eigen::Matrix3d> USV(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d d = USV.singularValues();
    const double a = d[0];
    const double b = d[1];
    d << (a + b) / 2., (a + b) / 2., 0.0;
    E = USV.matrixU() * d.asDiagonal() * USV.matrixV().transpose();

    (*essential_matrix) = E;
}

int poselib::relpose_8pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                         CameraPoseVector *output) {

    Eigen::Matrix3d essential_matrix;
    essential_matrix_8pt(x1, x2, &essential_matrix);
    // Generate plausible relative motion from E
    output->clear();
    poselib::motion_from_essential(essential_matrix, x1, x2, output);
    return output->size();
}
