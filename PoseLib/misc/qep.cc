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

#include "qep.h"

#include "sturm.h"
#include "univariate.h"

namespace poselib {
namespace qep {

int qep_linearize(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B,
                  const Eigen::Matrix<double, 4, 4> &C, double eig_vals[8], Eigen::Matrix<double, 3, 8> *eig_vecs) {
    Eigen::Matrix<double, 8, 8> M;
    M.block<4, 4>(0, 0) = B;
    M.block<4, 4>(0, 4) = C;
    M.block<4, 4>(4, 0).setIdentity();
    M.block<4, 4>(4, 4).setZero();
    M.block<4, 8>(0, 0) = -A.inverse() * M.block<4, 8>(0, 0);
    Eigen::EigenSolver<Eigen::Matrix<double, 8, 8>> es(M, true);

    Eigen::Matrix<std::complex<double>, 8, 1> D = es.eigenvalues();
    Eigen::Matrix<std::complex<double>, 8, 8> V = es.eigenvectors();

    int n_roots = 0;
    for (int i = 0; i < 8; ++i) {
        if (std::abs(D(i).imag()) > 1e-8)
            continue;

        eig_vecs->col(n_roots) = V.block<3, 1>(4, i).real() / V(7, i).real();
        eig_vals[n_roots++] = D(i).real();
    }
    return n_roots;
}

// Computes polynomial p(x) = det(x^2*I + x * A + B)
void detpoly4(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B, double coeffs[9]) {
    coeffs[0] = B(0, 0) * B(1, 1) * B(2, 2) * B(3, 3) - B(0, 0) * B(1, 1) * B(2, 3) * B(3, 2) -
                B(0, 0) * B(1, 2) * B(2, 1) * B(3, 3) + B(0, 0) * B(1, 2) * B(2, 3) * B(3, 1) +
                B(0, 0) * B(1, 3) * B(2, 1) * B(3, 2) - B(0, 0) * B(1, 3) * B(2, 2) * B(3, 1) -
                B(0, 1) * B(1, 0) * B(2, 2) * B(3, 3) + B(0, 1) * B(1, 0) * B(2, 3) * B(3, 2) +
                B(0, 1) * B(1, 2) * B(2, 0) * B(3, 3) - B(0, 1) * B(1, 2) * B(2, 3) * B(3, 0) -
                B(0, 1) * B(1, 3) * B(2, 0) * B(3, 2) + B(0, 1) * B(1, 3) * B(2, 2) * B(3, 0) +
                B(0, 2) * B(1, 0) * B(2, 1) * B(3, 3) - B(0, 2) * B(1, 0) * B(2, 3) * B(3, 1) -
                B(0, 2) * B(1, 1) * B(2, 0) * B(3, 3) + B(0, 2) * B(1, 1) * B(2, 3) * B(3, 0) +
                B(0, 2) * B(1, 3) * B(2, 0) * B(3, 1) - B(0, 2) * B(1, 3) * B(2, 1) * B(3, 0) -
                B(0, 3) * B(1, 0) * B(2, 1) * B(3, 2) + B(0, 3) * B(1, 0) * B(2, 2) * B(3, 1) +
                B(0, 3) * B(1, 1) * B(2, 0) * B(3, 2) - B(0, 3) * B(1, 1) * B(2, 2) * B(3, 0) -
                B(0, 3) * B(1, 2) * B(2, 0) * B(3, 1) + B(0, 3) * B(1, 2) * B(2, 1) * B(3, 0);
    coeffs[1] = A(0, 0) * B(1, 1) * B(2, 2) * B(3, 3) - A(0, 0) * B(1, 1) * B(2, 3) * B(3, 2) -
                A(0, 0) * B(1, 2) * B(2, 1) * B(3, 3) + A(0, 0) * B(1, 2) * B(2, 3) * B(3, 1) +
                A(0, 0) * B(1, 3) * B(2, 1) * B(3, 2) - A(0, 0) * B(1, 3) * B(2, 2) * B(3, 1) -
                A(0, 1) * B(1, 0) * B(2, 2) * B(3, 3) + A(0, 1) * B(1, 0) * B(2, 3) * B(3, 2) +
                A(0, 1) * B(1, 2) * B(2, 0) * B(3, 3) - A(0, 1) * B(1, 2) * B(2, 3) * B(3, 0) -
                A(0, 1) * B(1, 3) * B(2, 0) * B(3, 2) + A(0, 1) * B(1, 3) * B(2, 2) * B(3, 0) +
                A(0, 2) * B(1, 0) * B(2, 1) * B(3, 3) - A(0, 2) * B(1, 0) * B(2, 3) * B(3, 1) -
                A(0, 2) * B(1, 1) * B(2, 0) * B(3, 3) + A(0, 2) * B(1, 1) * B(2, 3) * B(3, 0) +
                A(0, 2) * B(1, 3) * B(2, 0) * B(3, 1) - A(0, 2) * B(1, 3) * B(2, 1) * B(3, 0) -
                A(0, 3) * B(1, 0) * B(2, 1) * B(3, 2) + A(0, 3) * B(1, 0) * B(2, 2) * B(3, 1) +
                A(0, 3) * B(1, 1) * B(2, 0) * B(3, 2) - A(0, 3) * B(1, 1) * B(2, 2) * B(3, 0) -
                A(0, 3) * B(1, 2) * B(2, 0) * B(3, 1) + A(0, 3) * B(1, 2) * B(2, 1) * B(3, 0) -
                A(1, 0) * B(0, 1) * B(2, 2) * B(3, 3) + A(1, 0) * B(0, 1) * B(2, 3) * B(3, 2) +
                A(1, 0) * B(0, 2) * B(2, 1) * B(3, 3) - A(1, 0) * B(0, 2) * B(2, 3) * B(3, 1) -
                A(1, 0) * B(0, 3) * B(2, 1) * B(3, 2) + A(1, 0) * B(0, 3) * B(2, 2) * B(3, 1) +
                A(1, 1) * B(0, 0) * B(2, 2) * B(3, 3) - A(1, 1) * B(0, 0) * B(2, 3) * B(3, 2) -
                A(1, 1) * B(0, 2) * B(2, 0) * B(3, 3) + A(1, 1) * B(0, 2) * B(2, 3) * B(3, 0) +
                A(1, 1) * B(0, 3) * B(2, 0) * B(3, 2) - A(1, 1) * B(0, 3) * B(2, 2) * B(3, 0) -
                A(1, 2) * B(0, 0) * B(2, 1) * B(3, 3) + A(1, 2) * B(0, 0) * B(2, 3) * B(3, 1) +
                A(1, 2) * B(0, 1) * B(2, 0) * B(3, 3) - A(1, 2) * B(0, 1) * B(2, 3) * B(3, 0) -
                A(1, 2) * B(0, 3) * B(2, 0) * B(3, 1) + A(1, 2) * B(0, 3) * B(2, 1) * B(3, 0) +
                A(1, 3) * B(0, 0) * B(2, 1) * B(3, 2) - A(1, 3) * B(0, 0) * B(2, 2) * B(3, 1) -
                A(1, 3) * B(0, 1) * B(2, 0) * B(3, 2) + A(1, 3) * B(0, 1) * B(2, 2) * B(3, 0) +
                A(1, 3) * B(0, 2) * B(2, 0) * B(3, 1) - A(1, 3) * B(0, 2) * B(2, 1) * B(3, 0) +
                A(2, 0) * B(0, 1) * B(1, 2) * B(3, 3) - A(2, 0) * B(0, 1) * B(1, 3) * B(3, 2) -
                A(2, 0) * B(0, 2) * B(1, 1) * B(3, 3) + A(2, 0) * B(0, 2) * B(1, 3) * B(3, 1) +
                A(2, 0) * B(0, 3) * B(1, 1) * B(3, 2) - A(2, 0) * B(0, 3) * B(1, 2) * B(3, 1) -
                A(2, 1) * B(0, 0) * B(1, 2) * B(3, 3) + A(2, 1) * B(0, 0) * B(1, 3) * B(3, 2) +
                A(2, 1) * B(0, 2) * B(1, 0) * B(3, 3) - A(2, 1) * B(0, 2) * B(1, 3) * B(3, 0) -
                A(2, 1) * B(0, 3) * B(1, 0) * B(3, 2) + A(2, 1) * B(0, 3) * B(1, 2) * B(3, 0) +
                A(2, 2) * B(0, 0) * B(1, 1) * B(3, 3) - A(2, 2) * B(0, 0) * B(1, 3) * B(3, 1) -
                A(2, 2) * B(0, 1) * B(1, 0) * B(3, 3) + A(2, 2) * B(0, 1) * B(1, 3) * B(3, 0) +
                A(2, 2) * B(0, 3) * B(1, 0) * B(3, 1) - A(2, 2) * B(0, 3) * B(1, 1) * B(3, 0) -
                A(2, 3) * B(0, 0) * B(1, 1) * B(3, 2) + A(2, 3) * B(0, 0) * B(1, 2) * B(3, 1) +
                A(2, 3) * B(0, 1) * B(1, 0) * B(3, 2) - A(2, 3) * B(0, 1) * B(1, 2) * B(3, 0) -
                A(2, 3) * B(0, 2) * B(1, 0) * B(3, 1) + A(2, 3) * B(0, 2) * B(1, 1) * B(3, 0) -
                A(3, 0) * B(0, 1) * B(1, 2) * B(2, 3) + A(3, 0) * B(0, 1) * B(1, 3) * B(2, 2) +
                A(3, 0) * B(0, 2) * B(1, 1) * B(2, 3) - A(3, 0) * B(0, 2) * B(1, 3) * B(2, 1) -
                A(3, 0) * B(0, 3) * B(1, 1) * B(2, 2) + A(3, 0) * B(0, 3) * B(1, 2) * B(2, 1) +
                A(3, 1) * B(0, 0) * B(1, 2) * B(2, 3) - A(3, 1) * B(0, 0) * B(1, 3) * B(2, 2) -
                A(3, 1) * B(0, 2) * B(1, 0) * B(2, 3) + A(3, 1) * B(0, 2) * B(1, 3) * B(2, 0) +
                A(3, 1) * B(0, 3) * B(1, 0) * B(2, 2) - A(3, 1) * B(0, 3) * B(1, 2) * B(2, 0) -
                A(3, 2) * B(0, 0) * B(1, 1) * B(2, 3) + A(3, 2) * B(0, 0) * B(1, 3) * B(2, 1) +
                A(3, 2) * B(0, 1) * B(1, 0) * B(2, 3) - A(3, 2) * B(0, 1) * B(1, 3) * B(2, 0) -
                A(3, 2) * B(0, 3) * B(1, 0) * B(2, 1) + A(3, 2) * B(0, 3) * B(1, 1) * B(2, 0) +
                A(3, 3) * B(0, 0) * B(1, 1) * B(2, 2) - A(3, 3) * B(0, 0) * B(1, 2) * B(2, 1) -
                A(3, 3) * B(0, 1) * B(1, 0) * B(2, 2) + A(3, 3) * B(0, 1) * B(1, 2) * B(2, 0) +
                A(3, 3) * B(0, 2) * B(1, 0) * B(2, 1) - A(3, 3) * B(0, 2) * B(1, 1) * B(2, 0);
    coeffs[2] = B(0, 0) * B(1, 1) * B(2, 2) - B(0, 0) * B(1, 2) * B(2, 1) - B(0, 1) * B(1, 0) * B(2, 2) +
                B(0, 1) * B(1, 2) * B(2, 0) + B(0, 2) * B(1, 0) * B(2, 1) - B(0, 2) * B(1, 1) * B(2, 0) +
                B(0, 0) * B(1, 1) * B(3, 3) - B(0, 0) * B(1, 3) * B(3, 1) - B(0, 1) * B(1, 0) * B(3, 3) +
                B(0, 1) * B(1, 3) * B(3, 0) + B(0, 3) * B(1, 0) * B(3, 1) - B(0, 3) * B(1, 1) * B(3, 0) +
                B(0, 0) * B(2, 2) * B(3, 3) - B(0, 0) * B(2, 3) * B(3, 2) - B(0, 2) * B(2, 0) * B(3, 3) +
                B(0, 2) * B(2, 3) * B(3, 0) + B(0, 3) * B(2, 0) * B(3, 2) - B(0, 3) * B(2, 2) * B(3, 0) +
                B(1, 1) * B(2, 2) * B(3, 3) - B(1, 1) * B(2, 3) * B(3, 2) - B(1, 2) * B(2, 1) * B(3, 3) +
                B(1, 2) * B(2, 3) * B(3, 1) + B(1, 3) * B(2, 1) * B(3, 2) - B(1, 3) * B(2, 2) * B(3, 1) +
                A(0, 0) * A(1, 1) * B(2, 2) * B(3, 3) - A(0, 0) * A(1, 1) * B(2, 3) * B(3, 2) -
                A(0, 0) * A(1, 2) * B(2, 1) * B(3, 3) + A(0, 0) * A(1, 2) * B(2, 3) * B(3, 1) +
                A(0, 0) * A(1, 3) * B(2, 1) * B(3, 2) - A(0, 0) * A(1, 3) * B(2, 2) * B(3, 1) -
                A(0, 0) * A(2, 1) * B(1, 2) * B(3, 3) + A(0, 0) * A(2, 1) * B(1, 3) * B(3, 2) +
                A(0, 0) * A(2, 2) * B(1, 1) * B(3, 3) - A(0, 0) * A(2, 2) * B(1, 3) * B(3, 1) -
                A(0, 0) * A(2, 3) * B(1, 1) * B(3, 2) + A(0, 0) * A(2, 3) * B(1, 2) * B(3, 1) +
                A(0, 0) * A(3, 1) * B(1, 2) * B(2, 3) - A(0, 0) * A(3, 1) * B(1, 3) * B(2, 2) -
                A(0, 0) * A(3, 2) * B(1, 1) * B(2, 3) + A(0, 0) * A(3, 2) * B(1, 3) * B(2, 1) +
                A(0, 0) * A(3, 3) * B(1, 1) * B(2, 2) - A(0, 0) * A(3, 3) * B(1, 2) * B(2, 1) -
                A(0, 1) * A(1, 0) * B(2, 2) * B(3, 3) + A(0, 1) * A(1, 0) * B(2, 3) * B(3, 2) +
                A(0, 1) * A(1, 2) * B(2, 0) * B(3, 3) - A(0, 1) * A(1, 2) * B(2, 3) * B(3, 0) -
                A(0, 1) * A(1, 3) * B(2, 0) * B(3, 2) + A(0, 1) * A(1, 3) * B(2, 2) * B(3, 0) +
                A(0, 1) * A(2, 0) * B(1, 2) * B(3, 3) - A(0, 1) * A(2, 0) * B(1, 3) * B(3, 2) -
                A(0, 1) * A(2, 2) * B(1, 0) * B(3, 3) + A(0, 1) * A(2, 2) * B(1, 3) * B(3, 0) +
                A(0, 1) * A(2, 3) * B(1, 0) * B(3, 2) - A(0, 1) * A(2, 3) * B(1, 2) * B(3, 0) -
                A(0, 1) * A(3, 0) * B(1, 2) * B(2, 3) + A(0, 1) * A(3, 0) * B(1, 3) * B(2, 2) +
                A(0, 1) * A(3, 2) * B(1, 0) * B(2, 3) - A(0, 1) * A(3, 2) * B(1, 3) * B(2, 0) -
                A(0, 1) * A(3, 3) * B(1, 0) * B(2, 2) + A(0, 1) * A(3, 3) * B(1, 2) * B(2, 0) +
                A(0, 2) * A(1, 0) * B(2, 1) * B(3, 3) - A(0, 2) * A(1, 0) * B(2, 3) * B(3, 1) -
                A(0, 2) * A(1, 1) * B(2, 0) * B(3, 3) + A(0, 2) * A(1, 1) * B(2, 3) * B(3, 0) +
                A(0, 2) * A(1, 3) * B(2, 0) * B(3, 1) - A(0, 2) * A(1, 3) * B(2, 1) * B(3, 0) -
                A(0, 2) * A(2, 0) * B(1, 1) * B(3, 3) + A(0, 2) * A(2, 0) * B(1, 3) * B(3, 1) +
                A(0, 2) * A(2, 1) * B(1, 0) * B(3, 3) - A(0, 2) * A(2, 1) * B(1, 3) * B(3, 0) -
                A(0, 2) * A(2, 3) * B(1, 0) * B(3, 1) + A(0, 2) * A(2, 3) * B(1, 1) * B(3, 0) +
                A(0, 2) * A(3, 0) * B(1, 1) * B(2, 3) - A(0, 2) * A(3, 0) * B(1, 3) * B(2, 1) -
                A(0, 2) * A(3, 1) * B(1, 0) * B(2, 3) + A(0, 2) * A(3, 1) * B(1, 3) * B(2, 0) +
                A(0, 2) * A(3, 3) * B(1, 0) * B(2, 1) - A(0, 2) * A(3, 3) * B(1, 1) * B(2, 0) -
                A(0, 3) * A(1, 0) * B(2, 1) * B(3, 2) + A(0, 3) * A(1, 0) * B(2, 2) * B(3, 1) +
                A(0, 3) * A(1, 1) * B(2, 0) * B(3, 2) - A(0, 3) * A(1, 1) * B(2, 2) * B(3, 0) -
                A(0, 3) * A(1, 2) * B(2, 0) * B(3, 1) + A(0, 3) * A(1, 2) * B(2, 1) * B(3, 0) +
                A(0, 3) * A(2, 0) * B(1, 1) * B(3, 2) - A(0, 3) * A(2, 0) * B(1, 2) * B(3, 1) -
                A(0, 3) * A(2, 1) * B(1, 0) * B(3, 2) + A(0, 3) * A(2, 1) * B(1, 2) * B(3, 0) +
                A(0, 3) * A(2, 2) * B(1, 0) * B(3, 1) - A(0, 3) * A(2, 2) * B(1, 1) * B(3, 0) -
                A(0, 3) * A(3, 0) * B(1, 1) * B(2, 2) + A(0, 3) * A(3, 0) * B(1, 2) * B(2, 1) +
                A(0, 3) * A(3, 1) * B(1, 0) * B(2, 2) - A(0, 3) * A(3, 1) * B(1, 2) * B(2, 0) -
                A(0, 3) * A(3, 2) * B(1, 0) * B(2, 1) + A(0, 3) * A(3, 2) * B(1, 1) * B(2, 0) +
                A(1, 0) * A(2, 1) * B(0, 2) * B(3, 3) - A(1, 0) * A(2, 1) * B(0, 3) * B(3, 2) -
                A(1, 0) * A(2, 2) * B(0, 1) * B(3, 3) + A(1, 0) * A(2, 2) * B(0, 3) * B(3, 1) +
                A(1, 0) * A(2, 3) * B(0, 1) * B(3, 2) - A(1, 0) * A(2, 3) * B(0, 2) * B(3, 1) -
                A(1, 0) * A(3, 1) * B(0, 2) * B(2, 3) + A(1, 0) * A(3, 1) * B(0, 3) * B(2, 2) +
                A(1, 0) * A(3, 2) * B(0, 1) * B(2, 3) - A(1, 0) * A(3, 2) * B(0, 3) * B(2, 1) -
                A(1, 0) * A(3, 3) * B(0, 1) * B(2, 2) + A(1, 0) * A(3, 3) * B(0, 2) * B(2, 1) -
                A(1, 1) * A(2, 0) * B(0, 2) * B(3, 3) + A(1, 1) * A(2, 0) * B(0, 3) * B(3, 2) +
                A(1, 1) * A(2, 2) * B(0, 0) * B(3, 3) - A(1, 1) * A(2, 2) * B(0, 3) * B(3, 0) -
                A(1, 1) * A(2, 3) * B(0, 0) * B(3, 2) + A(1, 1) * A(2, 3) * B(0, 2) * B(3, 0) +
                A(1, 1) * A(3, 0) * B(0, 2) * B(2, 3) - A(1, 1) * A(3, 0) * B(0, 3) * B(2, 2) -
                A(1, 1) * A(3, 2) * B(0, 0) * B(2, 3) + A(1, 1) * A(3, 2) * B(0, 3) * B(2, 0) +
                A(1, 1) * A(3, 3) * B(0, 0) * B(2, 2) - A(1, 1) * A(3, 3) * B(0, 2) * B(2, 0) +
                A(1, 2) * A(2, 0) * B(0, 1) * B(3, 3) - A(1, 2) * A(2, 0) * B(0, 3) * B(3, 1) -
                A(1, 2) * A(2, 1) * B(0, 0) * B(3, 3) + A(1, 2) * A(2, 1) * B(0, 3) * B(3, 0) +
                A(1, 2) * A(2, 3) * B(0, 0) * B(3, 1) - A(1, 2) * A(2, 3) * B(0, 1) * B(3, 0) -
                A(1, 2) * A(3, 0) * B(0, 1) * B(2, 3) + A(1, 2) * A(3, 0) * B(0, 3) * B(2, 1) +
                A(1, 2) * A(3, 1) * B(0, 0) * B(2, 3) - A(1, 2) * A(3, 1) * B(0, 3) * B(2, 0) -
                A(1, 2) * A(3, 3) * B(0, 0) * B(2, 1) + A(1, 2) * A(3, 3) * B(0, 1) * B(2, 0) -
                A(1, 3) * A(2, 0) * B(0, 1) * B(3, 2) + A(1, 3) * A(2, 0) * B(0, 2) * B(3, 1) +
                A(1, 3) * A(2, 1) * B(0, 0) * B(3, 2) - A(1, 3) * A(2, 1) * B(0, 2) * B(3, 0) -
                A(1, 3) * A(2, 2) * B(0, 0) * B(3, 1) + A(1, 3) * A(2, 2) * B(0, 1) * B(3, 0) +
                A(1, 3) * A(3, 0) * B(0, 1) * B(2, 2) - A(1, 3) * A(3, 0) * B(0, 2) * B(2, 1) -
                A(1, 3) * A(3, 1) * B(0, 0) * B(2, 2) + A(1, 3) * A(3, 1) * B(0, 2) * B(2, 0) +
                A(1, 3) * A(3, 2) * B(0, 0) * B(2, 1) - A(1, 3) * A(3, 2) * B(0, 1) * B(2, 0) +
                A(2, 0) * A(3, 1) * B(0, 2) * B(1, 3) - A(2, 0) * A(3, 1) * B(0, 3) * B(1, 2) -
                A(2, 0) * A(3, 2) * B(0, 1) * B(1, 3) + A(2, 0) * A(3, 2) * B(0, 3) * B(1, 1) +
                A(2, 0) * A(3, 3) * B(0, 1) * B(1, 2) - A(2, 0) * A(3, 3) * B(0, 2) * B(1, 1) -
                A(2, 1) * A(3, 0) * B(0, 2) * B(1, 3) + A(2, 1) * A(3, 0) * B(0, 3) * B(1, 2) +
                A(2, 1) * A(3, 2) * B(0, 0) * B(1, 3) - A(2, 1) * A(3, 2) * B(0, 3) * B(1, 0) -
                A(2, 1) * A(3, 3) * B(0, 0) * B(1, 2) + A(2, 1) * A(3, 3) * B(0, 2) * B(1, 0) +
                A(2, 2) * A(3, 0) * B(0, 1) * B(1, 3) - A(2, 2) * A(3, 0) * B(0, 3) * B(1, 1) -
                A(2, 2) * A(3, 1) * B(0, 0) * B(1, 3) + A(2, 2) * A(3, 1) * B(0, 3) * B(1, 0) +
                A(2, 2) * A(3, 3) * B(0, 0) * B(1, 1) - A(2, 2) * A(3, 3) * B(0, 1) * B(1, 0) -
                A(2, 3) * A(3, 0) * B(0, 1) * B(1, 2) + A(2, 3) * A(3, 0) * B(0, 2) * B(1, 1) +
                A(2, 3) * A(3, 1) * B(0, 0) * B(1, 2) - A(2, 3) * A(3, 1) * B(0, 2) * B(1, 0) -
                A(2, 3) * A(3, 2) * B(0, 0) * B(1, 1) + A(2, 3) * A(3, 2) * B(0, 1) * B(1, 0);
    coeffs[3] = A(0, 0) * B(1, 1) * B(2, 2) - A(0, 0) * B(1, 2) * B(2, 1) - A(0, 1) * B(1, 0) * B(2, 2) +
                A(0, 1) * B(1, 2) * B(2, 0) + A(0, 2) * B(1, 0) * B(2, 1) - A(0, 2) * B(1, 1) * B(2, 0) -
                A(1, 0) * B(0, 1) * B(2, 2) + A(1, 0) * B(0, 2) * B(2, 1) + A(1, 1) * B(0, 0) * B(2, 2) -
                A(1, 1) * B(0, 2) * B(2, 0) - A(1, 2) * B(0, 0) * B(2, 1) + A(1, 2) * B(0, 1) * B(2, 0) +
                A(2, 0) * B(0, 1) * B(1, 2) - A(2, 0) * B(0, 2) * B(1, 1) - A(2, 1) * B(0, 0) * B(1, 2) +
                A(2, 1) * B(0, 2) * B(1, 0) + A(2, 2) * B(0, 0) * B(1, 1) - A(2, 2) * B(0, 1) * B(1, 0) +
                A(0, 0) * B(1, 1) * B(3, 3) - A(0, 0) * B(1, 3) * B(3, 1) - A(0, 1) * B(1, 0) * B(3, 3) +
                A(0, 1) * B(1, 3) * B(3, 0) + A(0, 3) * B(1, 0) * B(3, 1) - A(0, 3) * B(1, 1) * B(3, 0) -
                A(1, 0) * B(0, 1) * B(3, 3) + A(1, 0) * B(0, 3) * B(3, 1) + A(1, 1) * B(0, 0) * B(3, 3) -
                A(1, 1) * B(0, 3) * B(3, 0) - A(1, 3) * B(0, 0) * B(3, 1) + A(1, 3) * B(0, 1) * B(3, 0) +
                A(3, 0) * B(0, 1) * B(1, 3) - A(3, 0) * B(0, 3) * B(1, 1) - A(3, 1) * B(0, 0) * B(1, 3) +
                A(3, 1) * B(0, 3) * B(1, 0) + A(3, 3) * B(0, 0) * B(1, 1) - A(3, 3) * B(0, 1) * B(1, 0) +
                A(0, 0) * B(2, 2) * B(3, 3) - A(0, 0) * B(2, 3) * B(3, 2) - A(0, 2) * B(2, 0) * B(3, 3) +
                A(0, 2) * B(2, 3) * B(3, 0) + A(0, 3) * B(2, 0) * B(3, 2) - A(0, 3) * B(2, 2) * B(3, 0) -
                A(2, 0) * B(0, 2) * B(3, 3) + A(2, 0) * B(0, 3) * B(3, 2) + A(2, 2) * B(0, 0) * B(3, 3) -
                A(2, 2) * B(0, 3) * B(3, 0) - A(2, 3) * B(0, 0) * B(3, 2) + A(2, 3) * B(0, 2) * B(3, 0) +
                A(3, 0) * B(0, 2) * B(2, 3) - A(3, 0) * B(0, 3) * B(2, 2) - A(3, 2) * B(0, 0) * B(2, 3) +
                A(3, 2) * B(0, 3) * B(2, 0) + A(3, 3) * B(0, 0) * B(2, 2) - A(3, 3) * B(0, 2) * B(2, 0) +
                A(1, 1) * B(2, 2) * B(3, 3) - A(1, 1) * B(2, 3) * B(3, 2) - A(1, 2) * B(2, 1) * B(3, 3) +
                A(1, 2) * B(2, 3) * B(3, 1) + A(1, 3) * B(2, 1) * B(3, 2) - A(1, 3) * B(2, 2) * B(3, 1) -
                A(2, 1) * B(1, 2) * B(3, 3) + A(2, 1) * B(1, 3) * B(3, 2) + A(2, 2) * B(1, 1) * B(3, 3) -
                A(2, 2) * B(1, 3) * B(3, 1) - A(2, 3) * B(1, 1) * B(3, 2) + A(2, 3) * B(1, 2) * B(3, 1) +
                A(3, 1) * B(1, 2) * B(2, 3) - A(3, 1) * B(1, 3) * B(2, 2) - A(3, 2) * B(1, 1) * B(2, 3) +
                A(3, 2) * B(1, 3) * B(2, 1) + A(3, 3) * B(1, 1) * B(2, 2) - A(3, 3) * B(1, 2) * B(2, 1) +
                A(0, 0) * A(1, 1) * A(2, 2) * B(3, 3) - A(0, 0) * A(1, 1) * A(2, 3) * B(3, 2) -
                A(0, 0) * A(1, 1) * A(3, 2) * B(2, 3) + A(0, 0) * A(1, 1) * A(3, 3) * B(2, 2) -
                A(0, 0) * A(1, 2) * A(2, 1) * B(3, 3) + A(0, 0) * A(1, 2) * A(2, 3) * B(3, 1) +
                A(0, 0) * A(1, 2) * A(3, 1) * B(2, 3) - A(0, 0) * A(1, 2) * A(3, 3) * B(2, 1) +
                A(0, 0) * A(1, 3) * A(2, 1) * B(3, 2) - A(0, 0) * A(1, 3) * A(2, 2) * B(3, 1) -
                A(0, 0) * A(1, 3) * A(3, 1) * B(2, 2) + A(0, 0) * A(1, 3) * A(3, 2) * B(2, 1) +
                A(0, 0) * A(2, 1) * A(3, 2) * B(1, 3) - A(0, 0) * A(2, 1) * A(3, 3) * B(1, 2) -
                A(0, 0) * A(2, 2) * A(3, 1) * B(1, 3) + A(0, 0) * A(2, 2) * A(3, 3) * B(1, 1) +
                A(0, 0) * A(2, 3) * A(3, 1) * B(1, 2) - A(0, 0) * A(2, 3) * A(3, 2) * B(1, 1) -
                A(0, 1) * A(1, 0) * A(2, 2) * B(3, 3) + A(0, 1) * A(1, 0) * A(2, 3) * B(3, 2) +
                A(0, 1) * A(1, 0) * A(3, 2) * B(2, 3) - A(0, 1) * A(1, 0) * A(3, 3) * B(2, 2) +
                A(0, 1) * A(1, 2) * A(2, 0) * B(3, 3) - A(0, 1) * A(1, 2) * A(2, 3) * B(3, 0) -
                A(0, 1) * A(1, 2) * A(3, 0) * B(2, 3) + A(0, 1) * A(1, 2) * A(3, 3) * B(2, 0) -
                A(0, 1) * A(1, 3) * A(2, 0) * B(3, 2) + A(0, 1) * A(1, 3) * A(2, 2) * B(3, 0) +
                A(0, 1) * A(1, 3) * A(3, 0) * B(2, 2) - A(0, 1) * A(1, 3) * A(3, 2) * B(2, 0) -
                A(0, 1) * A(2, 0) * A(3, 2) * B(1, 3) + A(0, 1) * A(2, 0) * A(3, 3) * B(1, 2) +
                A(0, 1) * A(2, 2) * A(3, 0) * B(1, 3) - A(0, 1) * A(2, 2) * A(3, 3) * B(1, 0) -
                A(0, 1) * A(2, 3) * A(3, 0) * B(1, 2) + A(0, 1) * A(2, 3) * A(3, 2) * B(1, 0) +
                A(0, 2) * A(1, 0) * A(2, 1) * B(3, 3) - A(0, 2) * A(1, 0) * A(2, 3) * B(3, 1) -
                A(0, 2) * A(1, 0) * A(3, 1) * B(2, 3) + A(0, 2) * A(1, 0) * A(3, 3) * B(2, 1) -
                A(0, 2) * A(1, 1) * A(2, 0) * B(3, 3) + A(0, 2) * A(1, 1) * A(2, 3) * B(3, 0) +
                A(0, 2) * A(1, 1) * A(3, 0) * B(2, 3) - A(0, 2) * A(1, 1) * A(3, 3) * B(2, 0) +
                A(0, 2) * A(1, 3) * A(2, 0) * B(3, 1) - A(0, 2) * A(1, 3) * A(2, 1) * B(3, 0) -
                A(0, 2) * A(1, 3) * A(3, 0) * B(2, 1) + A(0, 2) * A(1, 3) * A(3, 1) * B(2, 0) +
                A(0, 2) * A(2, 0) * A(3, 1) * B(1, 3) - A(0, 2) * A(2, 0) * A(3, 3) * B(1, 1) -
                A(0, 2) * A(2, 1) * A(3, 0) * B(1, 3) + A(0, 2) * A(2, 1) * A(3, 3) * B(1, 0) +
                A(0, 2) * A(2, 3) * A(3, 0) * B(1, 1) - A(0, 2) * A(2, 3) * A(3, 1) * B(1, 0) -
                A(0, 3) * A(1, 0) * A(2, 1) * B(3, 2) + A(0, 3) * A(1, 0) * A(2, 2) * B(3, 1) +
                A(0, 3) * A(1, 0) * A(3, 1) * B(2, 2) - A(0, 3) * A(1, 0) * A(3, 2) * B(2, 1) +
                A(0, 3) * A(1, 1) * A(2, 0) * B(3, 2) - A(0, 3) * A(1, 1) * A(2, 2) * B(3, 0) -
                A(0, 3) * A(1, 1) * A(3, 0) * B(2, 2) + A(0, 3) * A(1, 1) * A(3, 2) * B(2, 0) -
                A(0, 3) * A(1, 2) * A(2, 0) * B(3, 1) + A(0, 3) * A(1, 2) * A(2, 1) * B(3, 0) +
                A(0, 3) * A(1, 2) * A(3, 0) * B(2, 1) - A(0, 3) * A(1, 2) * A(3, 1) * B(2, 0) -
                A(0, 3) * A(2, 0) * A(3, 1) * B(1, 2) + A(0, 3) * A(2, 0) * A(3, 2) * B(1, 1) +
                A(0, 3) * A(2, 1) * A(3, 0) * B(1, 2) - A(0, 3) * A(2, 1) * A(3, 2) * B(1, 0) -
                A(0, 3) * A(2, 2) * A(3, 0) * B(1, 1) + A(0, 3) * A(2, 2) * A(3, 1) * B(1, 0) -
                A(1, 0) * A(2, 1) * A(3, 2) * B(0, 3) + A(1, 0) * A(2, 1) * A(3, 3) * B(0, 2) +
                A(1, 0) * A(2, 2) * A(3, 1) * B(0, 3) - A(1, 0) * A(2, 2) * A(3, 3) * B(0, 1) -
                A(1, 0) * A(2, 3) * A(3, 1) * B(0, 2) + A(1, 0) * A(2, 3) * A(3, 2) * B(0, 1) +
                A(1, 1) * A(2, 0) * A(3, 2) * B(0, 3) - A(1, 1) * A(2, 0) * A(3, 3) * B(0, 2) -
                A(1, 1) * A(2, 2) * A(3, 0) * B(0, 3) + A(1, 1) * A(2, 2) * A(3, 3) * B(0, 0) +
                A(1, 1) * A(2, 3) * A(3, 0) * B(0, 2) - A(1, 1) * A(2, 3) * A(3, 2) * B(0, 0) -
                A(1, 2) * A(2, 0) * A(3, 1) * B(0, 3) + A(1, 2) * A(2, 0) * A(3, 3) * B(0, 1) +
                A(1, 2) * A(2, 1) * A(3, 0) * B(0, 3) - A(1, 2) * A(2, 1) * A(3, 3) * B(0, 0) -
                A(1, 2) * A(2, 3) * A(3, 0) * B(0, 1) + A(1, 2) * A(2, 3) * A(3, 1) * B(0, 0) +
                A(1, 3) * A(2, 0) * A(3, 1) * B(0, 2) - A(1, 3) * A(2, 0) * A(3, 2) * B(0, 1) -
                A(1, 3) * A(2, 1) * A(3, 0) * B(0, 2) + A(1, 3) * A(2, 1) * A(3, 2) * B(0, 0) +
                A(1, 3) * A(2, 2) * A(3, 0) * B(0, 1) - A(1, 3) * A(2, 2) * A(3, 1) * B(0, 0);
    coeffs[4] = B(0, 0) * B(1, 1) - B(0, 1) * B(1, 0) + B(0, 0) * B(2, 2) - B(0, 2) * B(2, 0) + B(0, 0) * B(3, 3) -
                B(0, 3) * B(3, 0) + B(1, 1) * B(2, 2) - B(1, 2) * B(2, 1) + B(1, 1) * B(3, 3) - B(1, 3) * B(3, 1) +
                B(2, 2) * B(3, 3) - B(2, 3) * B(3, 2) + A(0, 0) * A(1, 1) * B(2, 2) - A(0, 0) * A(1, 2) * B(2, 1) -
                A(0, 0) * A(2, 1) * B(1, 2) + A(0, 0) * A(2, 2) * B(1, 1) - A(0, 1) * A(1, 0) * B(2, 2) +
                A(0, 1) * A(1, 2) * B(2, 0) + A(0, 1) * A(2, 0) * B(1, 2) - A(0, 1) * A(2, 2) * B(1, 0) +
                A(0, 2) * A(1, 0) * B(2, 1) - A(0, 2) * A(1, 1) * B(2, 0) - A(0, 2) * A(2, 0) * B(1, 1) +
                A(0, 2) * A(2, 1) * B(1, 0) + A(1, 0) * A(2, 1) * B(0, 2) - A(1, 0) * A(2, 2) * B(0, 1) -
                A(1, 1) * A(2, 0) * B(0, 2) + A(1, 1) * A(2, 2) * B(0, 0) + A(1, 2) * A(2, 0) * B(0, 1) -
                A(1, 2) * A(2, 1) * B(0, 0) + A(0, 0) * A(1, 1) * B(3, 3) - A(0, 0) * A(1, 3) * B(3, 1) -
                A(0, 0) * A(3, 1) * B(1, 3) + A(0, 0) * A(3, 3) * B(1, 1) - A(0, 1) * A(1, 0) * B(3, 3) +
                A(0, 1) * A(1, 3) * B(3, 0) + A(0, 1) * A(3, 0) * B(1, 3) - A(0, 1) * A(3, 3) * B(1, 0) +
                A(0, 3) * A(1, 0) * B(3, 1) - A(0, 3) * A(1, 1) * B(3, 0) - A(0, 3) * A(3, 0) * B(1, 1) +
                A(0, 3) * A(3, 1) * B(1, 0) + A(1, 0) * A(3, 1) * B(0, 3) - A(1, 0) * A(3, 3) * B(0, 1) -
                A(1, 1) * A(3, 0) * B(0, 3) + A(1, 1) * A(3, 3) * B(0, 0) + A(1, 3) * A(3, 0) * B(0, 1) -
                A(1, 3) * A(3, 1) * B(0, 0) + A(0, 0) * A(2, 2) * B(3, 3) - A(0, 0) * A(2, 3) * B(3, 2) -
                A(0, 0) * A(3, 2) * B(2, 3) + A(0, 0) * A(3, 3) * B(2, 2) - A(0, 2) * A(2, 0) * B(3, 3) +
                A(0, 2) * A(2, 3) * B(3, 0) + A(0, 2) * A(3, 0) * B(2, 3) - A(0, 2) * A(3, 3) * B(2, 0) +
                A(0, 3) * A(2, 0) * B(3, 2) - A(0, 3) * A(2, 2) * B(3, 0) - A(0, 3) * A(3, 0) * B(2, 2) +
                A(0, 3) * A(3, 2) * B(2, 0) + A(2, 0) * A(3, 2) * B(0, 3) - A(2, 0) * A(3, 3) * B(0, 2) -
                A(2, 2) * A(3, 0) * B(0, 3) + A(2, 2) * A(3, 3) * B(0, 0) + A(2, 3) * A(3, 0) * B(0, 2) -
                A(2, 3) * A(3, 2) * B(0, 0) + A(1, 1) * A(2, 2) * B(3, 3) - A(1, 1) * A(2, 3) * B(3, 2) -
                A(1, 1) * A(3, 2) * B(2, 3) + A(1, 1) * A(3, 3) * B(2, 2) - A(1, 2) * A(2, 1) * B(3, 3) +
                A(1, 2) * A(2, 3) * B(3, 1) + A(1, 2) * A(3, 1) * B(2, 3) - A(1, 2) * A(3, 3) * B(2, 1) +
                A(1, 3) * A(2, 1) * B(3, 2) - A(1, 3) * A(2, 2) * B(3, 1) - A(1, 3) * A(3, 1) * B(2, 2) +
                A(1, 3) * A(3, 2) * B(2, 1) + A(2, 1) * A(3, 2) * B(1, 3) - A(2, 1) * A(3, 3) * B(1, 2) -
                A(2, 2) * A(3, 1) * B(1, 3) + A(2, 2) * A(3, 3) * B(1, 1) + A(2, 3) * A(3, 1) * B(1, 2) -
                A(2, 3) * A(3, 2) * B(1, 1) + A(0, 0) * A(1, 1) * A(2, 2) * A(3, 3) -
                A(0, 0) * A(1, 1) * A(2, 3) * A(3, 2) - A(0, 0) * A(1, 2) * A(2, 1) * A(3, 3) +
                A(0, 0) * A(1, 2) * A(2, 3) * A(3, 1) + A(0, 0) * A(1, 3) * A(2, 1) * A(3, 2) -
                A(0, 0) * A(1, 3) * A(2, 2) * A(3, 1) - A(0, 1) * A(1, 0) * A(2, 2) * A(3, 3) +
                A(0, 1) * A(1, 0) * A(2, 3) * A(3, 2) + A(0, 1) * A(1, 2) * A(2, 0) * A(3, 3) -
                A(0, 1) * A(1, 2) * A(2, 3) * A(3, 0) - A(0, 1) * A(1, 3) * A(2, 0) * A(3, 2) +
                A(0, 1) * A(1, 3) * A(2, 2) * A(3, 0) + A(0, 2) * A(1, 0) * A(2, 1) * A(3, 3) -
                A(0, 2) * A(1, 0) * A(2, 3) * A(3, 1) - A(0, 2) * A(1, 1) * A(2, 0) * A(3, 3) +
                A(0, 2) * A(1, 1) * A(2, 3) * A(3, 0) + A(0, 2) * A(1, 3) * A(2, 0) * A(3, 1) -
                A(0, 2) * A(1, 3) * A(2, 1) * A(3, 0) - A(0, 3) * A(1, 0) * A(2, 1) * A(3, 2) +
                A(0, 3) * A(1, 0) * A(2, 2) * A(3, 1) + A(0, 3) * A(1, 1) * A(2, 0) * A(3, 2) -
                A(0, 3) * A(1, 1) * A(2, 2) * A(3, 0) - A(0, 3) * A(1, 2) * A(2, 0) * A(3, 1) +
                A(0, 3) * A(1, 2) * A(2, 1) * A(3, 0);
    coeffs[5] = A(0, 0) * B(1, 1) - A(0, 1) * B(1, 0) - A(1, 0) * B(0, 1) + A(1, 1) * B(0, 0) + A(0, 0) * B(2, 2) -
                A(0, 2) * B(2, 0) - A(2, 0) * B(0, 2) + A(2, 2) * B(0, 0) + A(0, 0) * B(3, 3) - A(0, 3) * B(3, 0) +
                A(1, 1) * B(2, 2) - A(1, 2) * B(2, 1) - A(2, 1) * B(1, 2) + A(2, 2) * B(1, 1) - A(3, 0) * B(0, 3) +
                A(3, 3) * B(0, 0) + A(1, 1) * B(3, 3) - A(1, 3) * B(3, 1) - A(3, 1) * B(1, 3) + A(3, 3) * B(1, 1) +
                A(2, 2) * B(3, 3) - A(2, 3) * B(3, 2) - A(3, 2) * B(2, 3) + A(3, 3) * B(2, 2) +
                A(0, 0) * A(1, 1) * A(2, 2) - A(0, 0) * A(1, 2) * A(2, 1) - A(0, 1) * A(1, 0) * A(2, 2) +
                A(0, 1) * A(1, 2) * A(2, 0) + A(0, 2) * A(1, 0) * A(2, 1) - A(0, 2) * A(1, 1) * A(2, 0) +
                A(0, 0) * A(1, 1) * A(3, 3) - A(0, 0) * A(1, 3) * A(3, 1) - A(0, 1) * A(1, 0) * A(3, 3) +
                A(0, 1) * A(1, 3) * A(3, 0) + A(0, 3) * A(1, 0) * A(3, 1) - A(0, 3) * A(1, 1) * A(3, 0) +
                A(0, 0) * A(2, 2) * A(3, 3) - A(0, 0) * A(2, 3) * A(3, 2) - A(0, 2) * A(2, 0) * A(3, 3) +
                A(0, 2) * A(2, 3) * A(3, 0) + A(0, 3) * A(2, 0) * A(3, 2) - A(0, 3) * A(2, 2) * A(3, 0) +
                A(1, 1) * A(2, 2) * A(3, 3) - A(1, 1) * A(2, 3) * A(3, 2) - A(1, 2) * A(2, 1) * A(3, 3) +
                A(1, 2) * A(2, 3) * A(3, 1) + A(1, 3) * A(2, 1) * A(3, 2) - A(1, 3) * A(2, 2) * A(3, 1);
    coeffs[6] = B(0, 0) + B(1, 1) + B(2, 2) + B(3, 3) + A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) -
                A(0, 2) * A(2, 0) + A(0, 0) * A(3, 3) - A(0, 3) * A(3, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1) +
                A(1, 1) * A(3, 3) - A(1, 3) * A(3, 1) + A(2, 2) * A(3, 3) - A(2, 3) * A(3, 2);
    coeffs[7] = A(0, 0) + A(1, 1) + A(2, 2) + A(3, 3);
    coeffs[8] = 1;
}

// Computes polynomial p(x) = det(x^2*I + x * A + B)
void detpoly3(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B, double coeffs[7]) {
    coeffs[0] = B(0, 0) * B(1, 1) * B(2, 2) - B(0, 0) * B(1, 2) * B(2, 1) - B(0, 1) * B(1, 0) * B(2, 2) +
                B(0, 1) * B(1, 2) * B(2, 0) + B(0, 2) * B(1, 0) * B(2, 1) - B(0, 2) * B(1, 1) * B(2, 0);
    coeffs[1] = A(0, 0) * B(1, 1) * B(2, 2) - A(0, 0) * B(1, 2) * B(2, 1) - A(0, 1) * B(1, 0) * B(2, 2) +
                A(0, 1) * B(1, 2) * B(2, 0) + A(0, 2) * B(1, 0) * B(2, 1) - A(0, 2) * B(1, 1) * B(2, 0) -
                A(1, 0) * B(0, 1) * B(2, 2) + A(1, 0) * B(0, 2) * B(2, 1) + A(1, 1) * B(0, 0) * B(2, 2) -
                A(1, 1) * B(0, 2) * B(2, 0) - A(1, 2) * B(0, 0) * B(2, 1) + A(1, 2) * B(0, 1) * B(2, 0) +
                A(2, 0) * B(0, 1) * B(1, 2) - A(2, 0) * B(0, 2) * B(1, 1) - A(2, 1) * B(0, 0) * B(1, 2) +
                A(2, 1) * B(0, 2) * B(1, 0) + A(2, 2) * B(0, 0) * B(1, 1) - A(2, 2) * B(0, 1) * B(1, 0);
    coeffs[2] = B(0, 0) * B(1, 1) - B(0, 1) * B(1, 0) + B(0, 0) * B(2, 2) - B(0, 2) * B(2, 0) + B(1, 1) * B(2, 2) -
                B(1, 2) * B(2, 1) + A(0, 0) * A(1, 1) * B(2, 2) - A(0, 0) * A(1, 2) * B(2, 1) -
                A(0, 0) * A(2, 1) * B(1, 2) + A(0, 0) * A(2, 2) * B(1, 1) - A(0, 1) * A(1, 0) * B(2, 2) +
                A(0, 1) * A(1, 2) * B(2, 0) + A(0, 1) * A(2, 0) * B(1, 2) - A(0, 1) * A(2, 2) * B(1, 0) +
                A(0, 2) * A(1, 0) * B(2, 1) - A(0, 2) * A(1, 1) * B(2, 0) - A(0, 2) * A(2, 0) * B(1, 1) +
                A(0, 2) * A(2, 1) * B(1, 0) + A(1, 0) * A(2, 1) * B(0, 2) - A(1, 0) * A(2, 2) * B(0, 1) -
                A(1, 1) * A(2, 0) * B(0, 2) + A(1, 1) * A(2, 2) * B(0, 0) + A(1, 2) * A(2, 0) * B(0, 1) -
                A(1, 2) * A(2, 1) * B(0, 0);
    coeffs[3] = A(0, 0) * B(1, 1) - A(0, 1) * B(1, 0) - A(1, 0) * B(0, 1) + A(1, 1) * B(0, 0) + A(0, 0) * B(2, 2) -
                A(0, 2) * B(2, 0) - A(2, 0) * B(0, 2) + A(2, 2) * B(0, 0) + A(1, 1) * B(2, 2) - A(1, 2) * B(2, 1) -
                A(2, 1) * B(1, 2) + A(2, 2) * B(1, 1) + A(0, 0) * A(1, 1) * A(2, 2) - A(0, 0) * A(1, 2) * A(2, 1) -
                A(0, 1) * A(1, 0) * A(2, 2) + A(0, 1) * A(1, 2) * A(2, 0) + A(0, 2) * A(1, 0) * A(2, 1) -
                A(0, 2) * A(1, 1) * A(2, 0);
    coeffs[4] = B(0, 0) + B(1, 1) + B(2, 2) + A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) -
                A(0, 2) * A(2, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
    coeffs[5] = A(0, 0) + A(1, 1) + A(2, 2);
    coeffs[6] = 1.0;
}

int qep_sturm(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B,
              const Eigen::Matrix<double, 4, 4> &C, double eig_vals[8], Eigen::Matrix<double, 3, 8> *eig_vecs) {

    double coeffs[9];

    Eigen::Matrix<double, 4, 4> Ainv = A.inverse();
    detpoly4(Ainv * B, Ainv * C, coeffs);

    int n_roots = sturm::bisect_sturm<8>(coeffs, eig_vals);

    // For computing the eigenvectors we try to use the top 3x3 block only. If this fails we revert to QR on the 4x3
    // system
    Eigen::Matrix<double, 4, 4> M;
    Eigen::Matrix<double, 3, 3> Minv;
    bool invertible;
    for (int i = 0; i < n_roots; ++i) {
        M = (eig_vals[i] * eig_vals[i]) * A + eig_vals[i] * B + C;
        M.block<3, 3>(0, 0).computeInverseWithCheck(Minv, invertible, 1e-8);

        if (invertible) {
            eig_vecs->col(i) = -Minv * M.block<3, 1>(0, 3);
        } else {
            eig_vecs->col(i) = -M.block<4, 3>(0, 0).colPivHouseholderQr().solve(M.block<4, 1>(0, 3));
        }
    }

    return n_roots;
}

int qep_sturm_div_1_q2(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B,
                       const Eigen::Matrix<double, 4, 4> &C, double eig_vals[6],
                       Eigen::Matrix<double, 3, 6> *eig_vecs) {

    double coeffs[9];

    Eigen::Matrix<double, 4, 4> Ainv = A.inverse();
    detpoly4(Ainv * B, Ainv * C, coeffs);

    // We know that (1+q*q) is a factor. Dividing by this gives us a deg 6 poly.
    coeffs[2] = coeffs[2] - coeffs[0];
    coeffs[3] = coeffs[3] - coeffs[1];
    coeffs[4] = coeffs[4] - coeffs[2];
    coeffs[5] = coeffs[7];
    coeffs[6] = coeffs[8];

    int n_roots = sturm::bisect_sturm<6>(coeffs, eig_vals);

    // For computing the eigenvectors we try to use the top 3x3 block only. If this fails we revert to QR on the 4x3
    // system
    Eigen::Matrix<double, 4, 4> M;
    Eigen::Matrix<double, 3, 3> Minv;
    bool invertible;
    for (int i = 0; i < n_roots; ++i) {
        M = (eig_vals[i] * eig_vals[i]) * A + eig_vals[i] * B + C;
        M.block<3, 3>(0, 0).computeInverseWithCheck(Minv, invertible, 1e-8);

        if (invertible) {
            eig_vecs->col(i) = -Minv * M.block<3, 1>(0, 3);
        } else {
            eig_vecs->col(i) = -M.block<4, 3>(0, 0).colPivHouseholderQr().solve(M.block<4, 1>(0, 3));
        }
    }

    return n_roots;
}

int qep_sturm(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B,
              const Eigen::Matrix<double, 3, 3> &C, double eig_vals[6], Eigen::Matrix<double, 3, 6> *eig_vecs) {

    double coeffs[7];

    Eigen::Matrix<double, 3, 3> Ainv = A.inverse();
    detpoly3(Ainv * B, Ainv * C, coeffs);

    int n_roots = sturm::bisect_sturm<6>(coeffs, eig_vals);

    // For computing the eigenvectors we first try to use the top 2x3 block only.
    Eigen::Matrix<double, 3, 3> M;
    for (int i = 0; i < n_roots; ++i) {
        M = (eig_vals[i] * eig_vals[i]) * A + eig_vals[i] * B + C;

        Eigen::Vector3d t = M.row(0).cross(M.row(1)).normalized();
        if (std::abs(M.row(2) * t) > 1e-8) {
            t = M.row(0).cross(M.row(2)).normalized();
            if (std::abs(M.row(1) * t) > 1e-8) {
                t = M.row(1).cross(M.row(2)).normalized();
            }
        }

        eig_vecs->col(i) = t;
    }

    return n_roots;
}

int qep_div_1_q2(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B,
                 const Eigen::Matrix<double, 3, 3> &C, double eig_vals[4], Eigen::Matrix<double, 3, 4> *eig_vecs) {

    double coeffs[7];

    Eigen::Matrix<double, 3, 3> Ainv = A.inverse();
    detpoly3(Ainv * B, Ainv * C, coeffs);

    int n_roots = univariate::solve_quartic_real(coeffs[5], coeffs[2] - coeffs[0], coeffs[1], coeffs[0], eig_vals);

    // For computing the eigenvectors we first try to use the top 2x3 block only.
    Eigen::Matrix<double, 3, 3> M;
    for (int i = 0; i < n_roots; ++i) {
        M = (eig_vals[i] * eig_vals[i]) * A + eig_vals[i] * B + C;

        Eigen::Vector3d t = M.row(0).cross(M.row(1)).normalized();
        if (std::abs(M.row(2) * t) > 1e-8) {
            t = M.row(0).cross(M.row(2)).normalized();
            if (std::abs(M.row(1) * t) > 1e-8) {
                t = M.row(1).cross(M.row(2)).normalized();
            }
        }

        eig_vecs->col(i) = t;
    }

    return n_roots;
}

} // namespace qep
} // namespace poselib