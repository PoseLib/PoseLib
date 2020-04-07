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

#pragma once
#include <Eigen/Dense>

namespace pose_lib {
namespace qep {

// Helper functions for solving quadratic eigenvalue problems
// (Currently only 4x4 problems are implemented)
// Note: The impl. assumes that fourth element of eigenvector is non-zero.
// The return eigenvectors are only the first three elements (fourth is normalized to 1)

// Solves the QEP by reduction to normal eigenvalue problem
int qep_linearize(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B, const Eigen::Matrix<double, 4, 4> &C, double eig_vals[8], Eigen::Matrix<double, 3, 8> *eig_vecs);

// Solves the QEP by sturm bracketing on det(lambda^2*A + lambda*B + C)
int qep_sturm(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B, const Eigen::Matrix<double, 4, 4> &C, double eig_vals[8], Eigen::Matrix<double, 3, 8> *eig_vecs);

} // namespace qep
} // namespace pose_lib