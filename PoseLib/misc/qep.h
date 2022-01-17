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

#ifndef POSELIB_MISC_QEP_H_
#define POSELIB_MISC_QEP_H_

#include <Eigen/Dense>

namespace poselib {
namespace qep {

// Helper functions for solving quadratic eigenvalue problems
// (Currently only 4x4 problems are implemented)
// Note: The impl. assumes that fourth element of eigenvector is non-zero.
// The return eigenvectors are only the first three elements (fourth is normalized to 1)

// Solves the QEP by reduction to normal eigenvalue problem
int qep_linearize(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B,
                  const Eigen::Matrix<double, 4, 4> &C, double eig_vals[8], Eigen::Matrix<double, 3, 8> *eig_vecs);

// Solves the QEP by sturm bracketing on det(lambda^2*A + lambda*B + C)
int qep_sturm(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B,
              const Eigen::Matrix<double, 4, 4> &C, double eig_vals[8], Eigen::Matrix<double, 3, 8> *eig_vecs);

// Solves the QEP by solving det(lambda^2*A + lambda*B + C) where we know that (1+lambda^2) is a factor.
// This is the case in the upright solvers from Sweeney et al.
// The roots are found using sturm bracketing.
int qep_sturm_div_1_q2(const Eigen::Matrix<double, 4, 4> &A, const Eigen::Matrix<double, 4, 4> &B,
                       const Eigen::Matrix<double, 4, 4> &C, double eig_vals[6], Eigen::Matrix<double, 3, 6> *eig_vecs);

// Solves the QEP by sturm bracketing on det(lambda^2*A + lambda*B + C)
int qep_sturm(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B,
              const Eigen::Matrix<double, 3, 3> &C, double eig_vals[6], Eigen::Matrix<double, 3, 6> *eig_vecs);

// Solves the QEP by solving det(lambda^2*A + lambda*B + C) where we know that (1+lambda^2) is a factor.
// This is the case in the upright solvers from Sweeney et al.
// The roots are found using the closed form solver for the quartic.
int qep_div_1_q2(const Eigen::Matrix<double, 3, 3> &A, const Eigen::Matrix<double, 3, 3> &B,
                 const Eigen::Matrix<double, 3, 3> &C, double eig_vals[4], Eigen::Matrix<double, 3, 4> *eig_vecs);

} // namespace qep
} // namespace poselib

#endif