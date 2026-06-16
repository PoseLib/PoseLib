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
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_COVARIANCE_H_
#define POSELIB_COVARIANCE_H_

#include "../robust_loss.h"
#include "jacobian_accumulator.h"

#include <Eigen/Dense>
#include <memory>

namespace poselib {

// Pseudo-inverse of a symmetric positive semi-definite matrix via eigendecomposition.
// Eigenvalues below tol * (largest eigenvalue) are treated as zero (dropped). This handles
// the rank-deficient case that arises when embedding a minimal covariance into the ambient
// (over-parametrized) space.
inline Eigen::MatrixXd pseudo_inverse_psd(const Eigen::MatrixXd &A) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(A);
    const Eigen::VectorXd &eigenvalues = solver.eigenvalues();
    Eigen::MatrixXd inv = Eigen::MatrixXd::Zero(A.rows(), A.cols());
    if (eigenvalues.size() == 0) {
        return inv;
    }
    const double tol = 1e-10 * eigenvalues.maxCoeff();
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) > tol) {
            inv += (1.0 / eigenvalues(i)) * solver.eigenvectors().col(i) *
                   solver.eigenvectors().col(i).transpose();
        }
    }
    return inv;
}

// Estimate the covariance of a refined model as the (pseudo-)inverse of the Gauss-Newton
// normal equations J'J evaluated at the solution, using the same robust loss as the
// refinement.
//
// If full == false, the covariance is returned in the minimal tangent parametrization
// (k x k, full rank). If full == true, it is mapped into the model's ambient
// parametrization (N x N, rank k, i.e. rank deficient) via the refiner's embedding
// Jacobian J (N x k):  cov_full = J * cov_minimal * J'.
template <typename Refiner>
Eigen::MatrixXd estimate_model_covariance(Refiner &refiner, const typename Refiner::param_t &param,
                                          std::shared_ptr<RobustLoss> loss, bool full) {
    NormalAccumulator acc;
    acc.initialize(refiner.num_params, loss);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, param);

    Eigen::MatrixXd JtJ = acc.JtJ.selfadjointView<Eigen::Lower>();
    Eigen::MatrixXd cov = pseudo_inverse_psd(JtJ);

    if (full) {
        Eigen::MatrixXd J = refiner.embedding_jacobian(param);
        cov = J * cov * J.transpose();
    }
    return cov;
}

} // namespace poselib

#endif
