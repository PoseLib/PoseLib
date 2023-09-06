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

#ifndef POSELIB_JAC_ACC_H_
#define POSELIB_JAC_ACC_H_

#include "../../types.h"
#include "optim_utils.h"
#include <PoseLib/robust/robust_loss.h>

namespace poselib {

template<typename RobustLoss = TrivialLoss>
class NormalAccumulator {
public:
    NormalAccumulator(int p_dim, RobustLoss loss = RobustLoss(), double res_scale = 1.0): param_dim(p_dim), loss_fcn(loss), residual_scale(res_scale) {
        JtJ.resize(param_dim, param_dim);
        Jtr.resize(param_dim, 1); 
    }

    void reset_residual() {
        residual_acc = 0;
    }
    void add_residual(const double res,
                      const double w = 1.0) {        
        residual_acc += w * loss_fcn.loss(res * res);
    }
    template<typename Derived> 
    void add_residual(const Eigen::MatrixBase<Derived> &res,
                      const double w = 1.0) {
        const double r_squared = res.squaredNorm();
        residual_acc += w * loss_fcn.loss(r_squared);
    }
    double get_residual() const {
        return residual_acc;
    }

    void reset_jacobian() {
        JtJ.setZero();
        Jtr.setZero();
    }
    template<typename Derived1, typename Derived2>
    void add_jacobian(const Eigen::MatrixBase<Derived1> &res,
                        const Eigen::MatrixBase<Derived2> &jac,
                        const double w = 1.0) {
        const double r_squared = res.squaredNorm();
        const double weight = w * loss_fcn.weight(r_squared);
        for(int i = 0; i < param_dim; ++i) {
            for(int j = 0; j <= i; ++j) {
                JtJ(i,j) += weight * (jac.col(i).dot(jac.col(j)));
            }
            Jtr(i) += weight * jac.col(i).dot(res);
        }
    }

    // Residuals that are 1-dim
    template<typename Derived>
    void add_jacobian(const double res,
                      const Eigen::MatrixBase<Derived> &jac,
                      const double w = 1.0) {
        const double r_squared = res * res;
        const double weight = w * loss_fcn.weight(r_squared);
        for(int i = 0; i < param_dim; ++i) {
            for(int j = 0; j <= i; ++j) {
                JtJ(i,j) += weight * (jac(i) * jac(j));
            }
            Jtr(i) += weight * jac(i) * res;
        }
    }

    double grad_norm() const {
        return Jtr.norm() * residual_scale;
    }

    Eigen::VectorXd solve(double lambda) {
        for(int i = 0; i < param_dim; ++i) {
            JtJ(i,i) += lambda;
        }

        Eigen::VectorXd sol = (residual_scale * JtJ).selfadjointView<Eigen::Lower>().llt().solve(-(residual_scale*Jtr));

        // Restore JtJ in-case we need it again
        for(int i = 0; i < param_dim; ++i) {
            JtJ(i,i) -= lambda;
        }

        return sol;
    }

    double residual_acc;
    int param_dim;
    RobustLoss loss_fcn;
    double residual_scale;
    Eigen::MatrixXd JtJ;
    Eigen::VectorXd Jtr;
};


}

#endif