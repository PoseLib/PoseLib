// Copyright (c) 2023, Viktor Larsson
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
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/robust_loss.h>
#include <PoseLib/types.h>

#ifndef POSELIB_OPTIM_TEST_UTILS_H_
#define POSELIB_OPTIM_TEST_UTILS_H_

using namespace poselib;

class TestAccumulator {
  public:
    TestAccumulator() {}

    void reset_residual() { residual_acc = 0; }
    void add_residual(const double res, const double w = 1.0) { residual_acc += w * res * res; }
    template <typename Derived> void add_residual(const Eigen::MatrixBase<Derived> &res, const double w = 1.0) {
        residual_acc += w * res.squaredNorm();
    }
    double get_residual() const { return residual_acc; }

    void reset_jacobian() {
        Js.clear();
        rs.clear();
        weights.clear();
    }
    template <typename Derived>
    void add_jacobian(const double res, const Eigen::MatrixBase<Derived> &jac, const double w = 1.0) {
        Js.push_back(jac);
        rs.push_back(Eigen::Matrix<double, 1, 1>(res));
        weights.push_back(w);
    }
    template <typename Derived1, typename Derived2>
    void add_jacobian(const Eigen::MatrixBase<Derived1> &res, const Eigen::MatrixBase<Derived2> &jac,
                      const double w = 1.0) {
        Js.push_back(jac);
        rs.push_back(res);
        weights.push_back(w);
    }

    double residual_acc;
    std::vector<Eigen::MatrixXd> Js;
    std::vector<Eigen::VectorXd> rs;
    std::vector<double> weights;

    // Debug stuff
    Eigen::MatrixXd JtJ;
    Eigen::VectorXd Jtr;
    TrivialLoss loss_fcn;
};

template <typename Refiner, typename Model> double verify_jacobian(Refiner &refiner, const Model &m, double delta) {
    int num_params = refiner.num_params;
    TestAccumulator acc;
    std::vector<TestAccumulator> acc_backward(num_params);
    std::vector<TestAccumulator> acc_forward(num_params);

    // Compute residual and jacobian
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, m);

    int num_res = acc.rs.size();

    // Compute finite differences
    std::vector<Eigen::MatrixXd> J_est;
    J_est.resize(num_res);
    for (int k = 0; k < num_res; ++k) {
        J_est[k].resize(acc.Js[k].rows(), acc.Js[k].cols());
    }

    for (int i = 0; i < num_params; ++i) {
        Eigen::Matrix<double, Eigen::Dynamic, 1> dp(num_params, 1);
        dp.setZero();
        dp(i) = delta;

        Model m_forward = refiner.step(dp, m);
        Model m_backward = refiner.step(-dp, m);

        acc_backward[i].reset_jacobian();
        acc_forward[i].reset_jacobian();
        refiner.compute_jacobian(acc_forward[i], m_forward);
        refiner.compute_jacobian(acc_backward[i], m_backward);

        for (int k = 0; k < num_res; ++k) {
            J_est[k].col(i) = (acc_forward[i].rs[k] - acc_backward[i].rs[k]) / (2.0 * delta);
        }
    }

    // std::cout << "J_est=\n" << J_est[0] << "\nJ=\n" << acc.Js[0] << "\n";
    // std::cout << "------\n";

    // Compare the jacobian with the residual
    double max_err = 0.0;
    for (int k = 0; k < num_res; ++k) {
        double err = (J_est[k] - acc.Js[k]).norm() / std::max(1.0, J_est[k].norm());
        max_err = std::max(err, max_err);
        if (max_err > 1e-3) {
            std::cout << "Jacobian failure! \n J=\n" << acc.Js[k] << "\nJ (finite) = \n" << J_est[k] << "\n";
            break;
        }
    }
    return max_err;
}

#endif
