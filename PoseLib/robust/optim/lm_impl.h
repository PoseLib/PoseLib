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

#ifndef POSELIB_OPTIM_LM_IMPL_
#define POSELIB_OPTIM_LM_IMPL_

#include "PoseLib/robust/optim/jacobian_accumulator.h"
#include "PoseLib/robust/robust_loss.h"
#include "PoseLib/types.h"
#include "optim_utils.h"

#include <memory>
namespace poselib {

/*
 Templated implementation of Levenberg-Marquadt.

 The Problem class must provide
    Problem::num_params - number of parameters to optimize over
    Problem::params_t - type for the parameters which optimize over
    Problem::compute_jacobian(acc, param) - compute jacobians
    Problem::compute_residual(acc, param) - compute the current residuals
    Problem::step(delta_params, param) - take a step in parameter space

    Check out refiner_base.h for the interface
*/

typedef std::function<void(const BundleStats &stats, RobustLoss *loss_fn)> IterationCallback;
// Callback which prints debug info from the iterations
void print_iteration(const BundleStats &stats, RobustLoss *loss_fn);

template <typename Problem, typename Accumulator = NormalAccumulator, typename Model = typename Problem::param_t>
BundleStats lm_impl(Problem &problem, Model *parameters, const BundleOptions &opt,
                    IterationCallback callback = nullptr) {

    std::shared_ptr<RobustLoss> loss_fn(RobustLoss::factory(opt));

    // Initialize
    BundleStats stats;
    Accumulator acc;
    acc.initialize(problem.num_params, loss_fn);
    acc.reset_residual();
    stats.cost = problem.compute_residual(acc, *parameters);
    stats.initial_cost = stats.cost;
    stats.grad_norm = -1;
    stats.step_norm = -1;
    stats.invalid_steps = 0;
    stats.lambda = opt.initial_lambda;

    bool recompute_jac = true;
    for (stats.iterations = 0; stats.iterations < opt.max_iterations; ++stats.iterations) {
        // We only recompute jacobian and residual vector if last step was successful
        if (recompute_jac) {
            acc.reset_jacobian();
            problem.compute_jacobian(acc, *parameters);
            stats.grad_norm = acc.grad_norm();
            if (stats.grad_norm < opt.gradient_tol) {
                break;
            }
        }

        Eigen::VectorXd sol = acc.solve(stats.lambda);
        stats.step_norm = sol.norm();
        if (stats.step_norm < opt.step_tol) {
            break;
        }

        Model parameters_new = problem.step(sol, *parameters);
        acc.reset_residual();
        double cost_new = problem.compute_residual(acc, parameters_new);

        if (cost_new < stats.cost) {
            *parameters = parameters_new;
            stats.lambda = std::max(opt.min_lambda, stats.lambda / 10);
            stats.cost = cost_new;
            recompute_jac = true;
        } else {
            stats.invalid_steps++;
            stats.lambda = std::min(opt.max_lambda, stats.lambda * 10);
            recompute_jac = false;
        }
        if (callback != nullptr) {
            callback(stats, loss_fn.get());
        }
    }
    return stats;
}

} // namespace poselib

#endif