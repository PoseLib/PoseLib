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

#ifndef POSELIB_HYBRID_H_
#define POSELIB_HYBRID_H_

#include "../../types.h"
#include "optim_utils.h"
#include "refiner_base.h"

namespace poselib {

// Composite refiner
// Note: Requires all refiners to have the same model and step-function!
template <typename Accumulator, typename Model = CameraPose, int NumParams = 6>
class HybridRefiner : public RefinerBase<Accumulator, Model> {
  public:
    HybridRefiner() {}

    double compute_residual(Accumulator &acc, const Model &model) {
        for (RefinerBase<Accumulator, Model> *ref : refiners) {
            ref->compute_residual(acc, model);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const Model &model) {
        for (RefinerBase<Accumulator, Model> *ref : refiners) {
            ref->compute_jacobian(acc, model);
        }
    }

    Model step(const Eigen::VectorXd &dp, const Model &model) const {
        // Assumption is that all step() functions are the same
        return refiners[0]->step(dp, model);
    }

    void register_refiner(RefinerBase<Accumulator, Model> *ref) { refiners.push_back(ref); }

    typedef Model param_t;
    static constexpr size_t num_params = NumParams;
    std::vector<RefinerBase<Accumulator, Model> *> refiners;
};

} // namespace poselib

#endif