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

#ifndef POSELIB_ROBUST_LOSS_H_
#define POSELIB_ROBUST_LOSS_H_

namespace poselib {

// Robust loss functions
class TrivialLoss {
  public:
    TrivialLoss(real_t) {} // dummy to ensure we have consistent calling interface
    TrivialLoss() {}
    real_t loss(real_t r2) const { return r2; }
    real_t weight(real_t r2) const { return 1.0; }
};

class TruncatedLoss {
  public:
    TruncatedLoss(real_t threshold) : squared_thr(threshold * threshold) {}
    real_t loss(real_t r2) const { return std::min(r2, squared_thr); }
    real_t weight(real_t r2) const { return (r2 < squared_thr) ? 1.0 : 0.0; }

  private:
    const real_t squared_thr;
};

// The method from
//  Le and Zach, Robust Fitting with Truncated Least Squares: A Bilevel Optimization Approach, 3DV 2021
// for truncated least squares optimization with IRLS.
class TruncatedLossLeZach {
  public:
    TruncatedLossLeZach(real_t threshold) : squared_thr(threshold * threshold), mu(0.5) {}
    real_t loss(real_t r2) const { return std::min(r2, squared_thr); }
    real_t weight(real_t r2) const {
        real_t r2_hat = r2 / squared_thr;
        real_t zstar = std::min(r2_hat, (real_t)1.0);

        if (r2_hat < 1.0) {
            return 0.5;
        } else {
            // assumes mu > 0.5
            real_t r2m1 = r2_hat - 1.0;
            real_t rho = (2.0 * r2m1 + std::sqrt(4.0 * r2m1 * r2m1 * mu * mu + 2 * mu * r2m1)) / mu;
            real_t a = (r2_hat + mu * rho * zstar - 0.5 * rho) / (1 + mu * rho);
            real_t zbar = std::max((real_t)0.0, std::min(a, (real_t)1.0));
            return (zstar - zbar) / rho;
        }
    }

  private:
    const real_t squared_thr;

  public:
    // hyper-parameter for penalty strength
    real_t mu;
    // schedule for increasing mu in each iteration
    static constexpr real_t alpha = 1.5;
};

class HuberLoss {
  public:
    HuberLoss(real_t threshold) : thr(threshold) {}
    real_t loss(real_t r2) const {
        const real_t r = std::sqrt(r2);
        if (r <= thr) {
            return r2;
        } else {
            return thr * (2.0 * r - thr);
        }
    }
    real_t weight(real_t r2) const {
        const real_t r = std::sqrt(r2);
        if (r <= thr) {
            return 1.0;
        } else {
            return thr / r;
        }
    }

  private:
    const real_t thr;
};
class CauchyLoss {
  public:
    CauchyLoss(real_t threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}
    real_t loss(real_t r2) const { return std::log1p(r2 * inv_sq_thr); }
    real_t weight(real_t r2) const {
        return std::max(std::numeric_limits<real_t>::min(), inv_sq_thr / (1.0f + r2 * inv_sq_thr));
    }

  private:
    const real_t inv_sq_thr;
};

} // namespace poselib

#endif