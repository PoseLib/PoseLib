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
    TrivialLoss(double) {} // dummy to ensure we have consistent calling interface
    TrivialLoss() {}
    double loss(double r2) const { return r2; }
    double weight(double r2) const { return 1.0; }
};

class TruncatedLoss {
  public:
    TruncatedLoss(double threshold) : squared_thr(threshold * threshold) {}
    double loss(double r2) const { return std::min(r2, squared_thr); }
    double weight(double r2) const { return (r2 < squared_thr) ? 1.0 : 0.0; }

  private:
    const double squared_thr;
};

// The method from
//  Le and Zach, Robust Fitting with Truncated Least Squares: A Bilevel Optimization Approach, 3DV 2021
// for truncated least squares optimization with IRLS.
class TruncatedLossLeZach {
  public:
    TruncatedLossLeZach(double threshold) : squared_thr(threshold * threshold), mu(0.5) {}
    double loss(double r2) const { return std::min(r2, squared_thr); }
    double weight(double r2) const {
        double r2_hat = r2 / squared_thr;
        double zstar = std::min(r2_hat, 1.0);

        if (r2_hat < 1.0) {
            return 0.5;
        } else {
            // assumes mu > 0.5
            double r2m1 = r2_hat - 1.0;
            double rho = (2.0 * r2m1 + std::sqrt(4.0 * r2m1 * r2m1 * mu * mu + 2 * mu * r2m1)) / mu;
            double a = (r2_hat + mu * rho * zstar - 0.5 * rho) / (1 + mu * rho);
            double zbar = std::max(0.0, std::min(a, 1.0));
            return (zstar - zbar) / rho;
        }
    }

  private:
    const double squared_thr;

  public:
    // hyper-parameter for penalty strength
    double mu;
    // schedule for increasing mu in each iteration
    static constexpr double alpha = 1.5;
};

class HuberLoss {
  public:
    HuberLoss(double threshold) : thr(threshold) {}
    double loss(double r2) const {
        const double r = std::sqrt(r2);
        if (r <= thr) {
            return r2;
        } else {
            return thr * (2.0 * r - thr);
        }
    }
    double weight(double r2) const {
        const double r = std::sqrt(r2);
        if (r <= thr) {
            return 1.0;
        } else {
            return thr / r;
        }
    }

  private:
    const double thr;
};
class CauchyLoss {
  public:
    CauchyLoss(double threshold) : inv_sq_thr(1.0 / (threshold * threshold)) {}
    double loss(double r2) const { return std::log1p(r2 * inv_sq_thr); }
    double weight(double r2) const {
        return std::max(std::numeric_limits<double>::min(), inv_sq_thr / (1.0 + r2 * inv_sq_thr));
    }

  private:
    const double inv_sq_thr;
};

} // namespace poselib

#endif