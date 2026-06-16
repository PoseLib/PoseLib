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

#include "p4llf.h"

#include "pnplf_impl.h"

namespace poselib {

int p4llf(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
          const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output, std::vector<double> *output_focal,
          bool filter_solutions) {

    std::vector<CameraPose> poses;
    std::vector<double> fx, fy;
    int n = p4llf(l, X, V, &poses, &fx, &fy, filter_solutions);

    if (filter_solutions) {
        int best_ind = -1;
        double best_err = 1.0;
        for (int i = 0; i < n; ++i) {
            double a = fx[i] / fy[i];
            double err = std::max(std::abs(a - 1.0), std::abs(1 / a - 1.0));
            if (err < best_err) {
                best_err = err;
                best_ind = i;
            }
        }
        if (best_err < 1.0 && best_ind > -1) {
            output_focal->push_back((fx[best_ind] + fy[best_ind]) / 2.0);
            output->push_back(poses[best_ind]);
        }
    } else {
        *output = poses;
        output_focal->resize(n);
        for (int i = 0; i < n; ++i) {
            (*output_focal)[i] = (fx[i] + fy[i]) / 2.0;
        }
    }
    return output->size();
}

int p4llf(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
          const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output, std::vector<double> *output_fx,
          std::vector<double> *output_fy, bool filter_solutions) {
    std::vector<Eigen::Vector2d> xp_empty;
    std::vector<Eigen::Vector3d> Xp_empty;
    return detail::pnplf_impl(xp_empty, Xp_empty, l, X, V, output, output_fx, output_fy, filter_solutions);
}

} // namespace poselib
