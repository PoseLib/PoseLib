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

#include "homography.h"

#include "PoseLib/robust/bundle.h"
#include "PoseLib/solvers/homography_4pt.h"

namespace poselib {

void HomographyEstimator::generate_models(std::vector<Eigen::Matrix3d> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    Eigen::Matrix3d H;
    int sols = homography_4pt(x1s, x2s, &H, true);
    if (sols > 0) {
        models->push_back(H);
    }
}

double HomographyEstimator::score_model(const Eigen::Matrix3d &H, size_t *inlier_count) const {
    return compute_homography_msac_score(H, x1, x2, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
}

void HomographyEstimator::refine_model(Eigen::Matrix3d *H) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    refine_homography(x1, x2, H, bundle_opt);
}

} // namespace poselib