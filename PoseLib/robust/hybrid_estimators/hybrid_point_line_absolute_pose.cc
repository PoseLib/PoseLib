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
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "hybrid_point_line_absolute_pose.h"

#include <PoseLib/robust/bundle.h>
#include <PoseLib/robust/utils.h>
#include <PoseLib/solvers/p1p2ll.h>
#include <PoseLib/solvers/p2p1ll.h>
#include <PoseLib/solvers/p3ll.h>
#include <PoseLib/solvers/p3p.h>
#include <numeric>
#include <stdexcept>

namespace poselib {

HybridPointLineAbsolutePoseEstimator::HybridPointLineAbsolutePoseEstimator(const HybridRansacOptions &opt,
                                                         const std::vector<Point2D> &points2D,
                                                         const std::vector<Point3D> &points3D,
                                                         const std::vector<Line2D> &lines2D,
                                                         const std::vector<Line3D> &lines3D)
    : opt_(opt), points2D_(points2D), points3D_(points3D), lines2D_(lines2D), lines3D_(lines3D) {
    rng_.seed(opt.seed);

    // Validate max_errors has at least 2 elements (point and line thresholds)
    if (opt_.max_errors.size() < 2) {
        throw std::invalid_argument("HybridRansacOptions::max_errors must have at least 2 elements "
                                    "(point and line error thresholds)");
    }

    // Pre-allocate buffers for minimal solvers
    xs_.resize(3);
    Xs_.resize(3);
    ls_.resize(3);
    Cs_.resize(3);
    Vs_.resize(3);

    // Initialize cached inlier info
    cached_inlier_ratios_.resize(2, 0.0);
    cached_inlier_indices_.resize(2);
}

std::vector<size_t> HybridPointLineAbsolutePoseEstimator::num_data() const { return {points2D_.size(), lines2D_.size()}; }

std::vector<std::vector<size_t>> HybridPointLineAbsolutePoseEstimator::min_sample_sizes() const {
    return {
        {3, 0}, // P3P
        {2, 1}, // P2P1LL
        {1, 2}, // P1P2LL
        {0, 3}  // P3LL
    };
}

std::vector<double> HybridPointLineAbsolutePoseEstimator::solver_probabilities() const {
    std::vector<double> probs(4);
    auto sample_sizes = min_sample_sizes();

    for (int i = 0; i < 4; ++i) {
        probs[i] = static_cast<double>(combination(points2D_.size(), sample_sizes[i][0]) *
                                       combination(lines2D_.size(), sample_sizes[i][1]));
    }
    return probs;
}

unsigned long long HybridPointLineAbsolutePoseEstimator::combination(size_t n, size_t k) {
    if (k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    if (k > n - k)
        k = n - k; // Use symmetry C(n,k) = C(n, n-k)

    unsigned long long result = 1;
    for (size_t i = 0; i < k; ++i) {
        result *= (n - i);
        result /= (i + 1);
    }
    return result;
}

void HybridPointLineAbsolutePoseEstimator::random_sample(size_t n, size_t k, std::vector<size_t> *sample) const {
    if (k == 0 || n == 0) {
        sample->clear();
        return;
    }
    sample->resize(k);

    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);

    for (size_t i = 0; i < k; ++i) {
        std::uniform_int_distribution<size_t> dist(i, n - 1);
        size_t j = dist(rng_);
        std::swap(indices[i], indices[j]);
        (*sample)[i] = indices[i];
    }
}

void HybridPointLineAbsolutePoseEstimator::generate_sample(size_t solver_idx, std::vector<std::vector<size_t>> *sample) const {
    auto sample_sizes = min_sample_sizes();
    sample->resize(2);

    // Sample points
    random_sample(points2D_.size(), sample_sizes[solver_idx][0], &(*sample)[0]);

    // Sample lines
    random_sample(lines2D_.size(), sample_sizes[solver_idx][1], &(*sample)[1]);
}

void HybridPointLineAbsolutePoseEstimator::generate_models(const std::vector<std::vector<size_t>> &sample, size_t solver_idx,
                                                  std::vector<CameraPose> *models) const {
    models->clear();

    size_t num_points = sample[0].size();
    size_t num_lines = sample[1].size();

    // Prepare point data (normalized bearing vectors)
    for (size_t i = 0; i < num_points; ++i) {
        size_t idx = sample[0][i];
        xs_[i] = points2D_[idx].homogeneous();
        xs_[i].normalize();
        Xs_[i] = points3D_[idx];
    }

    // Prepare line data
    for (size_t i = 0; i < num_lines; ++i) {
        size_t idx = sample[1][i];
        const Line2D &l2d = lines2D_[idx];
        const Line3D &l3d = lines3D_[idx];

        // Line normal in normalized camera frame
        Point3D p1_h = l2d.x1.homogeneous();
        Point3D p2_h = l2d.x2.homogeneous();
        ls_[i] = p1_h.cross(p2_h).normalized();

        // 3D line: point and direction
        Cs_[i] = l3d.X1;
        Vs_[i] = (l3d.X2 - l3d.X1).normalized();
    }

    // Call appropriate solver
    int ret = 0;

    switch (solver_idx) {
    case 0: // P3P
        ret = p3p(xs_, Xs_, models);
        break;
    case 1: // P2P1LL
        ret = p2p1ll(xs_, Xs_, ls_, Cs_, Vs_, models);
        break;
    case 2: // P1P2LL
        ret = p1p2ll(xs_, Xs_, ls_, Cs_, Vs_, models);
        break;
    case 3: // P3LL
        ret = p3ll(ls_, Cs_, Vs_, models);
        break;
    }
    (void)ret;
}

double HybridPointLineAbsolutePoseEstimator::score_model(const CameraPose &pose, size_t *inlier_count) const {
    const double sq_threshold_pt = opt_.max_errors[0] * opt_.max_errors[0];
    const double sq_threshold_line = opt_.max_errors[1] * opt_.max_errors[1];
    const double weight_pt = opt_.data_type_weights.size() > 0 ? opt_.data_type_weights[0] : 1.0;
    const double weight_line = opt_.data_type_weights.size() > 1 ? opt_.data_type_weights[1] : 1.0;

    // Compute MSAC scores using PoseLib's utils
    size_t pt_inliers = 0, line_inliers = 0;
    double score_pt = compute_msac_score(pose, points2D_, points3D_, sq_threshold_pt, &pt_inliers);
    double score_line = compute_msac_score(pose, lines2D_, lines3D_, sq_threshold_line, &line_inliers);

    double score = score_pt * weight_pt + score_line * weight_line;
    *inlier_count = pt_inliers + line_inliers;

    // Get inlier masks using PoseLib's utils
    std::vector<char> pt_mask, line_mask;
    get_inliers(pose, points2D_, points3D_, sq_threshold_pt, &pt_mask);
    get_inliers(pose, lines2D_, lines3D_, sq_threshold_line, &line_mask);

    // Convert masks to indices
    cached_inlier_indices_[0].clear();
    cached_inlier_indices_[1].clear();
    for (size_t i = 0; i < pt_mask.size(); ++i) {
        if (pt_mask[i])
            cached_inlier_indices_[0].push_back(i);
    }
    for (size_t i = 0; i < line_mask.size(); ++i) {
        if (line_mask[i])
            cached_inlier_indices_[1].push_back(i);
    }

    // Update cached inlier ratios
    cached_inlier_ratios_[0] =
        points2D_.size() > 0 ? static_cast<double>(pt_inliers) / static_cast<double>(points2D_.size()) : 0.0;
    cached_inlier_ratios_[1] =
        lines2D_.size() > 0 ? static_cast<double>(line_inliers) / static_cast<double>(lines2D_.size()) : 0.0;

    return score;
}

std::vector<double> HybridPointLineAbsolutePoseEstimator::inlier_ratios() const { return cached_inlier_ratios_; }

std::vector<std::vector<size_t>> HybridPointLineAbsolutePoseEstimator::inlier_indices() const { return cached_inlier_indices_; }

void HybridPointLineAbsolutePoseEstimator::refine_model(CameraPose *pose) const {
    // Collect inlier data
    std::vector<Point2D> inlier_points2D;
    std::vector<Point3D> inlier_points3D;
    std::vector<Line2D> inlier_lines2D;
    std::vector<Line3D> inlier_lines3D;

    for (size_t idx : cached_inlier_indices_[0]) {
        inlier_points2D.push_back(points2D_[idx]);
        inlier_points3D.push_back(points3D_[idx]);
    }

    for (size_t idx : cached_inlier_indices_[1]) {
        inlier_lines2D.push_back(lines2D_[idx]);
        inlier_lines3D.push_back(lines3D_[idx]);
    }

    if (inlier_points2D.empty() && inlier_lines2D.empty())
        return;

    // Use PoseLib's built-in bundle_adjust for points + lines
    BundleOptions bundle_opt;
    bundle_opt.max_iterations = 25;

    bundle_adjust(inlier_points2D, inlier_points3D, inlier_lines2D, inlier_lines3D, pose, bundle_opt);
}

} // namespace poselib
