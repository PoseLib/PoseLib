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

#include "hybrid_point_line_focal_absolute_pose.h"

#include "PoseLib/robust/bundle.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/solvers/p1p3llf.h"
#include "PoseLib/solvers/p2p2llf.h"
#include "PoseLib/solvers/p3p1llf.h"
#include "PoseLib/solvers/p4llf.h"
#include "PoseLib/solvers/p4pf.h"

#include <limits>
#include <numeric>
#include <stdexcept>

namespace poselib {

HybridPointLineFocalAbsolutePoseEstimator::HybridPointLineFocalAbsolutePoseEstimator(
    const HybridRansacOptions &opt, const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
    const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D)
    : opt_(opt), points2D_(points2D), points3D_(points3D), lines2D_(lines2D), lines3D_(lines3D) {
    rng_.seed(opt.seed);

    // Validate max_errors has at least 2 elements (point and line thresholds)
    if (opt_.max_errors.size() < 2) {
        throw std::invalid_argument("HybridRansacOptions::max_errors must have at least 2 elements "
                                    "(point and line error thresholds)");
    }
}

std::vector<size_t> HybridPointLineFocalAbsolutePoseEstimator::num_data() const {
    return {points2D_.size(), lines2D_.size()};
}

std::vector<std::vector<size_t>> HybridPointLineFocalAbsolutePoseEstimator::min_sample_sizes() const {
    return {
        {4, 0}, // P4Pf
        {3, 1}, // P3P1LLf
        {2, 2}, // P2P2LLf
        {1, 3}, // P1P3LLf
        {0, 4}  // P4LLf
    };
}

std::vector<double> HybridPointLineFocalAbsolutePoseEstimator::solver_probabilities() const {
    std::vector<double> probs(5);
    auto sample_sizes = min_sample_sizes();

    for (int i = 0; i < 5; ++i) {
        probs[i] = static_cast<double>(combination(points2D_.size(), sample_sizes[i][0]) *
                                       combination(lines2D_.size(), sample_sizes[i][1]));
    }
    return probs;
}

unsigned long long HybridPointLineFocalAbsolutePoseEstimator::combination(size_t n, size_t k) {
    if (k > n)
        return 0;
    if (k == 0 || k == n)
        return 1;
    if (k > n - k)
        k = n - k; // Use symmetry C(n,k) = C(n, n-k)

    double result = 1.0;
    for (size_t i = 0; i < k; ++i) {
        result *= static_cast<double>(n - i) / static_cast<double>(i + 1);
    }
    return static_cast<unsigned long long>(result + 0.5);
}

void HybridPointLineFocalAbsolutePoseEstimator::random_sample(size_t n, size_t k, std::vector<size_t> *sample) const {
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

void HybridPointLineFocalAbsolutePoseEstimator::generate_sample(size_t solver_idx,
                                                                std::vector<std::vector<size_t>> *sample) const {
    auto sample_sizes = min_sample_sizes();
    sample->resize(2);

    random_sample(points2D_.size(), sample_sizes[solver_idx][0], &(*sample)[0]);
    random_sample(lines2D_.size(), sample_sizes[solver_idx][1], &(*sample)[1]);
}

void HybridPointLineFocalAbsolutePoseEstimator::generate_models(const std::vector<std::vector<size_t>> &sample,
                                                                size_t solver_idx, std::vector<Image> *models) const {
    models->clear();

    const size_t num_points = sample[0].size();
    const size_t num_lines = sample[1].size();

    // Size the buffers exactly (the focal solvers read input sizes via .size()).
    xs_.resize(num_points);
    Xs_.resize(num_points);
    ls_.resize(num_lines);
    Cs_.resize(num_lines);
    Vs_.resize(num_lines);

    // Points: the focal solvers take 2D pixel coordinates directly.
    for (size_t i = 0; i < num_points; ++i) {
        size_t idx = sample[0][i];
        xs_[i] = points2D_[idx];
        Xs_[i] = points3D_[idx];
    }

    // Lines: image line normal (pixels) + 3D line point and direction.
    for (size_t i = 0; i < num_lines; ++i) {
        size_t idx = sample[1][i];
        const Line2D &l2d = lines2D_[idx];
        const Line3D &l3d = lines3D_[idx];

        ls_[i] = l2d.x1.homogeneous().cross(l2d.x2.homogeneous()).normalized();
        Cs_[i] = l3d.X1;
        Vs_[i] = (l3d.X2 - l3d.X1).normalized();
    }

    std::vector<CameraPose> poses;
    std::vector<double> focals;
    switch (solver_idx) {
    case 0: // P4Pf
        p4pf(xs_, Xs_, &poses, &focals);
        break;
    case 1: // P3P1LLf
        p3p1llf(xs_, Xs_, ls_, Cs_, Vs_, &poses, &focals);
        break;
    case 2: // P2P2LLf
        p2p2llf(xs_, Xs_, ls_, Cs_, Vs_, &poses, &focals);
        break;
    case 3: // P1P3LLf
        p1p3llf(xs_, Xs_, ls_, Cs_, Vs_, &poses, &focals);
        break;
    case 4: // P4LLf
        p4llf(ls_, Cs_, Vs_, &poses, &focals);
        break;
    }

    models->reserve(poses.size());
    for (size_t i = 0; i < poses.size(); ++i) {
        if (focals[i] < 0)
            continue;
        Camera camera;
        camera.model_id = CameraModelId::SIMPLE_PINHOLE;
        camera.width = 0;
        camera.height = 0;
        camera.params = {focals[i], 0.0, 0.0};
        models->emplace_back(poses[i], camera);
    }
}

double HybridPointLineFocalAbsolutePoseEstimator::score_model(const Image &image,
                                                              std::vector<size_t> *inliers_per_type) const {
    const double sq_threshold_pt = opt_.max_errors[0] * opt_.max_errors[0];
    const double sq_threshold_line = opt_.max_errors[1] * opt_.max_errors[1];
    const double weight_pt = opt_.data_type_weights.size() > 0 ? opt_.data_type_weights[0] : 1.0;
    const double weight_line = opt_.data_type_weights.size() > 1 ? opt_.data_type_weights[1] : 1.0;

    size_t pt_inliers = 0, line_inliers = 0;
    double score = std::numeric_limits<double>::max();
    if (image.camera.focal() >= 0) {
        double score_pt = compute_msac_score(image, points2D_, points3D_, sq_threshold_pt, &pt_inliers);
        double score_line = compute_msac_score(image, lines2D_, lines3D_, sq_threshold_line, &line_inliers);
        score = score_pt * weight_pt + score_line * weight_line;
    }

    if (inliers_per_type) {
        inliers_per_type->resize(2);
        (*inliers_per_type)[0] = pt_inliers;
        (*inliers_per_type)[1] = line_inliers;
    }

    return score;
}

void HybridPointLineFocalAbsolutePoseEstimator::refine_model(Image *image) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt_.max_errors[0];
    bundle_opt.max_iterations = 25;
    bundle_opt.refine_focal_length = true;
    bundle_opt.refine_principal_point = false;
    bundle_opt.refine_extra_params = false;

    BundleOptions line_bundle_opt = bundle_opt;
    line_bundle_opt.loss_scale = opt_.max_errors[1];

    const double weight_pt = opt_.data_type_weights.size() > 0 ? opt_.data_type_weights[0] : 1.0;
    const double weight_line = opt_.data_type_weights.size() > 1 ? opt_.data_type_weights[1] : 1.0;
    std::vector<double> weights_pts(points2D_.size(), weight_pt);
    std::vector<double> weights_lines(lines2D_.size(), weight_line);

    bundle_adjust(points2D_, points3D_, lines2D_, lines3D_, image, bundle_opt, line_bundle_opt, weights_pts,
                  weights_lines);
}

} // namespace poselib
