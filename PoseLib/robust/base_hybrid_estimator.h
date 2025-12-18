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
//
// Base class for hybrid RANSAC estimators with multiple data types and solvers.

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

namespace poselib {

// Base class for hybrid RANSAC estimators.
// Estimators supporting multiple data types (e.g., points + lines) and
// multiple minimal solvers should inherit from this class.
template <typename ModelType>
class BaseHybridRansacEstimator {
public:
    using model_type = ModelType;

    virtual ~BaseHybridRansacEstimator() = default;

    // Number of different data types (e.g., 2 for points + lines)
    virtual size_t num_data_types() const = 0;

    // Number of data points per type
    virtual std::vector<size_t> num_data() const = 0;

    // Number of minimal solvers available
    virtual size_t num_minimal_solvers() const = 0;

    // Minimum sample sizes: [solver_idx][type_idx] -> sample size
    virtual std::vector<std::vector<size_t>> min_sample_sizes() const = 0;

    // Prior probabilities for solver selection (based on data availability)
    virtual std::vector<double> solver_probabilities() const = 0;

    // Generate a random sample for the given solver
    // sample: [type_idx] -> vector of data indices
    virtual void generate_sample(
        size_t solver_idx,
        std::vector<std::vector<size_t>>* sample) const = 0;

    // Generate models from a sample
    virtual void generate_models(
        const std::vector<std::vector<size_t>>& sample,
        size_t solver_idx,
        std::vector<ModelType>* models) const = 0;

    // Score a model and return MSAC score
    // Updates internal inlier cache (accessible via inlier_ratios/inlier_indices)
    virtual double score_model(const ModelType& model,
                               size_t* inlier_count) const = 0;

    // Get per-type inlier ratios from the last score_model call
    virtual std::vector<double> inlier_ratios() const = 0;

    // Get per-type inlier indices from the last score_model call
    virtual std::vector<std::vector<size_t>> inlier_indices() const = 0;

    // Refine a model (e.g., using bundle adjustment)
    virtual void refine_model(ModelType* model) const = 0;

protected:
    BaseHybridRansacEstimator() = default;
};

// Type trait to check if a type is a hybrid RANSAC estimator
template <typename T, typename = void>
struct is_hybrid_ransac_estimator : std::false_type {};

template <typename T>
struct is_hybrid_ransac_estimator<
    T, std::void_t<typename T::model_type,
                   decltype(std::declval<T>().num_data_types()),
                   decltype(std::declval<T>().num_minimal_solvers())>>
    : std::is_base_of<BaseHybridRansacEstimator<typename T::model_type>, T> {};

template <typename T>
inline constexpr bool is_hybrid_ransac_estimator_v =
    is_hybrid_ransac_estimator<T>::value;

}  // namespace poselib
