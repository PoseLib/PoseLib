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

#ifndef POSELIB_ROBUST_ESTIMATORS_BASE_ESTIMATOR_H
#define POSELIB_ROBUST_ESTIMATORS_BASE_ESTIMATOR_H

#include <type_traits>
#include <vector>

namespace poselib {

// Base class for RANSAC estimators.
// All estimators used with the ransac() function must inherit from this class
// and implement the required virtual methods.
// The ModelType template parameter should match the model type used by the estimator.
template <typename ModelType> class BaseRansacEstimator {
  public:
    using model_type = ModelType;

    virtual ~BaseRansacEstimator() = default;

    // Required interface methods
    virtual void generate_models(std::vector<ModelType> *models) = 0;
    virtual double score_model(const ModelType &model, size_t *inlier_count) const = 0;
    virtual void refine_model(ModelType *model) const = 0;

    // Required property getters
    virtual size_t sample_sz() const = 0;
    virtual size_t num_data() const = 0;

  protected:
    BaseRansacEstimator() = default;
};

// Type trait to check if T inherits from BaseRansacEstimator<SomeType>
template <typename T, typename = void> struct is_ransac_estimator : std::false_type {};

template <typename T>
struct is_ransac_estimator<T, std::void_t<typename T::model_type>>
    : std::is_base_of<BaseRansacEstimator<typename T::model_type>, T> {};

template <typename T> inline constexpr bool is_ransac_estimator_v = is_ransac_estimator<T>::value;

} // namespace poselib

#endif
