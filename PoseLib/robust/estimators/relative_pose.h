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

#ifndef POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_RELATIVE_POSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/estimators/base_estimator.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"

namespace poselib {

class RelativePoseEstimator : public BaseRansacEstimator<CameraPose> {
  public:
    RelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                          const std::vector<Point2D> &points2D_2)
        : num_data_(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data_, sample_sz_, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz_);
        x2s.resize(sample_sz_);
        sample.resize(sample_sz_);
    }

    void generate_models(std::vector<CameraPose> *models) override;
    double score_model(const CameraPose &pose, size_t *inlier_count) const override;
    void refine_model(CameraPose *pose) const override;

    size_t sample_sz() const override { return sample_sz_; }
    size_t num_data() const override { return num_data_; }

  private:
    static constexpr size_t sample_sz_ = 5;
    const size_t num_data_;
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class RelativePoseMonoDepthEstimator : public BaseRansacEstimator<MonoDepthTwoViewGeometry> {
  public:
    RelativePoseMonoDepthEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                   const std::vector<Point2D> &points2D_2, const std::vector<double> &d1,
                                   const std::vector<double> &d2)
        : num_data_(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), d1(d1), d2(d2),
          sampler(num_data_, sample_sz_, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz_);
        x2s.resize(sample_sz_);
        d1s.resize(sample_sz_);
        d2s.resize(sample_sz_);
        X.resize(sample_sz_);
        sample.resize(sample_sz_);
        // the scale of the reprojection error to the sampson error
        scale_reproj = (opt.max_reproj_error > 0.0) ? (opt.max_epipolar_error * opt.max_epipolar_error) /
                                                          (opt.max_reproj_error * opt.max_reproj_error)
                                                    : 0.0;
    }

    void generate_models(std::vector<MonoDepthTwoViewGeometry> *models) override;
    double score_model(const MonoDepthTwoViewGeometry &model, size_t *inlier_count) const override;
    void refine_model(MonoDepthTwoViewGeometry *model) const override;

    size_t sample_sz() const override { return sample_sz_; }
    size_t num_data() const override { return num_data_; }

  private:
    static constexpr size_t sample_sz_ = 3;
    const size_t num_data_;
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1, &d2;
    RandomSampler sampler;

    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<double> d1s, d2s;
    std::vector<Point3D> X;
    std::vector<size_t> sample;
    double scale_reproj;
};

class SharedFocalRelativePoseEstimator : public BaseRansacEstimator<ImagePair> {
  public:
    SharedFocalRelativePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                     const std::vector<Point2D> &points2D_2)
        : num_data_(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data_, sample_sz_, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz_);
        x2s.resize(sample_sz_);
        sample.resize(sample_sz_);
    }

    void generate_models(ImagePairVector *models) override;
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const override;
    void refine_model(ImagePair *image_pair) const override;

    size_t sample_sz() const override { return sample_sz_; }
    size_t num_data() const override { return num_data_; }

  private:
    static constexpr size_t sample_sz_ = 6;
    const size_t num_data_;
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class SharedFocalMonodepthPoseEstimator : public BaseRansacEstimator<MonoDepthImagePair> {
  public:
    SharedFocalMonodepthPoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                      const std::vector<Point2D> &points2D_2, const std::vector<double> &d1,
                                      const std::vector<double> &d2)
        : num_data_(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), d1(d1), d2(d2),
          sampler(num_data_, sample_sz_, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz_);
        x2s.resize(sample_sz_);
        d1s.resize(sample_sz_);
        d2s.resize(sample_sz_);
        sample.resize(sample_sz_);
        x1h.resize(x1.size());
        x2h.resize(x2.size());
        for (size_t i = 0; i < x1.size(); ++i) {
            x1h[i] = x1[i].homogeneous();
            x2h[i] = x2[i].homogeneous();
        }

        scale_reproj =
            (opt.max_epipolar_error * opt.max_epipolar_error) / (opt.max_reproj_error * opt.max_reproj_error);
    }

    void generate_models(std::vector<MonoDepthImagePair> *models) override;
    double score_model(const MonoDepthImagePair &image_pair, size_t *inlier_count) const override;
    void refine_model(MonoDepthImagePair *image_pair) const override;

    size_t sample_sz() const override { return sample_sz_; }
    size_t num_data() const override { return num_data_; }

  private:
    static constexpr size_t sample_sz_ = 3;
    const size_t num_data_;
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<Eigen::Vector3d> x1h, x2h;
    std::vector<double> d1s, d2s;
    std::vector<size_t> sample;
    double scale_reproj;
};

class VaryingFocalMonodepthPoseEstimator : public BaseRansacEstimator<MonoDepthImagePair> {
  public:
    VaryingFocalMonodepthPoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                                       const std::vector<Point2D> &points2D_2, const std::vector<double> &d1,
                                       const std::vector<double> &d2)
        : num_data_(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2), d1(d1), d2(d2),
          sampler(num_data_, sample_sz_, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz_);
        x2s.resize(sample_sz_);
        d1s.resize(sample_sz_);
        d2s.resize(sample_sz_);
        sample.resize(sample_sz_);
        x1h.resize(x1.size());
        x2h.resize(x1.size());
        for (size_t i = 0; i < x1.size(); ++i) {
            x1h[i] = x1[i].homogeneous();
            x2h[i] = x2[i].homogeneous();
        }
        scale_reproj =
            (opt.max_epipolar_error * opt.max_epipolar_error) / (opt.max_reproj_error * opt.max_reproj_error);
    }

    void generate_models(std::vector<MonoDepthImagePair> *models) override;
    double score_model(const MonoDepthImagePair &image_pair, size_t *inlier_count) const override;
    void refine_model(MonoDepthImagePair *image_pair) const override;

    size_t sample_sz() const override { return sample_sz_; }
    size_t num_data() const override { return num_data_; }

  private:
    static constexpr size_t sample_sz_ = 3;
    const size_t num_data_;
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const std::vector<double> &d1;
    const std::vector<double> &d2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<Eigen::Vector3d> x1h, x2h;
    std::vector<double> d1s, d2s;
    std::vector<size_t> sample;
    double scale_reproj;
};

class GeneralizedRelativePoseEstimator : public BaseRansacEstimator<CameraPose> {
  public:
    GeneralizedRelativePoseEstimator(const RansacOptions &ransac_opt,
                                     const std::vector<PairwiseMatches> &pairwise_matches,
                                     const std::vector<CameraPose> &camera1_ext,
                                     const std::vector<CameraPose> &camera2_ext)
        : opt(ransac_opt), matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext) {
        rng = opt.seed;
        x1s.resize(sample_sz_);
        x2s.resize(sample_sz_);
        p1s.resize(sample_sz_);
        p2s.resize(sample_sz_);
        sample.resize(sample_sz_);

        num_data_ = 0;
        for (const PairwiseMatches &m : matches) {
            num_data_ += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models) override;
    double score_model(const CameraPose &pose, size_t *inlier_count) const override;
    void refine_model(CameraPose *pose) const override;

    size_t sample_sz() const override { return sample_sz_; }
    size_t num_data() const override { return num_data_; }

  private:
    static constexpr size_t sample_sz_ = 6;
    size_t num_data_;
    const RansacOptions &opt;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s, p1s, p2s;
    std::vector<size_t> sample;
};

class FundamentalEstimator : public BaseRansacEstimator<Eigen::Matrix3d> {
  public:
    FundamentalEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D_1,
                         const std::vector<Point2D> &points2D_2)
        : num_data_(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data_, sample_sz_, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        x1s.resize(sample_sz_);
        x2s.resize(sample_sz_);
        sample.resize(sample_sz_);
    }

    void generate_models(std::vector<Eigen::Matrix3d> *models) override;
    double score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const override;
    void refine_model(Eigen::Matrix3d *F) const override;

    size_t sample_sz() const override { return sample_sz_; }
    size_t num_data() const override { return num_data_; }

  private:
    static constexpr size_t sample_sz_ = 7;
    const size_t num_data_;
    const RansacOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

} // namespace poselib

#endif
