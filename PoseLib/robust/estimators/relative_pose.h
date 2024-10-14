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
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"

namespace poselib {

// Estimator for calibrated relative pose (essential matrix)
// Uses Sampson error for scoring and refinement
class RelativePoseEstimator {
  public:
    RelativePoseEstimator(const RelativePoseOptions &opt, const std::vector<Point2D> &points2D_1,
                          const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.ransac.seed, opt.ransac.progressive_sampling,
                  opt.ransac.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RelativePoseOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

// Estimator for relative pose (essential matrix) with given cameras
// Uses Tangent Sampson error for scoring and refinement
class CameraRelativePoseEstimator {
  public:
    CameraRelativePoseEstimator(const RelativePoseOptions &opt, const std::vector<Point2D> &points2D_1,
                                const std::vector<Point2D> &points2D_2, const Camera &camera1, const Camera &camera2)
        : num_data(points2D_1.size()), opt(opt), x1(points2D_1), x2(points2D_2), camera1(camera1), camera2(camera2),
          sampler(num_data, sample_sz, opt.ransac.seed, opt.ransac.progressive_sampling,
                  opt.ransac.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
        camera1.unproject_with_jac(x1, &d1, &M1);
        camera2.unproject_with_jac(x2, &d2, &M2);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RelativePoseOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;
    const Camera &camera1;
    const Camera &camera2;
    std::vector<Point3D> d1, d2;
    std::vector<Eigen::Matrix<double, 3, 2>> M1, M2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class SharedFocalRelativePoseEstimator {
  public:
    SharedFocalRelativePoseEstimator(const RelativePoseOptions &opt, const std::vector<Point2D> &points2D_1,
                                     const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.ransac.seed, opt.ransac.progressive_sampling,
                  opt.ransac.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(ImagePairVector *models);
    double score_model(const ImagePair &image_pair, size_t *inlier_count) const;
    void refine_model(ImagePair *image_pair) const;

    const size_t sample_sz = 6;
    const size_t num_data;

  private:
    const RelativePoseOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class GeneralizedRelativePoseEstimator {
  public:
    GeneralizedRelativePoseEstimator(const RelativePoseOptions &opt,
                                     const std::vector<PairwiseMatches> &pairwise_matches,
                                     const std::vector<CameraPose> &camera1_ext,
                                     const std::vector<CameraPose> &camera2_ext)
        : opt(opt), matches(pairwise_matches), rig1_poses(camera1_ext), rig2_poses(camera2_ext) {
        rng = opt.ransac.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        p1s.resize(sample_sz);
        p2s.resize(sample_sz);
        sample.resize(sample_sz);

        num_data = 0;
        for (const PairwiseMatches &m : matches) {
            num_data += m.x1.size();
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 6;
    size_t num_data;

  private:
    const RelativePoseOptions &opt;
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &rig1_poses;
    const std::vector<CameraPose> &rig2_poses;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s, p1s, p2s;
    std::vector<size_t> sample;
};

class FundamentalEstimator {
  public:
    FundamentalEstimator(const RelativePoseOptions &opt, const std::vector<Point2D> &points2D_1,
                         const std::vector<Point2D> &points2D_2)
        : num_data(points2D_1.size()), opt(opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.ransac.seed, opt.ransac.progressive_sampling,
                  opt.ransac.max_prosac_iterations) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<Eigen::Matrix3d> *models);
    double score_model(const Eigen::Matrix3d &F, size_t *inlier_count) const;
    void refine_model(Eigen::Matrix3d *F) const;

    const size_t sample_sz = 7;
    const size_t num_data;

  private:
    const RelativePoseOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

class RDFundamentalEstimator {
  public:
    RDFundamentalEstimator(const RelativePoseOptions &opt, const std::vector<Point2D> &points2D_1,
                           const std::vector<Point2D> &points2D_2, const std::vector<double> &ks, const double min_k,
                           const double max_k)
        : sample_sz(ks.empty() ? 10 : 7), num_data(points2D_1.size()), opt(opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.ransac.seed, opt.ransac.progressive_sampling,
                  opt.ransac.max_prosac_iterations),
          min_k(min_k), max_k(max_k) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        x1u.resize(x1.size());
        x2u.resize(x1.size());
        sample.resize(sample_sz);
        rd_vals = ks;
    }

    void generate_models(std::vector<ProjectiveImagePair> *models);
    double score_model(const ProjectiveImagePair &F_cam_pair, size_t *inlier_count);
    void refine_model(ProjectiveImagePair *F_cam_pair);

    const size_t sample_sz;
    const size_t num_data;

  private:
    const RelativePoseOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<Eigen::Vector2d> x1u, x2u;
    std::vector<size_t> sample;
    std::vector<double> rd_vals;
    const double min_k;
    const double max_k;
};

class SharedRDFundamentalEstimator {
  public:
    SharedRDFundamentalEstimator(const RelativePoseOptions &opt, const std::vector<Point2D> &points2D_1,
                                 const std::vector<Point2D> &points2D_2, const std::vector<double> &ks,
                                 const double min_k, const double max_k)
        : sample_sz(ks.empty() ? 9 : 7), num_data(points2D_1.size()), opt(opt), x1(points2D_1), x2(points2D_2),
          sampler(num_data, sample_sz, opt.ransac.seed, opt.ransac.progressive_sampling,
                  opt.ransac.max_prosac_iterations),
          min_k(min_k), max_k(max_k) {
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        x1u.resize(x1.size());
        x2u.resize(x1.size());
        sample.resize(sample_sz);
        rd_vals = ks;
    }

    void generate_models(std::vector<ProjectiveImagePair> *models);
    double score_model(const ProjectiveImagePair &F_cam_pair, size_t *inlier_count);
    void refine_model(ProjectiveImagePair *F_cam_pair);

    const size_t sample_sz;
    const size_t num_data;

  private:
    const RelativePoseOptions &opt;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<Eigen::Vector2d> x1u, x2u;
    std::vector<size_t> sample;
    std::vector<double> rd_vals;
    const double min_k;
    const double max_k;
};

} // namespace poselib

#endif