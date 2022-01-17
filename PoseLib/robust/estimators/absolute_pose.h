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

#ifndef POSELIB_ROBUST_ESTIMATORS_ABSOLUTE_POSE_H
#define POSELIB_ROBUST_ESTIMATORS_ABSOLUTE_POSE_H

#include "PoseLib/camera_pose.h"
#include "PoseLib/robust/sampling.h"
#include "PoseLib/robust/utils.h"
#include "PoseLib/types.h"

namespace poselib {

class AbsolutePoseEstimator {
  public:
    AbsolutePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D,
                          const std::vector<Point3D> &points3D)
        : num_data(points2D.size()), opt(ransac_opt), x(points2D), X(points3D),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Point3D> xs, Xs;
    std::vector<size_t> sample;
};

class GeneralizedAbsolutePoseEstimator {
  public:
    GeneralizedAbsolutePoseEstimator(const RansacOptions &ransac_opt, const std::vector<std::vector<Point2D>> &points2D,
                                     const std::vector<std::vector<Point3D>> &points3D,
                                     const std::vector<CameraPose> &camera_ext)
        : num_cams(points2D.size()), opt(ransac_opt), x(points2D), X(points3D), rig_poses(camera_ext) {
        rng = opt.seed;
        ps.resize(sample_sz);
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
        camera_centers.resize(num_cams);
        for (size_t k = 0; k < num_cams; ++k) {
            camera_centers[k] = camera_ext[k].center();
        }

        num_data = 0;
        num_pts_camera.resize(num_cams);
        for (size_t k = 0; k < num_cams; ++k) {
            num_pts_camera[k] = points2D[k].size();
            num_data += num_pts_camera[k];
        }
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3;
    size_t num_data;
    const size_t num_cams;

  private:
    const RansacOptions &opt;
    const std::vector<std::vector<Point2D>> &x;
    const std::vector<std::vector<Point3D>> &X;
    const std::vector<CameraPose> &rig_poses;
    std::vector<Point3D> camera_centers;
    std::vector<size_t> num_pts_camera; // number of points in each camera

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Point3D> ps, xs, Xs;
    std::vector<std::pair<size_t, size_t>> sample;
};

class AbsolutePosePointLineEstimator {
  public:
    AbsolutePosePointLineEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &x,
                                   const std::vector<Point3D> &X, const std::vector<Line2D> &l,
                                   const std::vector<Line3D> &L)
        : num_data(x.size() + l.size()), opt(ransac_opt), points2D(x), points3D(X), lines2D(l), lines3D(L) {
        rng = opt.seed;
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        ls.resize(sample_sz);
        Cs.resize(sample_sz);
        Vs.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &points2D;
    const std::vector<Point3D> &points3D;
    const std::vector<Line2D> &lines2D;
    const std::vector<Line3D> &lines3D;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Point3D> xs, Xs, ls, Cs, Vs;
    std::vector<size_t> sample;
};

class Radial1DAbsolutePoseEstimator {
  public:
    Radial1DAbsolutePoseEstimator(const RansacOptions &ransac_opt, const std::vector<Point2D> &points2D,
                                  const std::vector<Point3D> &points3D)
        : num_data(points2D.size()), opt(ransac_opt), x(points2D), X(points3D),
          sampler(num_data, sample_sz, opt.seed, opt.progressive_sampling, opt.max_prosac_iterations) {
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models);
    double score_model(const CameraPose &pose, size_t *inlier_count) const;
    void refine_model(CameraPose *pose) const;

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;

    RandomSampler sampler;
    // pre-allocated vectors for sampling
    std::vector<Point2D> xs;
    std::vector<Point3D> Xs;
    std::vector<size_t> sample;
};

} // namespace poselib

#endif
