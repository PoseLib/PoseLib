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

#ifndef POSELIB_GEN_ABSOLUTE_H_
#define POSELIB_GEN_ABSOLUTE_H_

#include "../../types.h"
#include "absolute.h"
#include "optim_utils.h"

namespace poselib {

template <typename Accumulator, typename ResidualWeightVectors = UniformWeightVectors>
class GeneralizedAbsolutePoseRefiner : public RefinerBase<Accumulator> {
  public:
    GeneralizedAbsolutePoseRefiner(const std::vector<std::vector<Point2D>> &points2D,
                                   const std::vector<std::vector<Point3D>> &points3D,
                                   const std::vector<CameraPose> &camera_ext, const std::vector<Camera> &camera_int,
                                   const ResidualWeightVectors &w = ResidualWeightVectors())
        : num_cams(points2D.size()), x(points2D), X(points3D), rig_poses(camera_ext), cameras(camera_int), weights(w) {}

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        for (int k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            const Camera &camera = cameras[k];
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;
            for (int i = 0; i < x[k].size(); ++i) {
                const Eigen::Vector3d Z = full_pose.apply(X[k][i]);
                // Note this assumes points that are behind the camera will stay behind the camera
                // during the optimization
                if (Z(2) < 0)
                    continue;
                Eigen::Vector2d xp;
                camera.project(Z, &xp);
                const Eigen::Vector2d res = xp - x[k][i];
                acc.add_residual(res, weights[k][i]);
            }
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {
        for (size_t k = 0; k < num_cams; ++k) {
            if (x[k].size() == 0) {
                continue;
            }
            CameraPose full_pose;
            full_pose.q = quat_multiply(rig_poses[k].q, pose.q);
            full_pose.t = rig_poses[k].rotate(pose.t) + rig_poses[k].t;
            AbsolutePoseRefiner<Accumulator, typename ResidualWeightVectors::value_type> cam_refiner(
                x[k], X[k], cameras[k], weights[k]);
            // Rk * (R*X + t) + tk
            // Rk * (R * (X + t)) + tk
            cam_refiner.compute_jacobian(acc, full_pose);
        }
    }

    CameraPose step(const Eigen::VectorXd &dp, const CameraPose &pose) const {
        CameraPose pose_new;
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }
    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;

  private:
    const size_t num_cams;
    const std::vector<std::vector<Point2D>> &x;
    const std::vector<std::vector<Point3D>> &X;
    const std::vector<CameraPose> &rig_poses;
    const std::vector<Camera> &cameras;
    const ResidualWeightVectors &weights;
};
} // namespace poselib

#endif