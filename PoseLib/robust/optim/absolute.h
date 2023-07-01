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

#ifndef POSELIB_ABSOLUTE_H_
#define POSELIB_ABSOLUTE_H_

#include "../../types.h"
#include "refiner_base.h"

namespace poselib {

template<typename Accumulator, typename ResidualWeightVector = UniformWeightVector>
class AbsolutePoseRefiner : public RefinerBase<Accumulator> {
public:
    AbsolutePoseRefiner(const std::vector<Point2D> &points2D,
                        const std::vector<Point3D> &points3D,
                        const Camera &cam, const ResidualWeightVector &w = ResidualWeightVector())
        : x(points2D), X(points3D), camera(cam), weights(w) {}

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        for(int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = pose.apply(X[i]);
            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;
            Eigen::Vector2d xp;
            camera.project(Z, &xp);
            const Eigen::Vector2d res = xp - x[i];
            acc.add_residual(res, weights[i]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {
        Eigen::Matrix3d R = pose.R();
        Eigen::Matrix<double,2,3> Jproj;
        Eigen::Matrix<double,2,6> J;
        Eigen::Matrix<double,2,1> res;
        for(int i = 0; i < x.size(); ++i) {
            const Eigen::Vector3d Z = R * X[i] + pose.t;

            // Note this assumes points that are behind the camera will stay behind the camera
            // during the optimization
            if (Z(2) < 0)
                continue;

            // Project with intrinsics
            Eigen::Vector2d zp;
            camera.project_with_jac(Z, &zp, &Jproj);

            // Compute reprojection error
            Eigen::Vector2d res = zp - x[i];
           
            // Jacobian w.r.t. rotation w
            Eigen::Matrix<double, 2, 3> dZ = Jproj * R;
            J.col(0) = -X[i](2) * dZ.col(1) + X[i](1) * dZ.col(2);
            J.col(1) = X[i](2) * dZ.col(0) - X[i](0) * dZ.col(2);
            J.col(2) = -X[i](1) * dZ.col(0) + X[i](0) * dZ.col(1);
            // Jacobian w.r.t. translation t
            J.block<2,3>(0,3) = dZ;

            acc.add_jacobian(res, J, weights[i]);
        }
    }

    CameraPose step(const Eigen::VectorXd &dp, const CameraPose &pose) const {
        CameraPose pose_new;
        // The rotation is parameterized via the lie-rep. and post-multiplication
        //   i.e. R(delta) = R * expm([delta]_x)
        // The pose is updated as
        //     R * dR * (X + dt) + t
        pose_new.q = quat_step_post(pose.q, dp.block<3, 1>(0, 0));
        pose_new.t = pose.t + pose.rotate(dp.block<3, 1>(3, 0));
        return pose_new;
    }

    typedef CameraPose param_t;
    static constexpr size_t num_params = 6;    
    const std::vector<Point2D> &x;
    const std::vector<Point3D> &X;
    const Camera &camera;
    const ResidualWeightVector &weights;
};


}

#endif