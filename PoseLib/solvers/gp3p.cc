// Copyright (c) 2020, Viktor Larsson
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
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gp3p.h"

#include "PoseLib/misc/re3q3.h"
#include "rig_frame.h"

namespace poselib {

namespace {
// Solves for camera pose such that: p+lambda*x = R*X+t, in the frame the data is given in.
int gp3p_fixed_frame(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                     const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output) {
    Eigen::Matrix<double, 6, 13> A;

    for (int i = 0; i < 3; ++i) {
        // xx = [x3 0 -x1; 0 x3 -x2]
        // eqs = [xx kron(X',xx), -xx*p] * [t; vec(R); 1]

        A.row(2 * i) << x[i](2), 0.0, -x[i](0), X[i](0) * x[i](2), 0.0, -X[i](0) * x[i](0), X[i](1) * x[i](2), 0.0,
            -X[i](1) * x[i](0), X[i](2) * x[i](2), 0.0, -X[i](2) * x[i](0), -p[i](0) * x[i](2) + p[i](2) * x[i](0);
        A.row(2 * i + 1) << 0.0, x[i](2), -x[i](1), 0.0, X[i](0) * x[i](2), -X[i](0) * x[i](1), 0.0, X[i](1) * x[i](2),
            -X[i](1) * x[i](1), 0.0, X[i](2) * x[i](2), -X[i](2) * x[i](1), -p[i](1) * x[i](2) + p[i](2) * x[i](1);
    }

    Eigen::Matrix3d B = A.block<3, 3>(0, 0).inverse();

    Eigen::Matrix<double, 3, 10> AR = A.block<3, 10>(3, 3) - A.block<3, 3>(3, 0) * B * A.block<3, 10>(0, 3);
    Eigen::Matrix<double, 4, 8> solutions;
    int n_sols = re3q3::re3q3_rotation(AR, &solutions);

    output->clear();
    for (int i = 0; i < n_sols; ++i) {
        CameraPose pose;
        pose.q = solutions.col(i);
        pose.t = -B * (A.block<3, 9>(0, 3) * quat_to_rotmatvec(pose.q) + A.block<3, 1>(0, 12));
        output->push_back(pose);
    }

    return n_sols;
}
} // namespace

// Solves for camera pose such that: p+lambda*x = R*X+t
int gp3p(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
         const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output) {
    // The formulation is singular for a ray with x3 = 0 and for rays 0 and 1 with equal x1/x3, which are
    // properties of the rig frame and of the order of the points only. In that case put the least parallel
    // pair of rays first and solve in a rotated frame (see rig_frame.h); the problem itself is unchanged.
    if (!rig_frame_is_ill_conditioned(x, 3, true))
        return gp3p_fixed_frame(p, x, X, output);

    int ind[3] = {0, 1, 2};
    double best = -1.0;
    for (int k = 0; k < 3; ++k) {
        const int i = k, j = (k + 1) % 3;
        const double s = x[i].normalized().cross(x[j].normalized()).squaredNorm();
        if (s > best) {
            best = s;
            ind[0] = i;
            ind[1] = j;
            ind[2] = (k + 2) % 3;
        }
    }
    std::vector<Eigen::Vector3d> pp(3), xp(3), Xp(3);
    for (int k = 0; k < 3; ++k) {
        pp[k] = p[ind[k]];
        xp[k] = x[ind[k]];
        Xp[k] = X[ind[k]];
    }
    const Eigen::Matrix3d Q = well_conditioned_rig_frame(xp, 3);
    std::vector<Eigen::Vector3d> pr, xr;
    rotate_rig(Q, pp, 3, &pr);
    rotate_rig(Q, xp, 3, &xr);
    gp3p_fixed_frame(pr, xr, Xp, output);
    derotate_poses(Q, output);
    return output->size();
}

} // namespace poselib
