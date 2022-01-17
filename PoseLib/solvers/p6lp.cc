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

#include "p6lp.h"

#include "PoseLib/misc/re3q3.h"

namespace poselib {

int p6lp(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
         std::vector<CameraPose> *output) {

    Eigen::Matrix3d A1, A2;
    Eigen::Matrix<double, 3, 9> B1, B2;

    A1 << l[0].transpose(), l[1].transpose(), l[2].transpose();
    B1 << X[0](0) * l[0].transpose(), X[0](1) * l[0].transpose(), X[0](2) * l[0].transpose(),
        X[1](0) * l[1].transpose(), X[1](1) * l[1].transpose(), X[1](2) * l[1].transpose(), X[2](0) * l[2].transpose(),
        X[2](1) * l[2].transpose(), X[2](2) * l[2].transpose();

    A2 << l[3].transpose(), l[4].transpose(), l[5].transpose();
    B2 << X[3](0) * l[3].transpose(), X[3](1) * l[3].transpose(), X[3](2) * l[3].transpose(),
        X[4](0) * l[4].transpose(), X[4](1) * l[4].transpose(), X[4](2) * l[4].transpose(), X[5](0) * l[5].transpose(),
        X[5](1) * l[5].transpose(), X[5](2) * l[5].transpose();

    // t + B1*R(:) = 0
    B1 = A1.inverse() * B1;

    // A2*t + B2*R(:) = 0  ==>   (B2 - A2*B1) * R(:) = 0
    B2 -= A2 * B1;

    Eigen::Matrix<double, 4, 8> solutions;
    int n_sols = re3q3::re3q3_rotation(B2, &solutions);

    output->clear();
    for (int i = 0; i < n_sols; ++i) {
        CameraPose pose;
        pose.q = solutions.col(i);
        pose.t = -B1 * quat_to_rotmatvec(pose.q);
        output->push_back(pose);
    }

    return n_sols;
}

} // namespace poselib
