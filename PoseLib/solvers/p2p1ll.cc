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

#include "p2p1ll.h"

#include "PoseLib/misc/re3q3.h"

namespace poselib {

int p2p1ll(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
           const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output) {

    // By some calculation we get that
    //   x2 ~ [(l'*x1)*kron(Xp2'-Xp1',I_3) - x1 * kron(X-Xp1,l')] * R(:)
    // From this we can extract two constraints on the rotation + the constraint l'*R*V = 0

    Eigen::Vector3d dX21 = Xp[1] - Xp[0];
    Eigen::Vector3d dX01 = X[0] - Xp[0];
    double lxp1 = l[0].dot(xp[0]);

    dX21 *= lxp1;

    Eigen::Matrix<double, 3, 9> B;

    Eigen::Matrix<double, 1, 9> b;
    b << -dX01(0) * l[0].transpose(), -dX01(1) * l[0].transpose(), -dX01(2) * l[0].transpose();
    B.row(0) = xp[0](0) * b;
    B.row(1) = xp[0](1) * b;
    B.row(2) = xp[0](2) * b;
    B(0, 0) += dX21(0);
    B(1, 1) += dX21(0);
    B(2, 2) += dX21(0);
    B(0, 3) += dX21(1);
    B(1, 4) += dX21(1);
    B(2, 5) += dX21(1);
    B(0, 6) += dX21(2);
    B(1, 7) += dX21(2);
    B(2, 8) += dX21(2);

    B.row(0) = xp[1](2) * B.row(0) - xp[1](0) * B.row(2);
    B.row(1) = xp[1](2) * B.row(1) - xp[1](1) * B.row(2);
    B.row(2) << V[0](0) * l[0].transpose(), V[0](1) * l[0].transpose(), V[0](2) * l[0].transpose();

    Eigen::Matrix<double, 4, 8> solutions;
    int n_sols = re3q3::re3q3_rotation(B, &solutions);

    output->clear();
    for (int i = 0; i < n_sols; ++i) {
        CameraPose pose;
        pose.q = solutions.col(i);

        Eigen::Matrix3d R = pose.R();
        double lambda = -l[0].dot(R * (X[0] - Xp[0])) / lxp1;

        pose.t = lambda * xp[0] - R * Xp[0];
        output->push_back(pose);
    }

    return n_sols;
}

} // namespace poselib
