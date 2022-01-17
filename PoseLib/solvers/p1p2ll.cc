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

#include "p1p2ll.h"

#include "PoseLib/misc/re3q3.h"

namespace poselib {

int p1p2ll(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
           const std::vector<Eigen::Vector3d> &V, std::vector<CameraPose> *output) {

    // We center coordinate system on Xp
    // Point-point equation then yield:  t = lambda*xp

    // Inserting into l'*(R*X + t) = l'*(R(X-Xp) + lambda*xp)

    // From the two equations we get
    //   lambda * l1'*xp + l1'*R*(X1-Xp) = 0
    //   lambda * l2'*xp + l2'*R*(X2-Xp) = 0
    // Eliminating lambda
    //   [(l1'*xp) * kron(X2-Xp,l2') - (l2'*xp) * kron(X1-Xp,l1')] * R(:) = 0

    double l1xp = l[0].dot(xp[0]);
    double l2xp = l[1].dot(xp[0]);

    Eigen::Matrix<double, 3, 9> B;

    Eigen::Vector3d z1 = l2xp * (X[0] - Xp[0]);
    Eigen::Vector3d z2 = l1xp * (X[1] - Xp[0]);

    // Two equations from l'*R*V = 0 and finally the equation from above
    B << V[0](0) * l[0].transpose(), V[0](1) * l[0].transpose(), V[0](2) * l[0].transpose(), V[1](0) * l[1].transpose(),
        V[1](1) * l[1].transpose(), V[1](2) * l[1].transpose(), z1(0) * l[0].transpose() - z2(0) * l[1].transpose(),
        z1(1) * l[0].transpose() - z2(1) * l[1].transpose(), z1(2) * l[0].transpose() - z2(2) * l[1].transpose();

    Eigen::Matrix<double, 4, 8> solutions;
    int n_sols = re3q3::re3q3_rotation(B, &solutions);

    output->clear();
    for (int i = 0; i < n_sols; ++i) {
        CameraPose pose;
        pose.q = solutions.col(i);

        Eigen::Matrix3d R = pose.R();
        double lambda = -l[0].dot(R * (X[0] - Xp[0])) / l1xp;

        pose.t = lambda * xp[0] - R * Xp[0];
        output->push_back(pose);
    }

    return n_sols;
}

} // namespace poselib
