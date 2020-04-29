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

#include "essential.h"
#include <array>

namespace pose_lib {

void essential_from_motion(const CameraPose &pose, Eigen::Matrix3d *E) {
    *E << 0.0, -pose.t(2), pose.t(1),
        pose.t(2), 0.0, -pose.t(0),
        -pose.t(1), pose.t(0), 0.0;
    *E = (*E) * pose.R;
}

void motion_from_essential_fast(const Eigen::Matrix3d &E, pose_lib::CameraPoseVector *relative_poses) {

    // TODO: This can be done more robustly. This will fail if E.col(0) and E.col(1) are parallel
    Eigen::Vector3d t;
    t = E.col(0).cross(E.col(1)).normalized();

    Eigen::Matrix3d A, B;
    A << 0.0, -t(2), t(1),
        t(2), 0.0, -t(0),
        -t(1), t(0), 0.0;

    B = E;

    A = A * B.norm() / A.norm();

    // TODO: This can be done more robustly as well.
    A.row(2) = A.row(0).cross(A.row(1));
    B.row(2) = B.row(0).cross(B.row(1));

    // TODO: This can also be done with only one inverse....

    Eigen::Matrix3d R;

    CameraPose pose;
    pose.R = A.inverse() * B;
    ;
    pose.t = t;
    relative_poses->push_back(pose);
    pose.t = -t;
    relative_poses->push_back(pose);

    A.row(0) = -A.row(0);
    A.row(1) = -A.row(1);
    pose.R = A.inverse() * B;
    ;
    relative_poses->push_back(pose);
    pose.t = t;
    relative_poses->push_back(pose);
}

void motion_from_essential_planar(double e01, double e21, double e10, double e12, pose_lib::CameraPoseVector *relative_poses) {

    Eigen::Vector2d z;
    z << -e01 * e10 - e21 * e12, -e21 * e10 + e01 * e12;
    z.normalize();

    CameraPose pose;
    pose.R << z(0), 0.0, -z(1), 0.0, 1.0, 0.0, z(1), 0.0, z(0);
    pose.t << e21, 0.0, -e01;
    pose.t.normalize();

    relative_poses->push_back(pose);
    pose.t = -pose.t;
    relative_poses->push_back(pose);

    // There are two more flipped solutions where
    //    R = [a 0 b; 0 -1 0; b 0 -a]
    // These are probably not interesting in the planar case

    /*
            z << e01 * e10 - e21 * e12, e21* e10 + e01 * e12;
            z.normalize();
            pose.R << z(0), 0.0, z(1), 0.0, -1.0, 0.0, z(1), 0.0, -z(0);        
            relative_poses->push_back(pose);
            pose.t = -pose.t;
            relative_poses->push_back(pose);
    */
}

void motion_from_essential(const Eigen::Matrix3d &E, pose_lib::CameraPoseVector *relative_poses) {
    Eigen::JacobiSVD<Eigen::Matrix3d> USV(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = USV.matrixU();
    Eigen::Matrix3d Vt = USV.matrixV().transpose();

    // Last column of U is undetermined since d = (a a 0).
    if (U.determinant() < 0) {
        U.col(2) *= -1;
    }
    // Last row of Vt is undetermined since d = (a a 0).
    if (Vt.determinant() < 0) {
        Vt.row(2) *= -1;
    }

    Eigen::Matrix3d W;
    W << 0, -1, 0,
        1, 0, 0,
        0, 0, 1;

    const Eigen::Matrix3d U_W_Vt = U * W * Vt;
    const Eigen::Matrix3d U_Wt_Vt = U * W.transpose() * Vt;

    const std::array<Eigen::Matrix3d, 2> R{{U_W_Vt, U_Wt_Vt}};
    const std::array<Eigen::Vector3d, 2> t{{U.col(2), -U.col(2)}};
    if (relative_poses) {
        relative_poses->reserve(4);
        pose_lib::CameraPose pose;
        pose.R = R[0];
        pose.t = t[0];
        relative_poses->emplace_back(pose);

        pose.R = R[1];
        pose.t = t[1];
        relative_poses->emplace_back(pose);

        pose.R = R[0];
        pose.t = t[1];
        relative_poses->emplace_back(pose);

        pose.R = R[1];
        pose.t = t[0];
        relative_poses->emplace_back(pose);
    }
}

} // namespace pose_lib