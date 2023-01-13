// Copyright (c) 2020, Pierre Moulon
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

#include "relpose_upright_planar_3pt.h"

#include "PoseLib/misc/essential.h"

int poselib::relpose_upright_planar_3pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                                        CameraPoseVector *output) {

    // Build the action matrix -> see (6,7) in the paper
    Eigen::Matrix<double, 4, 3> A;
    for (const int i : {0, 1, 2}) {
        const auto &bearing_a_i = x1[i];
        const auto &bearing_b_i = x2[i];
        A.col(i) << bearing_a_i.x() * bearing_b_i.y(), -bearing_a_i.z() * bearing_b_i.y(),
            -bearing_b_i.x() * bearing_a_i.y(), -bearing_b_i.z() * bearing_a_i.y();
    }

    const Eigen::Matrix4d Q = A.householderQr().householderQ();
    const Eigen::Vector4d nullspace = Q.col(3);

    output->clear();
    motion_from_essential_planar(nullspace(2), nullspace(3), -nullspace(0), nullspace(1), x1, x2, output);
    return output->size();
}
