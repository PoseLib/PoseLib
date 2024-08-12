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

#ifndef POSELIB_GEN_RELATIVE_H_
#define POSELIB_GEN_RELATIVE_H_

#include "../../types.h"
#include "optim_utils.h"
#include "refiner_base.h"
#include "relative.h"

namespace poselib {

inline void deriv_essential_wrt_pose(const Eigen::Matrix3d &R1, const Eigen::Vector3d &t1, const Eigen::Matrix3d &R2,
                                     const Eigen::Vector3d &t2, const Eigen::Matrix3d &R, const Eigen::Vector3d &t,
                                     Eigen::Matrix<double, 9, 3> &dR, Eigen::Matrix<double, 9, 3> &dt) {
    Eigen::Matrix3d R2R = R2 * R;
    Eigen::Vector3d Rt = R.transpose() * t;

    // The messy expressions below compute
    // dRdw = [vec(S1) vec(S2) vec(S3)];
    // dR = (kron(R1,skew(t2)*R2R+ R2*skew(t)*R) + kron(skew(t1)*R1,R2*R)) * dRdw
    // dt = [vec(R2*R*S1*R1.') vec(R2*R*S2*R1.') vec(R2*R*S3*R1.')]

    dR(0, 0) = R2R(0, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) - R2R(0, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
               R1(0, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
               R1(0, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
    dR(0, 1) = R2R(0, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) - R2R(0, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
               R1(0, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
               R1(0, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
    dR(0, 2) = R2R(0, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) - R2R(0, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
               R1(0, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
               R1(0, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
    dR(1, 0) = R2R(1, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) - R2R(1, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
               R1(0, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
               R1(0, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
    dR(1, 1) = R2R(1, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) - R2R(1, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
               R1(0, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
               R1(0, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
    dR(1, 2) = R2R(1, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) - R2R(1, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
               R1(0, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
               R1(0, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
    dR(2, 0) = R2R(2, 1) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) - R2R(2, 2) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) +
               R1(0, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
               R1(0, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
    dR(2, 1) = R2R(2, 2) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) - R2R(2, 0) * (R1(1, 2) * t1(2) - R1(2, 2) * t1(1)) -
               R1(0, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
               R1(0, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
    dR(2, 2) = R2R(2, 0) * (R1(1, 1) * t1(2) - R1(2, 1) * t1(1)) - R2R(2, 1) * (R1(1, 0) * t1(2) - R1(2, 0) * t1(1)) -
               R1(0, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
               R1(0, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
    dR(3, 0) = R2R(0, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) - R2R(0, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
               R1(1, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
               R1(1, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
    dR(3, 1) = R2R(0, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) - R2R(0, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
               R1(1, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
               R1(1, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
    dR(3, 2) = R2R(0, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) - R2R(0, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
               R1(1, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
               R1(1, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
    dR(4, 0) = R2R(1, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) - R2R(1, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
               R1(1, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
               R1(1, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
    dR(4, 1) = R2R(1, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) - R2R(1, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
               R1(1, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
               R1(1, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
    dR(4, 2) = R2R(1, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) - R2R(1, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
               R1(1, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
               R1(1, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
    dR(5, 0) = R2R(2, 2) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) - R2R(2, 1) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) +
               R1(1, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
               R1(1, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
    dR(5, 1) = R2R(2, 0) * (R1(0, 2) * t1(2) - R1(2, 2) * t1(0)) - R2R(2, 2) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) -
               R1(1, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
               R1(1, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
    dR(5, 2) = R2R(2, 1) * (R1(0, 0) * t1(2) - R1(2, 0) * t1(0)) - R2R(2, 0) * (R1(0, 1) * t1(2) - R1(2, 1) * t1(0)) -
               R1(1, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
               R1(1, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
    dR(6, 0) = R2R(0, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) - R2R(0, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
               R1(2, 1) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
               R1(2, 2) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1));
    dR(6, 1) = R2R(0, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) - R2R(0, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
               R1(2, 0) * (R2R(0, 0) * Rt(1) - R2R(0, 1) * Rt(0) - R2R(1, 2) * t2(2) + R2R(2, 2) * t2(1)) +
               R1(2, 2) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
    dR(6, 2) = R2R(0, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) - R2R(0, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
               R1(2, 0) * (R2R(0, 0) * Rt(2) - R2R(0, 2) * Rt(0) + R2R(1, 1) * t2(2) - R2R(2, 1) * t2(1)) -
               R1(2, 1) * (R2R(0, 1) * Rt(2) - R2R(0, 2) * Rt(1) - R2R(1, 0) * t2(2) + R2R(2, 0) * t2(1));
    dR(7, 0) = R2R(1, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) - R2R(1, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
               R1(2, 1) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
               R1(2, 2) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0));
    dR(7, 1) = R2R(1, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) - R2R(1, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
               R1(2, 0) * (R2R(1, 0) * Rt(1) - R2R(1, 1) * Rt(0) + R2R(0, 2) * t2(2) - R2R(2, 2) * t2(0)) +
               R1(2, 2) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
    dR(7, 2) = R2R(1, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) - R2R(1, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
               R1(2, 0) * (R2R(1, 0) * Rt(2) - R2R(1, 2) * Rt(0) - R2R(0, 1) * t2(2) + R2R(2, 1) * t2(0)) -
               R1(2, 1) * (R2R(1, 1) * Rt(2) - R2R(1, 2) * Rt(1) + R2R(0, 0) * t2(2) - R2R(2, 0) * t2(0));
    dR(8, 0) = R2R(2, 1) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) - R2R(2, 2) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) +
               R1(2, 1) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
               R1(2, 2) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0));
    dR(8, 1) = R2R(2, 2) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) - R2R(2, 0) * (R1(0, 2) * t1(1) - R1(1, 2) * t1(0)) -
               R1(2, 0) * (R2R(2, 0) * Rt(1) - R2R(2, 1) * Rt(0) - R2R(0, 2) * t2(1) + R2R(1, 2) * t2(0)) +
               R1(2, 2) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
    dR(8, 2) = R2R(2, 0) * (R1(0, 1) * t1(1) - R1(1, 1) * t1(0)) - R2R(2, 1) * (R1(0, 0) * t1(1) - R1(1, 0) * t1(0)) -
               R1(2, 0) * (R2R(2, 0) * Rt(2) - R2R(2, 2) * Rt(0) + R2R(0, 1) * t2(1) - R2R(1, 1) * t2(0)) -
               R1(2, 1) * (R2R(2, 1) * Rt(2) - R2R(2, 2) * Rt(1) - R2R(0, 0) * t2(1) + R2R(1, 0) * t2(0));
    dt(0, 0) = R2R(0, 2) * R1(0, 1) - R2R(0, 1) * R1(0, 2);
    dt(0, 1) = R2R(0, 0) * R1(0, 2) - R2R(0, 2) * R1(0, 0);
    dt(0, 2) = R2R(0, 1) * R1(0, 0) - R2R(0, 0) * R1(0, 1);
    dt(1, 0) = R2R(1, 2) * R1(0, 1) - R2R(1, 1) * R1(0, 2);
    dt(1, 1) = R2R(1, 0) * R1(0, 2) - R2R(1, 2) * R1(0, 0);
    dt(1, 2) = R2R(1, 1) * R1(0, 0) - R2R(1, 0) * R1(0, 1);
    dt(2, 0) = R2R(2, 2) * R1(0, 1) - R2R(2, 1) * R1(0, 2);
    dt(2, 1) = R2R(2, 0) * R1(0, 2) - R2R(2, 2) * R1(0, 0);
    dt(2, 2) = R2R(2, 1) * R1(0, 0) - R2R(2, 0) * R1(0, 1);
    dt(3, 0) = R2R(0, 2) * R1(1, 1) - R2R(0, 1) * R1(1, 2);
    dt(3, 1) = R2R(0, 0) * R1(1, 2) - R2R(0, 2) * R1(1, 0);
    dt(3, 2) = R2R(0, 1) * R1(1, 0) - R2R(0, 0) * R1(1, 1);
    dt(4, 0) = R2R(1, 2) * R1(1, 1) - R2R(1, 1) * R1(1, 2);
    dt(4, 1) = R2R(1, 0) * R1(1, 2) - R2R(1, 2) * R1(1, 0);
    dt(4, 2) = R2R(1, 1) * R1(1, 0) - R2R(1, 0) * R1(1, 1);
    dt(5, 0) = R2R(2, 2) * R1(1, 1) - R2R(2, 1) * R1(1, 2);
    dt(5, 1) = R2R(2, 0) * R1(1, 2) - R2R(2, 2) * R1(1, 0);
    dt(5, 2) = R2R(2, 1) * R1(1, 0) - R2R(2, 0) * R1(1, 1);
    dt(6, 0) = R2R(0, 2) * R1(2, 1) - R2R(0, 1) * R1(2, 2);
    dt(6, 1) = R2R(0, 0) * R1(2, 2) - R2R(0, 2) * R1(2, 0);
    dt(6, 2) = R2R(0, 1) * R1(2, 0) - R2R(0, 0) * R1(2, 1);
    dt(7, 0) = R2R(1, 2) * R1(2, 1) - R2R(1, 1) * R1(2, 2);
    dt(7, 1) = R2R(1, 0) * R1(2, 2) - R2R(1, 2) * R1(2, 0);
    dt(7, 2) = R2R(1, 1) * R1(2, 0) - R2R(1, 0) * R1(2, 1);
    dt(8, 0) = R2R(2, 2) * R1(2, 1) - R2R(2, 1) * R1(2, 2);
    dt(8, 1) = R2R(2, 0) * R1(2, 2) - R2R(2, 2) * R1(2, 0);
    dt(8, 2) = R2R(2, 1) * R1(2, 0) - R2R(2, 0) * R1(2, 1);
}

template <typename Accumulator, typename ResidualWeightVectors = UniformWeightVectors>
class GeneralizedPinholeRelativePoseRefiner : public RefinerBase<Accumulator> {
  public:
    GeneralizedPinholeRelativePoseRefiner(const std::vector<PairwiseMatches> &pairwise_matches,
                                          const std::vector<CameraPose> &camera1_ext,
                                          const std::vector<CameraPose> &camera2_ext,
                                          const ResidualWeightVectors &w = ResidualWeightVectors())
        : matches(pairwise_matches), cam1_ext(camera1_ext), cam2_ext(camera2_ext), weights(w) {}

    double compute_residual(Accumulator &acc, const CameraPose &pose) {
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            if (matches[match_k].x1.size() == 0) {
                continue;
            }
            const PairwiseMatches &m = matches[match_k];
            Eigen::Vector4d q1 = cam1_ext[m.cam_id1].q;
            Eigen::Vector3d t1 = cam1_ext[m.cam_id1].t;

            Eigen::Vector4d q2 = cam2_ext[m.cam_id2].q;
            Eigen::Vector3d t2 = cam2_ext[m.cam_id2].t;

            CameraPose relpose;
            // R0 = R2 * R * R1'
            relpose.q = quat_multiply(q2, quat_multiply(pose.q, quat_conj(q1)));
            // t0 = t2 + R2 * t - R2 * R * R1' * t1
            relpose.t = t2 + quat_rotate(q2, pose.t) - relpose.rotate(t1);

            Eigen::Matrix3d E;
            essential_from_motion(relpose, &E);

            for (size_t k = 0; k < m.x1.size(); ++k) {
                double C = m.x2[k].homogeneous().dot(E * m.x1[k].homogeneous());

                // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
                Eigen::Vector4d J_C;
                J_C << E.block<3, 2>(0, 0).transpose() * m.x2[k].homogeneous(),
                    E.block<2, 3>(0, 0) * m.x1[k].homogeneous();
                const double nJ_C = J_C.norm();
                const double inv_nJ_C = 1.0 / nJ_C;
                const double r = C * inv_nJ_C;
                acc.add_residual(r, weights[match_k][k]);
            }
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const CameraPose &pose) {
        Eigen::Matrix3d R = pose.R();
        size_t num_residuals = 0;
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            const PairwiseMatches &m = matches[match_k];

            // Cameras are
            // [R1 t1]
            // [R2 t2] * [R t; 0 1] = [R2*R t2+R2*t]

            // Relative pose is
            // [R2*R*R1' t2+R2*t-R2*R*R1'*t1]
            // Essential matrix is
            // [t2]_x*R2*R*R1' + [R2*t]_x*R2*R*R1' - R2*R*R1'*[t1]_x

            Eigen::Vector4d q1 = cam1_ext[m.cam_id1].q;
            Eigen::Matrix3d R1 = quat_to_rotmat(q1);
            Eigen::Vector3d t1 = cam1_ext[m.cam_id1].t;

            Eigen::Vector4d q2 = cam2_ext[m.cam_id2].q;
            Eigen::Matrix3d R2 = quat_to_rotmat(q2);
            Eigen::Vector3d t2 = cam2_ext[m.cam_id2].t;

            CameraPose relpose;
            relpose.q = quat_multiply(q2, quat_multiply(pose.q, quat_conj(q1)));
            relpose.t = t2 + R2 * pose.t - relpose.rotate(t1);
            Eigen::Matrix3d E;
            essential_from_motion(relpose, &E);

            // TODO: Replace with something nice
            Eigen::Matrix<double, 9, 3> dR;
            Eigen::Matrix<double, 9, 3> dt;
            deriv_essential_wrt_pose(R1, t1, R2, t2, R, pose.t, dR, dt);

            for (size_t k = 0; k < m.x1.size(); ++k) {
                double C = m.x2[k].homogeneous().dot(E * m.x1[k].homogeneous());

                // J_C is the Jacobian of the epipolar constraint w.r.t. the image points
                Eigen::Vector4d J_C;
                J_C << E.block<3, 2>(0, 0).transpose() * m.x2[k].homogeneous(),
                    E.block<2, 3>(0, 0) * m.x1[k].homogeneous();
                const double nJ_C = J_C.norm();
                const double inv_nJ_C = 1.0 / nJ_C;
                const double r = C * inv_nJ_C;

                // Compute weight from robust loss function (used in the IRLS)
                const double weight = weights[match_k][k];
                num_residuals++;

                // Compute Jacobian of Sampson error w.r.t the fundamental/essential matrix (3x3)
                Eigen::Matrix<double, 1, 9> dF;
                dF << m.x1[k](0) * m.x2[k](0), m.x1[k](0) * m.x2[k](1), m.x1[k](0), m.x1[k](1) * m.x2[k](0),
                    m.x1[k](1) * m.x2[k](1), m.x1[k](1), m.x2[k](0), m.x2[k](1), 1.0;
                const double s = C * inv_nJ_C * inv_nJ_C;
                dF(0) -= s * (J_C(2) * m.x1[k](0) + J_C(0) * m.x2[k](0));
                dF(1) -= s * (J_C(3) * m.x1[k](0) + J_C(0) * m.x2[k](1));
                dF(2) -= s * (J_C(0));
                dF(3) -= s * (J_C(2) * m.x1[k](1) + J_C(1) * m.x2[k](0));
                dF(4) -= s * (J_C(3) * m.x1[k](1) + J_C(1) * m.x2[k](1));
                dF(5) -= s * (J_C(1));
                dF(6) -= s * (J_C(2));
                dF(7) -= s * (J_C(3));
                dF *= inv_nJ_C;

                // and then w.r.t. the pose parameters
                Eigen::Matrix<double, 1, 6> J;
                J.block<1, 3>(0, 0) = dF * dR;
                J.block<1, 3>(0, 3) = dF * dt;

                acc.add_jacobian(r, J, weight);
            }
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
    const std::vector<PairwiseMatches> &matches;
    const std::vector<CameraPose> &cam1_ext;
    const std::vector<CameraPose> &cam2_ext;
    const ResidualWeightVectors &weights;
};
} // namespace poselib

#endif