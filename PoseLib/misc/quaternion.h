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
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#ifndef POSELIB_QUATERNION_H_
#define POSELIB_QUATERNION_H_

#include <Eigen/Dense>

namespace pose_lib {

inline Eigen::Matrix3d quat_to_rotmat(const Eigen::Vector4d &q) {
    return Eigen::Quaterniond(q(0),q(1),q(2),q(3)).toRotationMatrix();
}
inline Eigen::Matrix<double,9,1> quat_to_rotmatvec(const Eigen::Vector4d &q) {
    Eigen::Matrix3d R = quat_to_rotmat(q);
    return Eigen::Map<Eigen::Matrix<double,9,1>>(R.data());
}

inline Eigen::Vector4d rotmat_to_quat(const Eigen::Matrix3d &R) {
    Eigen::Quaterniond q_flip(R);
    Eigen::Vector4d q;
    q << q_flip.w(), q_flip.x(), q_flip.y(), q_flip.z();
    return q;
}
inline Eigen::Vector4d quat_multiply(const Eigen::Vector4d &qa, const Eigen::Vector4d &qb) {
    const double qa1 = qa(0),qa2 = qa(1),qa3 = qa(2),qa4 = qa(3);
    const double qb1 = qb(0), qb2 = qb(1), qb3 = qb(2), qb4 = qb(3);

    Eigen::Vector4d q(qa1*qb1 - qa2*qb2 - qa3*qb3 - qa4*qb4,
                      qa1*qb2 + qa2*qb1 + qa3*qb4 - qa4*qb3,
                      qa1*qb3 + qa3*qb1 - qa2*qb4 + qa4*qb2,
                      qa1*qb4 + qa2*qb3 - qa3*qb2 + qa4*qb1);
    return q;
}

inline Eigen::Vector3d quat_rotate(const Eigen::Vector4d &q, const Eigen::Vector3d &p) {
    const double q1 = q(0), q2 = q(1), q3 = q(2), q4 = q(3);
    const double p1 = p(0), p2 = p(1), p3 = p(2);
    const double px1 = - p1*q2 - p2*q3 - p3*q4;
    const double px2 = p1*q1 - p2*q4 + p3*q3;
    const double px3 = p2*q1 + p1*q4 - p3*q2;
    const double px4 = p2*q2 - p1*q3 + p3*q1;
    return Eigen::Vector3d(
        px2*q1 - px1*q2 - px3*q4 + px4*q3,
        px3*q1 - px1*q3 + px2*q4 - px4*q2,
        px3*q2 - px2*q3 - px1*q4 + px4*q1
    );
}
inline Eigen::Vector4d quat_conj(const Eigen::Vector4d &q) {
    return Eigen::Vector4d(q(0), -q(1), -q(2), -q(3));
}

inline Eigen::Vector4d quat_exp(const Eigen::Vector3d &w) {
    const double nw = w.norm();
    const double theta = nw/2;

    // TODO if nw < eps
    Eigen::Vector4d q;
    q << std::cos(theta), std::sin(theta) * w / nw;
    return q;
}

inline Eigen::Vector4d quat_step_pre(const Eigen::Vector4d &q, const Eigen::Vector3d &w_delta) {
    return quat_multiply(quat_exp(w_delta), q);
}
inline Eigen::Vector4d quat_step_post(const Eigen::Vector4d &q, const Eigen::Vector3d &w_delta) {
    return quat_multiply(q, quat_exp(w_delta));
}

}


#endif