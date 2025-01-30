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

#ifndef POSELIB_MISC_QUATERNION_H_
#define POSELIB_MISC_QUATERNION_H_

#include "PoseLib/real_matrix.h"

#include <Eigen/Dense>

//  We dont use Eigen::Quaternion here since we want qw,qx,qy,qz ordering
namespace poselib {

inline Matrix3x3 quat_to_rotmat(const Vector4 &q) {
    return Eigen::Quaternion(q(0), q(1), q(2), q(3)).toRotationMatrix();
}
inline Eigen::Matrix<Real, 9, 1> quat_to_rotmatvec(const Vector4 &q) {
    Matrix3x3 R = quat_to_rotmat(q);
    Eigen::Matrix<Real, 9, 1> r = Eigen::Map<Eigen::Matrix<Real, 9, 1>>(R.data());
    return r;
}

inline Vector4 rotmat_to_quat(const Matrix3x3 &R) {
    Quaternion q_flip(R);
    Vector4 q;
    q << q_flip.w(), q_flip.x(), q_flip.y(), q_flip.z();
    q.normalize();
    return q;
}
inline Vector4 quat_multiply(const Vector4 &qa, const Vector4 &qb) {
    const Real qa1 = qa(0), qa2 = qa(1), qa3 = qa(2), qa4 = qa(3);
    const Real qb1 = qb(0), qb2 = qb(1), qb3 = qb(2), qb4 = qb(3);

    return Vector4(qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4, qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
                   qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2, qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1);
}

inline Vector3 quat_rotate(const Vector4 &q, const Vector3 &p) {
    const Real q1 = q(0), q2 = q(1), q3 = q(2), q4 = q(3);
    const Real p1 = p(0), p2 = p(1), p3 = p(2);
    const Real px1 = -p1 * q2 - p2 * q3 - p3 * q4;
    const Real px2 = p1 * q1 - p2 * q4 + p3 * q3;
    const Real px3 = p2 * q1 + p1 * q4 - p3 * q2;
    const Real px4 = p2 * q2 - p1 * q3 + p3 * q1;
    return Vector3(px2 * q1 - px1 * q2 - px3 * q4 + px4 * q3, px3 * q1 - px1 * q3 + px2 * q4 - px4 * q2,
                   px3 * q2 - px2 * q3 - px1 * q4 + px4 * q1);
}
inline Vector4 quat_conj(const Vector4 &q) { return Vector4(q(0), -q(1), -q(2), -q(3)); }

inline Vector4 quat_exp(const Vector3 &w) {
    const Real theta2 = w.squaredNorm();
    const Real theta = std::sqrt(theta2);
    const Real theta_half = 0.5 * theta;

    Real re, im;
    if (theta > 1e-6) {
        re = std::cos(theta_half);
        im = std::sin(theta_half) / theta;
    } else {
        // we are close to zero, use taylor expansion to avoid problems
        // with zero divisors in sin(theta/2)/theta
        const Real theta4 = theta2 * theta2;
        re = 1.0 - (1.0 / 8.0) * theta2 + (1.0 / 384.0) * theta4;
        im = 0.5 - (1.0 / 48.0) * theta2 + (1.0 / 3840.0) * theta4;

        // for the linearized part we re-normalize to ensure unit length
        // here s should be roughly 1.0 anyways, so no problem with zero div
        const Real s = std::sqrt(re * re + im * im * theta2);
        re /= s;
        im /= s;
    }
    return Vector4(re, im * w(0), im * w(1), im * w(2));
}

inline Vector4 quat_step_pre(const Vector4 &q, const Vector3 &w_delta) { return quat_multiply(quat_exp(w_delta), q); }
inline Vector4 quat_step_post(const Vector4 &q, const Vector3 &w_delta) { return quat_multiply(q, quat_exp(w_delta)); }

} // namespace poselib

#endif