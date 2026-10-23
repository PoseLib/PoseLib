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
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <cmath>
#include <vector>

namespace poselib {

// The gp3p, gp4ps (Kukelova), up2p, ugp2p and ugp3ps solvers keep only the two rows
//     [x3 0 -x1; 0 x3 -x2]
// of the cross product [x]_x (R*X + t - p) = 0. For a bearing vector with x3 = 0 these two rows are
// dependent and the constraint carried by the third row is lost, so the solver returns no pose or a wrong
// one although the problem is well posed. gp3p in addition eliminates t through the 3x3 block of rows
// {0, 1, 2}, whose determinant is x0_3 * (x0_1 * x1_3 - x0_3 * x1_1): it is singular when rays 0 and 1
// have the same x1/x3 ratio (the same image column). Both conditions depend on the coordinate frame of the
// rig, not on the problem. The helpers below detect them and give a rotation Q of the rig frame in which
// they are far away; the solver then runs on (Q*p, Q*x) and the poses are rotated back.

constexpr double kRigFrameTolerance = 1e-3;

// True when some bearing vector is (nearly) orthogonal to the z-axis, or when rays 0 and 1 (nearly) share
// x1/x3 (only checked if check_column is set; it matters for gp3p).
inline bool rig_frame_is_ill_conditioned(const std::vector<Eigen::Vector3d> &x, size_t n, bool check_column) {
    for (size_t i = 0; i < n; ++i) {
        if (std::abs(x[i](2)) < kRigFrameTolerance * x[i].norm())
            return true;
    }
    if (check_column) {
        const double column_det = x[0](0) * x[1](2) - x[0](2) * x[1](0);
        if (std::abs(column_det) < kRigFrameTolerance * x[0].norm() * x[1].norm())
            return true;
    }
    return false;
}

// General rotation (rows ex, ey, ez). ez is the normalized signed sum of the n bearing vectors whose signs
// maximize its norm: flipping any one sign cannot increase |s|, which gives |x_i . s| >= |x_i|^2 and
// therefore |x_i . ez| >= 1/n for unit x_i. ex is the difference of rays 0 and 1 on the plane z' = 1, so in
// the new frame their x1/x3 ratios differ by the length of that difference (zero only for parallel rays).
inline Eigen::Matrix3d well_conditioned_rig_frame(const std::vector<Eigen::Vector3d> &x, size_t n) {
    Eigen::Vector3d best = x[0];
    double best_norm = -1.0;
    for (int mask = 0; mask < (1 << (n - 1)); ++mask) {
        Eigen::Vector3d s = x[0];
        for (size_t i = 1; i < n; ++i)
            s += ((mask >> (i - 1)) & 1) ? Eigen::Vector3d(-x[i]) : x[i];
        const double sn = s.squaredNorm();
        if (sn > best_norm) {
            best_norm = sn;
            best = s;
        }
    }
    const Eigen::Vector3d ez = best.normalized();
    Eigen::Vector3d d = x[0] / x[0].dot(ez) - x[1] / x[1].dot(ez);
    d -= d.dot(ez) * ez;
    if (d.squaredNorm() < 1e-24)
        d = ez.unitOrthogonal();
    const Eigen::Vector3d ex = d.normalized();
    const Eigen::Vector3d ey = ez.cross(ex);
    Eigen::Matrix3d Q;
    Q.row(0) = ex;
    Q.row(1) = ey;
    Q.row(2) = ez;
    return Q;
}

// Rotation about the y-axis (gravity) for the upright solvers, so that the rotation to estimate stays
// upright. The new z-axis is the signed sum of the bearing vectors projected onto the xz-plane.
inline Eigen::Matrix3d well_conditioned_upright_frame(const std::vector<Eigen::Vector3d> &x, size_t n) {
    Eigen::Vector2d best(x[0](0), x[0](2));
    double best_norm = -1.0;
    for (int mask = 0; mask < (1 << (n - 1)); ++mask) {
        Eigen::Vector2d s(x[0](0), x[0](2));
        for (size_t i = 1; i < n; ++i) {
            const Eigen::Vector2d xi(x[i](0), x[i](2));
            s += ((mask >> (i - 1)) & 1) ? Eigen::Vector2d(-xi) : xi;
        }
        const double sn = s.squaredNorm();
        if (sn > best_norm) {
            best_norm = sn;
            best = s;
        }
    }
    Eigen::Matrix3d Q = Eigen::Matrix3d::Identity();
    if (best.squaredNorm() < 1e-24)
        return Q;
    best.normalize();
    // Rows: ex = (n_z, 0, -n_x), ey = (0, 1, 0), ez = (n_x, 0, n_z): a rotation about y.
    Q(0, 0) = best(1);
    Q(0, 2) = -best(0);
    Q(2, 0) = best(0);
    Q(2, 2) = best(1);
    return Q;
}

// p + lambda * x = R * X + t  <=>  Q p + lambda * Q x = (Q R) X + Q t
inline void rotate_rig(const Eigen::Matrix3d &Q, const std::vector<Eigen::Vector3d> &v, size_t n,
                       std::vector<Eigen::Vector3d> *out) {
    out->resize(v.size());
    for (size_t i = 0; i < v.size(); ++i)
        (*out)[i] = (i < n) ? Eigen::Vector3d(Q * v[i]) : v[i];
}

inline void derotate_poses(const Eigen::Matrix3d &Q, std::vector<CameraPose> *poses) {
    for (CameraPose &pose : *poses) {
        const Eigen::Matrix3d R = Q.transpose() * pose.R();
        const Eigen::Vector3d t = Q.transpose() * pose.t;
        pose = CameraPose(R, t);
    }
}

} // namespace poselib
