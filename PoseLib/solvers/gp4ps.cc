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

#include "gp4ps.h"

#include "PoseLib/misc/re3q3.h"
#include "PoseLib/misc/univariate.h"

namespace poselib {

// Solves for camera pose such that: p+lambda*x = R*X+t
// Note: This function assumes that the bearing vectors (x) are normalized!
int gp4ps(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
          const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output, std::vector<double> *output_scales,
          bool filter_solutions) {

    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) {
            if ((X[i] - X[j]).squaredNorm() < 1e-10) {

                // we have a duplicated 3d point
                std::vector<Eigen::Vector3d> pp = p;
                std::vector<Eigen::Vector3d> xp = x;
                std::vector<Eigen::Vector3d> Xp = X;

                std::swap(pp[0], pp[i]);
                std::swap(xp[0], xp[i]);
                std::swap(Xp[0], Xp[i]);

                std::swap(pp[1], pp[j]);
                std::swap(xp[1], xp[j]);
                std::swap(Xp[1], Xp[j]);

                return gp4ps_camposeco(pp, xp, Xp, output, output_scales);
            }
        }
    }

    return gp4ps_kukelova(p, x, X, output, output_scales, filter_solutions);
}

// Solves for camera pose such that: scale*p+lambda*x = R*X+t
int gp4ps_kukelova(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                   const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output,
                   std::vector<double> *output_scales, bool filter_solutions) {

    Eigen::Matrix<double, 8, 13> A;

    for (int i = 0; i < 4; ++i) {
        // xx = [x3 0 -x1; 0 x3 -x2]
        // eqs = [xx kron(X',xx), -xx*p] * [t; scale; vec(R)]

        A.row(2 * i) << x[i](2), 0.0, -x[i](0), -p[i](0) * x[i](2) + p[i](2) * x[i](0), X[i](0) * x[i](2), 0.0,
            -X[i](0) * x[i](0), X[i](1) * x[i](2), 0.0, -X[i](1) * x[i](0), X[i](2) * x[i](2), 0.0, -X[i](2) * x[i](0);
        A.row(2 * i + 1) << 0.0, x[i](2), -x[i](1), -p[i](1) * x[i](2) + p[i](2) * x[i](1), 0.0, X[i](0) * x[i](2),
            -X[i](0) * x[i](1), 0.0, X[i](1) * x[i](2), -X[i](1) * x[i](1), 0.0, X[i](2) * x[i](2), -X[i](2) * x[i](1);
    }

    Eigen::Matrix4d B = A.block<4, 4>(0, 0).inverse();

    Eigen::Matrix<double, 3, 9> AR = A.block<3, 9>(4, 4) - A.block<3, 4>(4, 0) * B * A.block<4, 9>(0, 4);

    Eigen::Matrix<double, 4, 8> solutions;
    int n_sols = re3q3::re3q3_rotation(AR, &solutions);

    Eigen::Vector4d ts;

    output->clear();
    output_scales->clear();
    CameraPose best_pose;
    double best_scale = 1.0;
    double best_res = 0.0;
    for (int i = 0; i < n_sols; ++i) {
        CameraPose pose;
        pose.q = solutions.col(i);
        ts = -B * (A.block<4, 9>(0, 4) * quat_to_rotmatvec(pose.q));
        pose.t = ts.block<3, 1>(0, 0);
        double scale = ts(3);

        if (filter_solutions) {
            double res = std::abs(x[3].dot((pose.R() * X[3] + pose.t - scale * p[3]).normalized()));
            if (res > best_res) {
                best_pose = pose;
                best_scale = scale;
                best_res = res;
            }
        } else {
            output->push_back(pose);
            output_scales->push_back(scale);
        }
    }

    if (filter_solutions && best_res > 0.0) {
        output->push_back(best_pose);
        output_scales->push_back(best_scale);
    }

    return output->size();
}

// Solves for camera pose such that: scale*p+lambda*x = R*X+t
// Assumes that X[0] == X[1] !
int gp4ps_camposeco(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                    const std::vector<Eigen::Vector3d> &X, std::vector<CameraPose> *output,
                    std::vector<double> *output_scales) {
    // Locally triangulate the 3D point
    const double a = x[0].dot(x[1]);
    const double b1 = x[0].dot(p[1] - p[0]);
    const double b2 = x[1].dot(p[1] - p[0]);
    const double lambda = (a * b2 - b1) / (a * a - 1);

    const Eigen::Vector3d Xc = p[0] + lambda * x[0];

    // Shift rig coordinate system by Xc
    Eigen::Vector3d q0 = p[2] - Xc;
    Eigen::Vector3d q1 = p[3] - Xc;

    // Ensure q is orthogonal to x
    q0 -= q0.dot(x[2]) * x[2];
    q1 -= q1.dot(x[3]) * x[3];
    const double D21 = (X[2] - X[0]).squaredNorm();
    const double D31 = (X[3] - X[0]).squaredNorm();
    const double D23 = (X[3] - X[2]).squaredNorm();

    const double inv1 = 1.0 / D31;
    const double k1 = -inv1 * D21;
    const double k2 = inv1 * (D31 * (q0(0) * q0(0) + q0(1) * q0(1) + q0(2) * q0(2)) -
                              D21 * (q1(0) * q1(0) + q1(1) * q1(1) + q1(2) * q1(2)));
    const double inv2 = 1.0 / (D21 * (x[2](0) * x[2](0) + x[2](1) * x[2](1) + x[2](2) * x[2](2)) -
                               D23 * (x[2](0) * x[2](0) + x[2](1) * x[2](1) + x[2](2) * x[2](2)));
    const double k3 = inv2 * (-D21 * (2 * x[2](0) * x[3](0) + 2 * x[2](1) * x[3](1) + 2 * x[2](2) * x[3](2)));
    const double k4 = inv2 * (D21 * (x[3](0) * x[3](0) + x[3](1) * x[3](1) + x[3](2) * x[3](2)));
    const double k5 =
        inv2 * (D21 * (2 * x[2](0) * (q0(0) - q1(0)) + 2 * x[2](1) * (q0(1) - q1(1)) + 2 * x[2](2) * (q0(2) - q1(2))) -
                D23 * (2 * q0(0) * x[2](0) + 2 * q0(1) * x[2](1) + 2 * q0(2) * x[2](2)));
    const double k6 =
        inv2 * (-D21 * (2 * x[3](0) * (q0(0) - q1(0)) + 2 * x[3](1) * (q0(1) - q1(1)) + 2 * x[3](2) * (q0(2) - q1(2))));
    const double k7 = inv2 * (D21 * ((q0(0) - q1(0)) * (q0(0) - q1(0)) + (q0(1) - q1(1)) * (q0(1) - q1(1)) +
                                     (q0(2) - q1(2)) * (q0(2) - q1(2))) -
                              D23 * (q0(0) * q0(0) + q0(1) * q0(1) + q0(2) * q0(2)));

    // Quartic in lambda3
    const double inv_c4 = 1.0 / (k1 * k1 + k3 * k3 * k1 - 2 * k4 * k1 + k4 * k4);
    const double c3 = inv_c4 * 2.0 * (k1 * k3 * k5 - k1 * k6 + k4 * k6);
    const double c2 = inv_c4 * (k2 * k3 * k3 + k1 * k5 * k5 + k6 * k6 + 2.0 * k1 * k2 - 2.0 * k2 * k4 - 2.0 * k1 * k7 +
                                2.0 * k4 * k7);
    const double c1 = inv_c4 * (2.0 * k2 * k3 * k5 - 2.0 * k2 * k6 + 2.0 * k6 * k7);
    const double c0 = inv_c4 * (k2 * k2 + k2 * k5 * k5 + k7 * k7 - 2.0 * k2 * k7);

    double roots[4];
    const int n_sols = univariate::solve_quartic_real(c3, c2, c1, c0, roots);

    Eigen::Matrix3d YY;
    YY.col(0) = X[2] - X[0];
    YY.col(1) = X[3] - X[0];
    YY.col(2) = YY.col(0).cross(YY.col(1));
    const double sY = YY.col(0).norm();
    YY = YY.inverse().eval();

    Eigen::Matrix3d XX;

    output->clear();
    output_scales->clear();
    for (int i = 0; i < n_sols; ++i) {
        const double lambda3 = roots[i];
        const double lambda2 = (k2 - k7 + (k1 - k4) * lambda3 * lambda3 - k6 * lambda3) / (k3 * lambda3 + k5);

        XX.col(0) = q0 + lambda2 * x[2];
        XX.col(1) = q1 + lambda3 * x[3];

        CameraPose pose;
        double scale = sY / (XX.col(0)).norm();

        XX.col(0) *= scale;
        XX.col(1) *= scale;
        XX.col(2) = XX.col(0).cross(XX.col(1));

        pose.q = rotmat_to_quat(XX * YY);
        pose.t = scale * Xc - pose.R() * X[0];

        output->push_back(pose);
        output_scales->push_back(scale);
    }
    return output->size();
}

} // namespace poselib