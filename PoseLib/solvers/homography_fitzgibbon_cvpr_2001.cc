// Copyright (c) 2020 Marcus Valtonen Örnhag
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "homography_fitzgibbon_cvpr_2001.h"

#include <float.h>  // For DBL_MAX
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include "PoseLib/misc/radial.h"

namespace poselib {
inline Eigen::Matrix3d vec2asym(const Eigen::Vector3d &t);

int homography_fitzgibbon_cvpr_2001(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2, Eigen::Matrix3d *H, double *distortion_parameter)
{
//(HomLib::PoseData get(const Eigen::MatrixXd& x1n, const Eigen::MatrixXd& x2n) {
    // This is a five point method
    int n_points = 5;

    // Make homogenous
    Eigen::MatrixXd y1(3, n_points);
    Eigen::MatrixXd y2(3, n_points);
    for (int i = 0; i < n_points; ++i) {
        y1.col(i) = x1[i]; // TODO: Last entry must be one? Check paper
        y2.col(i) = x2[i];
    }

    // Compute the distance to center point
    Eigen::MatrixXd z1(3, n_points);
    z1 << Eigen::MatrixXd::Zero(2, n_points), y1.colwise().hnormalized().colwise().squaredNorm();
    Eigen::MatrixXd z2(3, n_points);
    z2 << Eigen::MatrixXd::Zero(2, n_points), y2.colwise().hnormalized().colwise().squaredNorm();

    // Initialize D0, D1 and D2
    Eigen::MatrixXd D0(9, 9);
    D0.setZero();
    Eigen::MatrixXd D1(9, 9);
    D1.setZero();
    Eigen::MatrixXd D2(9, 9);
    D2.setZero();

    Eigen::Matrix3d Bx2, Bz2;
    Eigen::Matrix3d e1, e2;
    Eigen::Vector3d tmp;

    for (int k = 0; k < n_points; k++) {
        tmp = y2.col(k);
        Bx2 = poselib::vec2asym(tmp);
        tmp = z2.col(k);
        Bz2 = poselib::vec2asym(tmp);

        // D0
        e1 = Bx2.row(0).transpose() * y1.col(k).transpose();
        e2 = Bx2.row(1).transpose() * y1.col(k).transpose();
        D0.row(2*k) = Eigen::Map<Eigen::VectorXd>(e1.data(), 9);
        if (k < 4) {  // Assure it is 9x9
            D0.row(2*k + 1) = Eigen::Map<Eigen::VectorXd>(e2.data(), 9);
        }
        // D1
        e1 = Bx2.row(0).transpose() * z1.col(k).transpose() + Bz2.row(0).transpose() * y1.col(k).transpose();
        e2 = Bx2.row(1).transpose() * z1.col(k).transpose() + Bz2.row(1).transpose() * y1.col(k).transpose();
        D1.row(2*k) = Eigen::Map<Eigen::VectorXd>(e1.data(), 9);
        if (k < 4) {
            D1.row(2*k + 1) = Eigen::Map<Eigen::VectorXd>(e2.data(), 9);
        }

        // D2
        e1 = Bz2.row(0).transpose() * z1.col(k).transpose();
        e2 = Bz2.row(1).transpose() * z1.col(k).transpose();
        D2.row(2*k) = Eigen::Map<Eigen::VectorXd>(e1.data(), 9);
        if (k < 4) {
            D2.row(2*k + 1) = Eigen::Map<Eigen::VectorXd>(e2.data(), 9);
        }
    }

    // Create generalized eigenvalue problem
    Eigen::MatrixXd A(18, 18);
    Eigen::MatrixXd B(18, 18);
    A.setZero();
    B.setZero();

    A.topLeftCorner(9, 9) = -D0;
    A.bottomRightCorner(9, 9) = Eigen::MatrixXd::Identity(9, 9);
    B.topLeftCorner(9, 9) = D1;
    B.topRightCorner(9, 9) = D2;
    B.bottomLeftCorner(9, 9) = Eigen::MatrixXd::Identity(9, 9);

    Eigen::GeneralizedEigenSolver<Eigen::MatrixXd> ges;
    ges.compute(A, B, true);
    Eigen::VectorXcd l;
    Eigen::MatrixXd X;
    l = ges.eigenvalues();
    Eigen::MatrixXcd eigvecs;
    eigvecs = ges.eigenvectors();
    X = eigvecs.real().topRows(9);

    // Extract correct solution
    Eigen::Matrix3d Htmp;
    Eigen::MatrixXd z(3, 5);
    double res;
    double minres = DBL_MAX;
    double ltmp;
    Eigen::Array<bool, 1, 18> is_ok;
    is_ok = l.array().isFinite() && l.array().imag() == 0;

    for (int k = 0; k < 18; k++) {
        if (is_ok(k)) {
            ltmp = l(k).real();
            Htmp = Eigen::Map<Eigen::Matrix3d>(X.col(k).data(), 3, 3);
            z = Htmp * radialundistort(y1.colwise().hnormalized(), ltmp).colwise().homogeneous();
            res = (y2.colwise().hnormalized() - radialdistort(z.colwise().hnormalized(), ltmp)).squaredNorm();
            if (res < minres) {
                minres = res;
                *H = Htmp;
                *distortion_parameter = ltmp;
            }
        }
    }

    return 1;
}

// TODO(marcusvaltonen): Refactor -> helpers when necessary
inline Eigen::Matrix3d vec2asym(const Eigen::Vector3d &t) {
    Eigen::Matrix3d t_hat;
    t_hat << 0, -t(2), t(1),
             t(2), 0, -t(0),
            -t(1), t(0), 0;
    return t_hat;
}
}  // namespace poselib
