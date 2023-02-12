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

#include <Eigen/Dense>

#include "radial.h"

namespace poselib {
Eigen::MatrixXd radialdistort(const Eigen::MatrixXd& x, double kappa) {
    // We expect inhomogenous input data
    assert(x.rows() == 2);

    Eigen::ArrayXd ru2 = x.colwise().squaredNorm();
    Eigen::ArrayXd ru = ru2.sqrt();

    Eigen::ArrayXd rd;
    // Compute distorted radius
    if (kappa == 0) {
        rd = ru;
    } else {
        rd = 0.5 / kappa / ru - std::copysign(1.0, kappa) * (0.25 / std::pow(kappa, 2) / ru2 - 1.0 / kappa).sqrt();
    }

    // compute distorted coordinates
    Eigen::MatrixXd y(2, x.cols());
    y = (rd / ru).replicate(1, x.rows()).transpose() * x.array();

    // TODO(marcusvaltonen): Avoid divion by zero (centre coordinate - usually synthethic images)
    return y;
}

Eigen::MatrixXd radialundistort(const Eigen::MatrixXd& x, double kappa) {
    // We expect inhomogenous input data
    assert(x.rows() == 2);

    Eigen::VectorXd rd2 = x.colwise().squaredNorm();

    // Compute undistorted coordinates
    Eigen::MatrixXd y(3, x.cols());
    y.topRows(2) = x;
    y.bottomRows(1) = (Eigen::VectorXd::Ones(x.cols()) + kappa * rd2).transpose();

    return y.colwise().hnormalized();
}
}  // namespace poselib
