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

#ifndef POSELIB_HOMOGRAPHY_KUKELOVA_CVPR_2015_H_
#define POSELIB_HOMOGRAPHY_KUKELOVA_CVPR_2015_H_

#include <Eigen/Dense>
#include <vector>

namespace poselib {

/**
 * Minimal five-point algorithm for solving for estimating a radially
 * distorted homography using the one-parameter division model, where the
 * distortion parameters need not to be the same.
 * Implementation of [1] section 3.
 *
 * [1] "Radial Distortion Homography", Kukelova et al., CVPR, 2015
 *
 */
int homography_kukelova_cvpr_2015(
    const std::vector<Eigen::Vector3d> &p1,
    const std::vector<Eigen::Vector3d> &p2,
    std::vector<Eigen::Matrix3d> *H,
    std::vector<double> *distortion_parameter1,
    std::vector<double> *distortion_parameter2
);

} // poselib

#endif  // POSELIB_HOMOGRAPHY_KUKELOVA_CVPR_2015_H_
