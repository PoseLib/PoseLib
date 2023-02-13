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

#ifndef POSELIB_HOMOGRAPHY_VALTONENORNHAG_WACV_2021_FHF_H_
#define POSELIB_HOMOGRAPHY_VALTONENORNHAG_WACV_2021_FHF_H_

#include <Eigen/Dense>
#include <vector>

namespace poselib {

/**
 * Minimal 2-point algorithm for estimating a homography with
 * known rotation and unknown (and equal) focal length.
 * Implementation of [1], section 4.2.
 *
 * [1] "Efficient real-time radial distortion correction for UAVs",
 * Valtonen Ornhag et al., WACV 2021
 *
 */
int homography_valtonenornhag_wacv_2021_fHf(
    const std::vector<Eigen::Vector3d> &p1,
    const std::vector<Eigen::Vector3d> &p2,
    const Eigen::Matrix3d &R1,
    const Eigen::Matrix3d &R2,
    std::vector<Eigen::Matrix3d> *H,
    std::vector<double> *f
);

}  // namespace poselib

#endif  // POSELIB_HOMOGRAPHY_VALTONENORNHAG_WACV_2021_FHF_H_
