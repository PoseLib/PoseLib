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

#ifndef POSELIB_HOMOGRAPHY_FITZGIBBON_CVPR_2001_H_
#define POSELIB_HOMOGRAPHY_FITZGIBBON_CVPR_2001_H_

#include <Eigen/Dense>

namespace poselib {

int homography_fitzgibbon_cvpr_2001(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2, Eigen::Matrix3d *H, double *distortion_parameter);

}; // namespace poselib

#endif  // POSELIB_HOMOGRAPHY_FITZGIBBON_CVPR_2001_H_
