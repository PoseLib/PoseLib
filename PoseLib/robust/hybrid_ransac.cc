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
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "hybrid_ransac.h"

#include <PoseLib/robust/hybrid_estimators/hybrid_point_line_absolute_pose.h>
#include <PoseLib/robust/hybrid_ransac_impl.h>
#include <PoseLib/robust/utils.h>

namespace poselib {

HybridRansacStats hybrid_ransac_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                     const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                                     const HybridRansacOptions &opt, CameraPose *pose,
                                     std::vector<char> *inliers_points, std::vector<char> *inliers_lines) {
    // Initialize pose
    pose->q << 1.0, 0.0, 0.0, 0.0;
    pose->t.setZero();

    // Create estimator
    HybridPointLineAbsolutePoseEstimator estimator(opt, points2D, points3D, lines2D, lines3D);

    // Run hybrid RANSAC
    HybridRansacStats stats = hybrid_ransac(estimator, opt, pose);

    // Get final inliers (square thresholds since options store unsquared)
    if (opt.max_errors.size() >= 2) {
        double sq_threshold_points = opt.max_errors[0] * opt.max_errors[0];
        double sq_threshold_lines = opt.max_errors[1] * opt.max_errors[1];
        get_inliers(*pose, points2D, points3D, sq_threshold_points, inliers_points);
        get_inliers(*pose, lines2D, lines3D, sq_threshold_lines, inliers_lines);
    }

    return stats;
}

} // namespace poselib
