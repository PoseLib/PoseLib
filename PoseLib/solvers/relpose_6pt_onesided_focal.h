#pragma once

#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {

// Solves for relative pose with one unknown focal length from 6 correspondences
// (one-sided focal). The first camera (observing x1) is uncalibrated with unknown
// focal length and zero principal point; the second camera (observing x2) is
// calibrated. The focal enters via Kinv = diag(1, 1, f) where E = F * Kinv,
// and diag(1, 1, f) * x1 is the calibrated bearing in the first camera.
// The solver was introduced in
//    Kukelova et al., A Clever Elimination Strategy for Efficient Minimal Solvers, CVPR 2017
// The principal point is assumed to be centered.
// The solver returns the ImagePair where you can recover
// the pose:     out_image_pairs[k].pose
// and focal:    out_image_pairs[k].camera1.focal()
int relpose_6pt_onesided_focal(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                               ImagePairVector *out_image_pairs, bool use_elim = true);

}; // namespace poselib
