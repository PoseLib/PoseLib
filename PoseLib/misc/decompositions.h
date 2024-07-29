#ifndef POSELIB_DECOMPOSITIONS_H
#define POSELIB_DECOMPOSITIONS_H

#include "PoseLib/misc/colmap_models.h"
#include "PoseLib/types.h"

#include <Eigen/Core>

namespace poselib {

// Estimate two different focal lengths from F using the formula from:
// Bougnoux,"From Projective to Euclidean space under any practical situation, a criticism of self-calibration"(ICCV 98)
std::pair<Camera, Camera> focals_from_fundamental(const Eigen::Matrix3d &F, const Point2D &pp1, const Point2D &pp2);

// Estimate two different focal lengths using the method from
// Kocur, Kyselica, Kukelova, "Robust Self-calibration of Focal Lengths from the Fundamental Matrix" (CVPR 2024)
std::tuple<Camera, Camera, int>
focals_from_fundamental_iterative(const Eigen::Matrix3d &F, const Camera &camera1_prior, const Camera &camera2_prior,
                                  const int &max_iters = 50,
                                  const Eigen::Vector4d &weights = Eigen::Vector4d(5.0e-4, 1.0, 5.0e-4, 1.0));

} // namespace poselib

#endif // POSELIB_DECOMPOSITIONS_H
