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
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "robust.h"

#include "PoseLib/robust/utils.h"

namespace poselib {

RansacStats estimate_absolute_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                   const Camera &camera, const RansacOptions &ransac_opt,
                                   const BundleOptions &bundle_opt, CameraPose *pose, std::vector<char> *inliers) {

    std::vector<Point2D> points2D_calib(points2D.size());
    for (size_t k = 0; k < points2D.size(); ++k) {
        camera.unproject(points2D[k], &points2D_calib[k]);
    }

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_reproj_error /= camera.focal();

    RansacStats stats = ransac_pnp(points2D_calib, points3D, ransac_opt_scaled, pose, inliers);

    if (stats.num_inliers > 3) {
        // Collect inlier for additional bundle adjustment
        std::vector<Point2D> points2D_inliers;
        std::vector<Point3D> points3D_inliers;
        points2D_inliers.reserve(points2D.size());
        points3D_inliers.reserve(points3D.size());

        // We re-scale with focal length to improve numerics in the opt.
        const double scale = 1.0 / camera.focal();
        Camera norm_camera = camera;
        norm_camera.rescale(scale);
        BundleOptions bundle_opt_scaled = bundle_opt;
        bundle_opt_scaled.loss_scale *= scale;
        for (size_t k = 0; k < points2D.size(); ++k) {
            if (!(*inliers)[k])
                continue;
            points2D_inliers.push_back(points2D[k] * scale);
            points3D_inliers.push_back(points3D[k]);
        }

        bundle_adjust(points2D_inliers, points3D_inliers, norm_camera, pose, bundle_opt_scaled);
    }

    return stats;
}

RansacStats estimate_generalized_absolute_pose(const std::vector<std::vector<Point2D>> &points2D,
                                               const std::vector<std::vector<Point3D>> &points3D,
                                               const std::vector<CameraPose> &camera_ext,
                                               const std::vector<Camera> &cameras, const RansacOptions &ransac_opt,
                                               const BundleOptions &bundle_opt, CameraPose *pose,
                                               std::vector<std::vector<char>> *inliers) {

    const size_t num_cams = cameras.size();

    // Normalize image points for the RANSAC
    std::vector<std::vector<Point2D>> points2D_calib;
    points2D_calib.resize(num_cams);
    double scaled_threshold = 0;
    size_t total_num_pts = 0;
    for (size_t cam_k = 0; cam_k < num_cams; ++cam_k) {
        const size_t pts = points2D[cam_k].size();
        points2D_calib[cam_k].resize(pts);
        for (size_t pt_k = 0; pt_k < pts; ++pt_k) {
            cameras[cam_k].unproject(points2D[cam_k][pt_k], &points2D_calib[cam_k][pt_k]);
        }
        total_num_pts += pts;
        scaled_threshold += (ransac_opt.max_reproj_error * pts) / cameras[cam_k].focal();
    }
    scaled_threshold /= static_cast<double>(total_num_pts);

    // TODO allow per-camera thresholds
    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_reproj_error = scaled_threshold;

    RansacStats stats = ransac_gen_pnp(points2D_calib, points3D, camera_ext, ransac_opt_scaled, pose, inliers);

    if (stats.num_inliers > 3) {
        // Collect inlier for additional bundle adjustment
        std::vector<std::vector<Point2D>> points2D_inliers;
        std::vector<std::vector<Point3D>> points3D_inliers;
        points2D_inliers.resize(num_cams);
        points3D_inliers.resize(num_cams);

        for (size_t cam_k = 0; cam_k < num_cams; ++cam_k) {
            const size_t pts = points2D[cam_k].size();
            points2D_inliers[cam_k].reserve(pts);
            points3D_inliers[cam_k].reserve(pts);

            for (size_t pt_k = 0; pt_k < pts; ++pt_k) {
                if (!(*inliers)[cam_k][pt_k])
                    continue;
                points2D_inliers[cam_k].push_back(points2D[cam_k][pt_k]);
                points3D_inliers[cam_k].push_back(points3D[cam_k][pt_k]);
            }
        }

        generalized_bundle_adjust(points2D_inliers, points3D_inliers, camera_ext, cameras, pose, bundle_opt);
    }

    return stats;
}

RansacStats estimate_absolute_pose_pnpl(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                        const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                                        const Camera &camera, const RansacOptions &ransac_opt,
                                        const BundleOptions &bundle_opt, CameraPose *pose,
                                        std::vector<char> *inliers_points, std::vector<char> *inliers_lines) {

    std::vector<Point2D> points2D_calib(points2D.size());
    for (size_t k = 0; k < points2D.size(); ++k) {
        camera.unproject(points2D[k], &points2D_calib[k]);
    }

    std::vector<Line2D> lines2D_calib(lines2D.size());
    for (size_t k = 0; k < lines2D.size(); ++k) {
        camera.unproject(lines2D[k].x1, &lines2D_calib[k].x1);
        camera.unproject(lines2D[k].x2, &lines2D_calib[k].x2);
    }

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_reproj_error /= camera.focal();
    ransac_opt_scaled.max_epipolar_error /= camera.focal();

    RansacStats stats = ransac_pnpl(points2D_calib, points3D, lines2D_calib, lines3D, ransac_opt_scaled, pose,
                                    inliers_points, inliers_lines);

    if (stats.num_inliers > 3) {
        // Collect inlier for additional bundle adjustment
        std::vector<Point2D> points2D_inliers;
        std::vector<Point3D> points3D_inliers;
        points2D_inliers.reserve(points2D.size());
        points3D_inliers.reserve(points3D.size());
        for (size_t k = 0; k < inliers_points->size(); ++k) {
            if (!(*inliers_points)[k])
                continue;
            points2D_inliers.push_back(points2D_calib[k]);
            points3D_inliers.push_back(points3D[k]);
        }

        std::vector<Line2D> lines2D_inliers;
        std::vector<Line3D> lines3D_inliers;
        lines2D_inliers.reserve(lines2D.size());
        lines3D_inliers.reserve(lines3D.size());
        for (size_t k = 0; k < inliers_lines->size(); ++k) {
            if (!(*inliers_lines)[k])
                continue;
            lines2D_inliers.push_back(lines2D_calib[k]);
            lines3D_inliers.push_back(lines3D[k]);
        }

        BundleOptions bundle_opt_scaled = bundle_opt;
        bundle_opt_scaled.loss_scale /= camera.focal();

        bundle_adjust(points2D_inliers, points3D_inliers, lines2D_inliers, lines3D_inliers, pose, bundle_opt_scaled,
                      bundle_opt_scaled);
    }

    return stats;
}

RansacStats estimate_relative_pose(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2,
                                   const Camera &camera1, const Camera &camera2, const RansacOptions &ransac_opt,
                                   const BundleOptions &bundle_opt, CameraPose *pose, std::vector<char> *inliers) {

    const size_t num_pts = points2D_1.size();

    std::vector<Point2D> x1_calib(num_pts);
    std::vector<Point2D> x2_calib(num_pts);
    for (size_t k = 0; k < num_pts; ++k) {
        camera1.unproject(points2D_1[k], &x1_calib[k]);
        camera2.unproject(points2D_2[k], &x2_calib[k]);
    }

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_epipolar_error =
        ransac_opt.max_epipolar_error * 0.5 * (1.0 / camera1.focal() + 1.0 / camera2.focal());

    RansacStats stats = ransac_relpose(x1_calib, x2_calib, ransac_opt_scaled, pose, inliers);

    if (stats.num_inliers > 5) {
        // Collect inlier for additional bundle adjustment
        // TODO: use camera models for this refinement!
        std::vector<Point2D> x1_inliers;
        std::vector<Point2D> x2_inliers;
        x1_inliers.reserve(stats.num_inliers);
        x2_inliers.reserve(stats.num_inliers);

        for (size_t k = 0; k < num_pts; ++k) {
            if (!(*inliers)[k])
                continue;
            x1_inliers.push_back(x1_calib[k]);
            x2_inliers.push_back(x2_calib[k]);
        }

        BundleOptions scaled_bundle_opt = bundle_opt;
        scaled_bundle_opt.loss_scale = bundle_opt.loss_scale * 0.5 * (1.0 / camera1.focal() + 1.0 / camera2.focal());

        refine_relpose(x1_inliers, x2_inliers, pose, scaled_bundle_opt);
    }

    return stats;
}

RansacStats estimate_shared_focal_relative_pose(const std::vector<Point2D> &points2D_1,
                                                const std::vector<Point2D> &points2D_2, const Point2D &pp,
                                                const RansacOptions &ransac_opt, const BundleOptions &bundle_opt,
                                                ImagePair *image_pair, std::vector<char> *inliers) {

    const size_t num_pts = points2D_1.size();

    Eigen::Matrix3d T1, T2;
    std::vector<Point2D> x1_norm = points2D_1;
    std::vector<Point2D> x2_norm = points2D_2;

    for (size_t i = 0; i < x1_norm.size(); i++) {
        x1_norm[i] -= pp;
        x2_norm[i] -= pp;
    }

    // We normalize points here to improve conditioning. Note that the normalization
    // only ammounts to a uniform rescaling of the image coordinate system
    // and the cost we minimize is equivalent to the cost in the original image
    // We do not perform shifting as we require pp to remain at [0, 0]
    double scale = normalize_points(x1_norm, x2_norm, T1, T2, true, false, true);

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_epipolar_error /= scale;
    BundleOptions bundle_opt_scaled = bundle_opt;
    bundle_opt_scaled.loss_scale /= scale;

    RansacStats stats = ransac_shared_focal_relpose(x1_norm, x2_norm, ransac_opt_scaled, image_pair, inliers);

    if (stats.num_inliers > 6) {
        std::vector<Point2D> x1_inliers;
        std::vector<Point2D> x2_inliers;
        x1_inliers.reserve(stats.num_inliers);
        x2_inliers.reserve(stats.num_inliers);

        for (size_t k = 0; k < num_pts; ++k) {
            if (!(*inliers)[k])
                continue;
            x1_inliers.push_back(x1_norm[k]);
            x2_inliers.push_back(x2_norm[k]);
        }

        refine_shared_focal_relpose(x1_inliers, x2_inliers, image_pair, bundle_opt_scaled);
    }

    image_pair->camera1.params[0] *= scale;
    image_pair->camera1.params[1] = pp(0);
    image_pair->camera1.params[2] = pp(1);

    image_pair->camera2 = image_pair->camera1;

    return stats;
}

RansacStats estimate_fundamental(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                 const RansacOptions &ransac_opt, const BundleOptions &bundle_opt, Eigen::Matrix3d *F,
                                 std::vector<char> *inliers) {

    const size_t num_pts = x1.size();
    if (num_pts < 7) {
        return RansacStats();
    }

    // We normalize points here to improve conditioning. Note that the normalization
    // only ammounts to a uniform rescaling and shift of the image coordinate system
    // and the cost we minimize is equivalent to the cost in the original image
    // for RFC we do not perform the shift as the pp needs to remain at [0, 0]

    Eigen::Matrix3d T1, T2;
    std::vector<Point2D> x1_norm = x1;
    std::vector<Point2D> x2_norm = x2;

    double scale = normalize_points(x1_norm, x2_norm, T1, T2, true, !ransac_opt.real_focal_check, true);
    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_epipolar_error /= scale;
    BundleOptions bundle_opt_scaled = bundle_opt;
    bundle_opt_scaled.loss_scale /= scale;

    RansacStats stats = ransac_fundamental(x1_norm, x2_norm, ransac_opt_scaled, F, inliers);

    if (stats.num_inliers > 7) {
        // Collect inlier for additional non-linear refinement
        std::vector<Point2D> x1_inliers;
        std::vector<Point2D> x2_inliers;
        x1_inliers.reserve(stats.num_inliers);
        x2_inliers.reserve(stats.num_inliers);

        for (size_t k = 0; k < num_pts; ++k) {
            if (!(*inliers)[k])
                continue;
            x1_inliers.push_back(x1_norm[k]);
            x2_inliers.push_back(x2_norm[k]);
        }

        refine_fundamental(x1_inliers, x2_inliers, F, bundle_opt_scaled);
    }

    *F = T2.transpose() * (*F) * T1;
    *F /= F->norm();

    return stats;
}

RansacStats estimate_homography(const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                                const RansacOptions &ransac_opt, const BundleOptions &bundle_opt, Eigen::Matrix3d *H,
                                std::vector<char> *inliers) {

    const size_t num_pts = x1.size();
    if (num_pts < 4) {
        return RansacStats();
    }

    Eigen::Matrix3d T1, T2;
    std::vector<Point2D> x1_norm = x1;
    std::vector<Point2D> x2_norm = x2;

    double scale = normalize_points(x1_norm, x2_norm, T1, T2, true, true, true);
    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_reproj_error /= scale;
    BundleOptions bundle_opt_scaled = bundle_opt;
    bundle_opt_scaled.loss_scale /= scale;

    RansacStats stats = ransac_homography(x1_norm, x2_norm, ransac_opt_scaled, H, inliers);

    if (stats.num_inliers > 4) {
        // Collect inlier for additional non-linear refinement
        std::vector<Point2D> x1_inliers;
        std::vector<Point2D> x2_inliers;
        x1_inliers.reserve(stats.num_inliers);
        x2_inliers.reserve(stats.num_inliers);

        for (size_t k = 0; k < num_pts; ++k) {
            if (!(*inliers)[k])
                continue;
            x1_inliers.push_back(x1_norm[k]);
            x2_inliers.push_back(x2_norm[k]);
        }

        refine_homography(x1_inliers, x2_inliers, H, bundle_opt_scaled);
    }

    *H = T2.inverse() * (*H) * T1;
    *H /= H->norm();

    return stats;
}

RansacStats estimate_generalized_relative_pose(const std::vector<PairwiseMatches> &matches,
                                               const std::vector<CameraPose> &camera1_ext,
                                               const std::vector<Camera> &cameras1,
                                               const std::vector<CameraPose> &camera2_ext,
                                               const std::vector<Camera> &cameras2, const RansacOptions &ransac_opt,
                                               const BundleOptions &bundle_opt, CameraPose *relative_pose,
                                               std::vector<std::vector<char>> *inliers) {

    std::vector<PairwiseMatches> calib_matches = matches;
    for (PairwiseMatches &m : calib_matches) {
        for (size_t k = 0; k < m.x1.size(); ++k) {
            cameras1[m.cam_id1].unproject(m.x1[k], &m.x1[k]);
            cameras2[m.cam_id2].unproject(m.x2[k], &m.x2[k]);
        }
    }

    double scaling_factor = 0;
    for (size_t k = 0; k < cameras1.size(); ++k) {
        scaling_factor += 1.0 / cameras1[k].focal();
    }
    for (size_t k = 0; k < cameras2.size(); ++k) {
        scaling_factor += 1.0 / cameras2[k].focal();
    }
    scaling_factor /= cameras1.size() + cameras2.size();

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_epipolar_error *= scaling_factor;

    RansacStats stats =
        ransac_gen_relpose(calib_matches, camera1_ext, camera2_ext, ransac_opt_scaled, relative_pose, inliers);

    if (stats.num_inliers > 6) {
        // Collect inlier for additional bundle adjustment
        // TODO: use camera models for this refinement!
        // TODO: check that inliers are actually meaningfully distributed

        std::vector<PairwiseMatches> inlier_matches;
        inlier_matches.resize(calib_matches.size());
        for (size_t match_k = 0; match_k < matches.size(); ++match_k) {
            const PairwiseMatches &m = calib_matches[match_k];

            inlier_matches[match_k].cam_id1 = m.cam_id1;
            inlier_matches[match_k].cam_id2 = m.cam_id2;
            inlier_matches[match_k].x1.reserve(m.x1.size());
            inlier_matches[match_k].x2.reserve(m.x2.size());

            for (size_t k = 0; k < m.x1.size(); ++k) {
                if ((*inliers)[match_k][k]) {
                    inlier_matches[match_k].x1.push_back(m.x1[k]);
                    inlier_matches[match_k].x2.push_back(m.x2[k]);
                }
            }
        }

        BundleOptions scaled_bundle_opt = bundle_opt;
        scaled_bundle_opt.loss_scale *= scaling_factor;

        refine_generalized_relpose(inlier_matches, camera1_ext, camera2_ext, relative_pose, scaled_bundle_opt);
    }
    return stats;
}

RansacStats estimate_hybrid_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                 const std::vector<PairwiseMatches> &matches2D_2D, const Camera &camera,
                                 const std::vector<CameraPose> &map_ext, const std::vector<Camera> &map_cameras,
                                 const RansacOptions &ransac_opt, const BundleOptions &bundle_opt, CameraPose *pose,
                                 std::vector<char> *inliers_2D_3D, std::vector<std::vector<char>> *inliers_2D_2D) {

    if (points2D.size() < 3) {
        // Not possible to generate minimal sample (until hybrid estimators are added into the ransac as well)
        return RansacStats();
    }

    // Compute normalized image points
    std::vector<PairwiseMatches> matches_calib = matches2D_2D;
    for (PairwiseMatches &m : matches_calib) {
        for (size_t k = 0; k < m.x1.size(); ++k) {
            map_cameras[m.cam_id1].unproject(m.x1[k], &m.x1[k]);
            camera.unproject(m.x2[k], &m.x2[k]);
        }
    }
    std::vector<Point2D> points2D_calib = points2D;
    for (size_t k = 0; k < points2D_calib.size(); ++k) {
        camera.unproject(points2D_calib[k], &points2D_calib[k]);
    }

    // TODO: different thresholds for 2D-2D and 2D-3D constraints
    double scaling_factor = 1.0 / camera.focal();
    for (size_t k = 0; k < map_cameras.size(); ++k) {
        scaling_factor += 1.0 / map_cameras[k].focal();
    }
    scaling_factor /= 1 + map_cameras.size();

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_reproj_error *= 1.0 / camera.focal();
    ransac_opt_scaled.max_epipolar_error *= scaling_factor;

    RansacStats stats = ransac_hybrid_pose(points2D_calib, points3D, matches_calib, map_ext, ransac_opt_scaled, pose,
                                           inliers_2D_3D, inliers_2D_2D);

    if (stats.num_inliers > 3) {
        // Collect inliers
        std::vector<Point2D> points2D_inliers;
        std::vector<Point3D> points3D_inliers;
        std::vector<PairwiseMatches> matches_inliers(matches_calib.size());
        points2D_inliers.reserve(points2D.size());
        points3D_inliers.reserve(points3D.size());

        for (size_t pt_k = 0; pt_k < inliers_2D_3D->size(); ++pt_k) {
            if ((*inliers_2D_3D)[pt_k]) {
                points2D_inliers.push_back(points2D_calib[pt_k]);
                points3D_inliers.push_back(points3D[pt_k]);
            }
        }

        for (size_t match_k = 0; match_k < inliers_2D_2D->size(); ++match_k) {
            matches_inliers[match_k].cam_id1 = matches_calib[match_k].cam_id1;
            matches_inliers[match_k].cam_id2 = matches_calib[match_k].cam_id2;

            matches_inliers[match_k].x1.reserve(matches_calib[match_k].x1.size());
            matches_inliers[match_k].x2.reserve(matches_calib[match_k].x1.size());

            for (size_t pt_k = 0; pt_k < (*inliers_2D_2D)[match_k].size(); ++pt_k) {
                if ((*inliers_2D_2D)[match_k][pt_k]) {
                    matches_inliers[match_k].x1.push_back(matches_calib[match_k].x1[pt_k]);
                    matches_inliers[match_k].x2.push_back(matches_calib[match_k].x2[pt_k]);
                }
            }
        }

        // TODO: a nicer way to scale the robust loss for the epipolar part
        refine_hybrid_pose(points2D_inliers, points3D_inliers, matches_inliers, map_ext, pose, bundle_opt,
                           bundle_opt.loss_scale * ransac_opt.max_epipolar_error / ransac_opt.max_reproj_error);
    }

    return stats;
}

RansacStats estimate_1D_radial_absolute_pose(const std::vector<Point2D> &points2D, const std::vector<Point3D> &points3D,
                                             const RansacOptions &ransac_opt, const BundleOptions &bundle_opt,
                                             CameraPose *pose, std::vector<char> *inliers) {
    if (points2D.size() < 5) {
        // Not possible to generate minimal sample
        return RansacStats();
    }

    // scale by the average norm (improves numerics in the bundle)
    double scale = 0.0;
    for (size_t k = 0; k < points2D.size(); ++k) {
        scale += points2D[k].norm();
    }
    scale = points2D.size() / scale;

    std::vector<Point2D> points2D_scaled = points2D;
    for (size_t k = 0; k < points2D_scaled.size(); ++k) {
        points2D_scaled[k] *= scale;
    }

    RansacOptions ransac_opt_scaled = ransac_opt;
    BundleOptions bundle_opt_scaled = bundle_opt;

    ransac_opt_scaled.max_reproj_error *= scale;
    bundle_opt_scaled.loss_scale *= scale;

    RansacStats stats = ransac_1D_radial_pnp(points2D_scaled, points3D, ransac_opt_scaled, pose, inliers);

    if (stats.num_inliers > 5) {
        // Collect inlier for additional bundle adjustment
        std::vector<Point2D> points2D_inliers;
        std::vector<Point3D> points3D_inliers;
        points2D_inliers.reserve(points2D.size());
        points3D_inliers.reserve(points3D.size());

        for (size_t k = 0; k < points2D.size(); ++k) {
            if (!(*inliers)[k])
                continue;
            points2D_inliers.push_back(points2D_scaled[k]);
            points3D_inliers.push_back(points3D[k]);
        }

        bundle_adjust_1D_radial(points2D_inliers, points3D_inliers, pose, bundle_opt_scaled);
    }

    return stats;
}

} // namespace poselib