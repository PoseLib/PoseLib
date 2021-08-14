#include "robust.h"

namespace pose_lib {

int estimate_absolute_pose(const std::vector<Eigen::Vector2d> &points2D,
                           const std::vector<Eigen::Vector3d> &points3D,
                           const Camera &camera, const RansacOptions &ransac_opt,
                           const BundleOptions &bundle_opt,
                           CameraPose *pose, std::vector<char> *inliers) {

    std::vector<Eigen::Vector2d> points2D_calib(points2D.size());
    for (size_t k = 0; k < points2D.size(); ++k) {
        camera.unproject(points2D[k], &points2D_calib[k]);
    }

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_reproj_error /= camera.focal();

    int num_inl = ransac_pose(points2D_calib, points3D, ransac_opt_scaled, pose, inliers);

    if (num_inl > 3) {
        // Collect inlier for additional bundle adjustment
        std::vector<Eigen::Vector2d> points2D_inliers;
        std::vector<Eigen::Vector3d> points3D_inliers;
        points2D_inliers.reserve(points2D.size());
        points3D_inliers.reserve(points3D.size());

        for (size_t k = 0; k < points2D.size(); ++k) {
            if (!(*inliers)[k])
                continue;
            points2D_inliers.push_back(points2D[k]);
            points3D_inliers.push_back(points3D[k]);
        }

        bundle_adjust(points2D_inliers, points3D_inliers, camera, pose, bundle_opt);
    }

    return num_inl;
}

int estimate_generalized_absolute_pose(
    const std::vector<std::vector<Eigen::Vector2d>> &points2D,
    const std::vector<std::vector<Eigen::Vector3d>> &points3D,
    const std::vector<CameraPose> &camera_ext,
    const std::vector<Camera> &cameras,
    const RansacOptions &ransac_opt,
    const BundleOptions &bundle_opt,
    CameraPose *pose, std::vector<std::vector<char>> *inliers) {

    const size_t num_cams = cameras.size();

    // Normalize image points for the RANSAC
    std::vector<std::vector<Eigen::Vector2d>> points2D_calib;
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

    int num_inl = ransac_gen_pose(points2D_calib, points3D, camera_ext, ransac_opt_scaled, pose, inliers);

    if (num_inl > 3) {
        // Collect inlier for additional bundle adjustment
        std::vector<std::vector<Eigen::Vector2d>> points2D_inliers;
        std::vector<std::vector<Eigen::Vector3d>> points3D_inliers;
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

    return num_inl;
}

int estimate_relative_pose(
    const std::vector<Eigen::Vector2d> &points2D_1,
    const std::vector<Eigen::Vector2d> &points2D_2,
    const Camera &camera1, const Camera &camera2,
    const RansacOptions &ransac_opt, const BundleOptions &bundle_opt,
    CameraPose *pose, std::vector<char> *inliers) {

    const size_t num_pts = points2D_1.size();

    std::vector<Eigen::Vector2d> x1_calib(num_pts);
    std::vector<Eigen::Vector2d> x2_calib(num_pts);
    for (size_t k = 0; k < num_pts; ++k) {
        camera1.unproject(points2D_1[k], &x1_calib[k]);
        camera2.unproject(points2D_2[k], &x2_calib[k]);
    }

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_reproj_error /= 0.5 * (camera1.focal() + camera2.focal());

    int num_inl = ransac_relpose(x1_calib, x2_calib, ransac_opt_scaled, pose, inliers);

    if (num_inl > 5) {
        // Collect inlier for additional bundle adjustment
        // TODO: use camera models for this refinement!
        std::vector<Eigen::Vector2d> x1_inliers;
        std::vector<Eigen::Vector2d> x2_inliers;
        x1_inliers.reserve(num_pts);
        x2_inliers.reserve(num_pts);

        for (size_t k = 0; k < num_pts; ++k) {
            if (!(*inliers)[k])
                continue;
            x1_inliers.push_back(x1_calib[k]);
            x2_inliers.push_back(x2_calib[k]);
        }

        refine_sampson(x1_inliers, x2_inliers, pose, bundle_opt);
    }

    return num_inl;
}

} // namespace pose_lib