#include "robust.h"


namespace pose_lib {


int estimate_absolute_pose(const std::vector<Eigen::Vector2d>& points2D,
                           const std::vector<Eigen::Vector3d>& points3D,
                           const Camera &camera, const RansacOptions &ransac_opt,
                           const BundleOptions &bundle_opt,
                           CameraPose *pose, std::vector<char> *inliers) {
    
    std::vector<Eigen::Vector2d> points2D_calib(points2D.size());
    for(size_t k = 0; k < points2D.size(); ++k) {
        camera.unproject(points2D[k], &points2D_calib[k]);
    }

    RansacOptions ransac_opt_scaled = ransac_opt;
    ransac_opt_scaled.max_reproj_error /= camera.focal();    

    int num_inl = ransac_pose(points2D_calib, points3D, ransac_opt_scaled, pose, inliers);

    if(num_inl > 3) {
        std::vector<Eigen::Vector2d> points2D_inliers;
        std::vector<Eigen::Vector3d> points3D_inliers;
        points2D_inliers.reserve(points2D.size());
        points3D_inliers.reserve(points3D.size());

        for(size_t k = 0; k < points2D.size(); ++k) {
            if(!(*inliers)[k])
                continue;
            points2D_inliers.push_back(points2D[k]);
            points3D_inliers.push_back(points3D[k]);
        }

        bundle_adjust(points2D_inliers, points3D_inliers, camera, pose, bundle_opt);
    }

    return num_inl;
}




}