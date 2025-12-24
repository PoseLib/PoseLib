#include "../../helpers.h"
#include "../../pybind11_extension.h"

#include <PoseLib/poselib.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace poselib {
namespace {

std::pair<CameraPose, py::dict> estimate_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                               const std::vector<Eigen::Vector2d> &points2D_2,
                                                               const Camera &camera1, const Camera &camera2,
                                                               const py::dict &ransac_opt_dict,
                                                               const py::dict &bundle_opt_dict,
                                                               const std::optional<CameraPose> &initial_pose) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        ransac_opt.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats =
        estimate_relative_pose(points2D_1, points2D_2, camera1, camera2, ransac_opt, bundle_opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(pose, output_dict);
}

std::pair<CameraPose, py::dict>
estimate_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                               const std::vector<Eigen::Vector2d> &points2D_2, const py::dict &camera1_dict,
                               const py::dict &camera2_dict, const py::dict &ransac_opt_dict,
                               const py::dict &bundle_opt_dict, const std::optional<CameraPose> &initial_pose) {
    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    return estimate_relative_pose_wrapper(points2D_1, points2D_2, camera1, camera2, ransac_opt_dict, bundle_opt_dict,
                                          initial_pose);
}

std::pair<MonoDepthTwoViewGeometry, py::dict> estimate_monodepth_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2, const Camera &camera1,
    const Camera &camera2, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict,
    const std::optional<MonoDepthTwoViewGeometry> &initial_pose) {
    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    MonoDepthTwoViewGeometry monodepth_geometry;
    if (initial_pose.has_value()) {
        monodepth_geometry = initial_pose.value();
        ransac_opt.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_monodepth_relative_pose(points2D_1, points2D_2, depth_1, depth_2, camera1, camera2,
                                                         ransac_opt, bundle_opt, &monodepth_geometry, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(monodepth_geometry, output_dict);
}

std::pair<MonoDepthTwoViewGeometry, py::dict> estimate_monodepth_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2, const py::dict &camera1_dict,
    const py::dict &camera2_dict, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict,
    const std::optional<MonoDepthTwoViewGeometry> &initial_pose) {
    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    return estimate_monodepth_relative_pose_wrapper(points2D_1, points2D_2, depth_1, depth_2, camera1, camera2,
                                                    ransac_opt_dict, bundle_opt_dict, initial_pose);
}

std::pair<ImagePair, py::dict>
estimate_shared_focal_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                            const std::vector<Eigen::Vector2d> &points2D_2, const Eigen::Vector2d &pp,
                                            const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict,
                                            const std::optional<ImagePair> &initial_image_pair) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    ImagePair image_pair;
    if (initial_image_pair.has_value()) {
        image_pair = initial_image_pair.value();
        ransac_opt.score_initial_model = true;
    }

    std::vector<char> inlier_mask;

    std::vector<Image> output;

    py::gil_scoped_release release;
    RansacStats stats = estimate_shared_focal_relative_pose(points2D_1, points2D_2, pp, ransac_opt, bundle_opt,
                                                            &image_pair, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(image_pair, output_dict);
}

std::pair<MonoDepthImagePair, py::dict> estimate_monodepth_shared_focal_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2, const py::dict &ransac_opt_dict,
    const py::dict &bundle_opt_dict, const std::optional<MonoDepthImagePair> &initial_image_pair) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    if (ransac_opt.max_epipolar_error > 0.0)
        bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    else
        bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;

    update_bundle_options(bundle_opt_dict, bundle_opt);

    MonoDepthImagePair image_pair;
    if (initial_image_pair.has_value()) {
        image_pair = initial_image_pair.value();
        ransac_opt.score_initial_model = true;
    }

    std::vector<char> inlier_mask;

    std::vector<Image> output;

    py::gil_scoped_release release;
    RansacStats stats = estimate_shared_focal_monodepth_relative_pose(
        points2D_1, points2D_2, depth_1, depth_2, ransac_opt, bundle_opt, &image_pair, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(image_pair, output_dict);
}

std::pair<MonoDepthImagePair, py::dict> estimate_monodepth_varying_focal_relative_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D_1, const std::vector<Eigen::Vector2d> &points2D_2,
    const std::vector<double> &depth_1, const std::vector<double> &depth_2, const py::dict &ransac_opt_dict,
    const py::dict &bundle_opt_dict, const std::optional<MonoDepthImagePair> &initial_image_pair) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    if (ransac_opt.max_epipolar_error > 0.0)
        bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    else
        bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;

    update_bundle_options(bundle_opt_dict, bundle_opt);

    MonoDepthImagePair image_pair;
    if (initial_image_pair.has_value()) {
        image_pair = initial_image_pair.value();
        ransac_opt.score_initial_model = true;
    }

    std::vector<char> inlier_mask;

    std::vector<Image> output;

    py::gil_scoped_release release;
    RansacStats stats = estimate_varying_focal_monodepth_relative_pose(
        points2D_1, points2D_2, depth_1, depth_2, ransac_opt, bundle_opt, &image_pair, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(image_pair, output_dict);
}

std::pair<CameraPose, py::dict> refine_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                             const std::vector<Eigen::Vector2d> &points2D_2,
                                                             const CameraPose &initial_pose, const Camera &camera1,
                                                             const Camera &camera2, const py::dict &bundle_opt_dict) {

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    py::gil_scoped_release release;

    // Normalize image points
    std::vector<Eigen::Vector2d> x1_calib = points2D_1;
    std::vector<Eigen::Vector2d> x2_calib = points2D_2;

    for (size_t i = 0; i < x1_calib.size(); ++i) {
        camera1.unproject(points2D_1[i], &x1_calib[i]);
        camera2.unproject(points2D_2[i], &x2_calib[i]);
    }
    bundle_opt.loss_scale *= (1.0 / camera1.focal() + 1.0 / camera2.focal()) * 0.5;

    CameraPose refined_pose = initial_pose;
    BundleStats stats = refine_relpose(x1_calib, x2_calib, &refined_pose, bundle_opt);

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_pose, output_dict);
}

std::pair<CameraPose, py::dict> refine_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                             const std::vector<Eigen::Vector2d> &points2D_2,
                                                             const CameraPose &initial_pose,
                                                             const py::dict &camera1_dict, const py::dict &camera2_dict,
                                                             const py::dict &bundle_opt_dict) {

    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);
    return refine_relative_pose_wrapper(points2D_1, points2D_2, initial_pose, camera1, camera2, bundle_opt_dict);
}

std::pair<Eigen::Matrix3d, py::dict> estimate_fundamental_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                                  const std::vector<Eigen::Vector2d> &points2D_2,
                                                                  const py::dict &ransac_opt_dict,
                                                                  const py::dict &bundle_opt_dict,
                                                                  const std::optional<Eigen::Matrix3d> &initial_F) {
    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    Eigen::Matrix3d F;
    if (initial_F.has_value()) {
        F = initial_F.value();
        ransac_opt.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_fundamental(points2D_1, points2D_2, ransac_opt, bundle_opt, &F, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(F, output_dict);
}

std::pair<Eigen::Matrix3d, py::dict> refine_fundamental_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                                const std::vector<Eigen::Vector2d> &points2D_2,
                                                                const Eigen::Matrix3d &initial_F,
                                                                const py::dict &bundle_opt_dict) {

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    py::gil_scoped_release release;

    // Normalize image points
    std::vector<Eigen::Vector2d> x1_norm = points2D_1;
    std::vector<Eigen::Vector2d> x2_norm = points2D_2;

    Eigen::Matrix3d T1, T2;
    double scale = normalize_points(x1_norm, x2_norm, T1, T2, true, true, true);
    BundleOptions bundle_opt_scaled = bundle_opt;
    bundle_opt_scaled.loss_scale /= scale;

    Eigen::Matrix3d refined_F = T2.transpose().inverse() * initial_F * T1.inverse();
    BundleStats stats = refine_fundamental(x1_norm, x2_norm, &refined_F, bundle_opt_scaled);

    refined_F = T2.transpose() * refined_F * T1;
    refined_F /= refined_F.norm();

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_F, output_dict);
}

std::pair<CameraPose, py::dict> estimate_generalized_relative_pose_wrapper(
    const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
    const std::vector<Camera> &cameras1, const std::vector<CameraPose> &camera2_ext,
    const std::vector<Camera> &cameras2, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict,
    const std::optional<CameraPose> &initial_pose) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        ransac_opt.score_initial_model = true;
    }
    std::vector<std::vector<char>> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_generalized_relative_pose(matches, camera1_ext, cameras1, camera2_ext, cameras2,
                                                           ransac_opt, bundle_opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vectors(inlier_mask);
    return std::make_pair(pose, output_dict);
}

std::pair<CameraPose, py::dict> estimate_generalized_relative_pose_wrapper(
    const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
    const std::vector<py::dict> &cameras1_dict, const std::vector<CameraPose> &camera2_ext,
    const std::vector<py::dict> &cameras2_dict, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict,
    const std::optional<CameraPose> &initial_pose) {

    std::vector<Camera> cameras1, cameras2;
    for (const py::dict &camera_dict : cameras1_dict) {
        cameras1.push_back(camera_from_dict(camera_dict));
    }
    for (const py::dict &camera_dict : cameras2_dict) {
        cameras2.push_back(camera_from_dict(camera_dict));
    }

    return estimate_generalized_relative_pose_wrapper(matches, camera1_ext, cameras1, camera2_ext, cameras2,
                                                      ransac_opt_dict, bundle_opt_dict, initial_pose);
}

std::pair<CameraPose, py::dict> refine_generalized_relative_pose_wrapper(
    const std::vector<PairwiseMatches> &matches, const CameraPose &initial_pose,
    const std::vector<CameraPose> &camera1_ext, const std::vector<Camera> &cameras1,
    const std::vector<CameraPose> &camera2_ext, const std::vector<Camera> &cameras2, const py::dict &bundle_opt_dict) {

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    py::gil_scoped_release release;

    // Compute normalized matches
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

    bundle_opt.loss_scale *= scaling_factor;

    CameraPose refined_pose = initial_pose;
    BundleStats stats = refine_generalized_relpose(calib_matches, camera1_ext, camera2_ext, &refined_pose, bundle_opt);

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_pose, output_dict);
}

std::pair<CameraPose, py::dict> refine_generalized_relative_pose_wrapper(const std::vector<PairwiseMatches> &matches,
                                                                         const CameraPose &initial_pose,
                                                                         const std::vector<CameraPose> &camera1_ext,
                                                                         const std::vector<py::dict> &cameras1_dict,
                                                                         const std::vector<CameraPose> &camera2_ext,
                                                                         const std::vector<py::dict> &cameras2_dict,
                                                                         const py::dict &bundle_opt_dict) {

    std::vector<Camera> cameras1, cameras2;
    for (const py::dict &camera_dict : cameras1_dict) {
        cameras1.push_back(camera_from_dict(camera_dict));
    }
    for (const py::dict &camera_dict : cameras2_dict) {
        cameras2.push_back(camera_from_dict(camera_dict));
    }

    return refine_generalized_relative_pose_wrapper(matches, initial_pose, camera1_ext, cameras1, camera2_ext, cameras2,
                                                    bundle_opt_dict);
}

std::pair<std::vector<CameraPose>, std::vector<Point3D>> motion_from_homography_wrapper(Eigen::Matrix3d &H) {
    std::vector<CameraPose> poses;
    std::vector<Point3D> normals;
    motion_from_homography(H, &poses, &normals);
    return std::make_pair(poses, normals);
}

std::tuple<Camera, Camera, int> focals_from_fundamental_iterative_wrapper(const Eigen::Matrix3d F,
                                                                          const py::dict &camera1_dict,
                                                                          const py::dict &camera2_dict,
                                                                          const int max_iters,
                                                                          const Eigen::Vector4d &weights) {

    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    py::gil_scoped_release release;
    return focals_from_fundamental_iterative(F, camera1, camera2, max_iters, weights);
}

} // namespace

void register_relative_pose(py::module &m) {
    m.def("estimate_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &, const Camera &,
                            const Camera &, const py::dict &, const py::dict &, const std::optional<CameraPose> &>(
              &estimate_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("camera1"), py::arg("camera2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
          "Relative pose estimation with non-linear refinement.");

    m.def(
        "estimate_relative_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &, const py::dict &,
                          const py::dict &, const py::dict &, const py::dict &, const std::optional<CameraPose> &>(
            &estimate_relative_pose_wrapper),
        py::arg("points2D_1"), py::arg("points2D_2"), py::arg("camera1_dict"), py::arg("camera2_dict"),
        py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
        "Relative pose estimation with non-linear refinement.");

    m.def("estimate_monodepth_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const std::vector<double> &, const std::vector<double> &, const Camera &, const Camera &,
                            const py::dict &, const py::dict &, const std::optional<MonoDepthTwoViewGeometry> &>(
              &estimate_monodepth_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"), py::arg("camera1"),
          py::arg("camera2"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          py::arg("initial_pose") = py::none(), "Pose estimation using depth estimates with non-linear refinement.");

    m.def(
        "estimate_monodepth_relative_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                          const std::vector<double> &, const std::vector<double> &, const py::dict &, const py::dict &,
                          const py::dict &, const py::dict &, const std::optional<MonoDepthTwoViewGeometry> &>(
            &estimate_monodepth_relative_pose_wrapper),
        py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"), py::arg("camera1_dict"),
        py::arg("camera2_dict"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
        py::arg("initial_pose") = py::none(),
        "Relative pose estimation using depth estimates with non-linear refinement.");

    m.def("estimate_shared_focal_relative_pose", &estimate_shared_focal_relative_pose_wrapper, py::arg("points2D_1"),
          py::arg("points2D_2"), py::arg("pp") = Eigen::Vector2d::Zero(), py::arg("ransac_opt") = py::dict(),
          py::arg("bundle_opt") = py::dict(), py::arg("initial_image_pair") = py::none(),
          "Relative pose estimation with unknown equal focal lengths with non-linear refinement.");

    m.def("estimate_monodepth_shared_focal_relative_pose", &estimate_monodepth_shared_focal_relative_pose_wrapper,
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          py::arg("initial_image_pair") = py::none(),
          "Relative pose estimation with depth estimates and unknown equal focal lengths with non-linear refinement.");

    m.def("estimate_monodepth_varying_focal_relative_pose", &estimate_monodepth_varying_focal_relative_pose_wrapper,
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("depth_1"), py::arg("depth_2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          py::arg("initial_image_pair") = py::none(),
          "Relative pose estimation with depth estimates and unknown different focal lengths with non-linear "
          "refinement.");

    m.def("estimate_fundamental", &estimate_fundamental_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), py::arg("initial_F") = py::none(),
          "Fundamental matrix estimation with non-linear refinement. Note: if you have known intrinsics you should use "
          "estimate_relative_pose instead!");

    m.def("estimate_generalized_relative_pose",
          py::overload_cast<const std::vector<PairwiseMatches> &, const std::vector<CameraPose> &,
                            const std::vector<Camera> &, const std::vector<CameraPose> &, const std::vector<Camera> &,
                            const py::dict &, const py::dict &, const std::optional<CameraPose> &>(
              &estimate_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("camera1_ext"), py::arg("cameras1"), py::arg("camera2_ext"), py::arg("cameras2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
          "Generalized relative pose estimation with non-linear refinement.");
    m.def(
        "estimate_generalized_relative_pose",
        py::overload_cast<const std::vector<PairwiseMatches> &, const std::vector<CameraPose> &,
                          const std::vector<py::dict> &, const std::vector<CameraPose> &, const std::vector<py::dict> &,
                          const py::dict &, const py::dict &, const std::optional<CameraPose> &>(
            &estimate_generalized_relative_pose_wrapper),
        py::arg("matches"), py::arg("camera1_ext"), py::arg("camera1_dict"), py::arg("camera2_ext"),
        py::arg("camera2_dict"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
        py::arg("initial_pose") = py::none(), "Generalized relative pose estimation with non-linear refinement.");

    m.def("motion_from_homography", &motion_from_homography_wrapper, py::arg("H"));
    m.def("focals_from_fundamental", &focals_from_fundamental, py::arg("F"), py::arg("pp1"), py::arg("pp2"));
    m.def("focals_from_fundamental_iterative", &focals_from_fundamental_iterative, py::arg("F"), py::arg("camera1"),
          py::arg("camera2"), py::arg("max_iters") = 50,
          py::arg("weights") = Eigen::Vector4d(5.0e-4, 1.0, 5.0e-4, 1.0));
    m.def("focals_from_fundamental_iterative", &focals_from_fundamental_iterative_wrapper, py::arg("F"),
          py::arg("camera1_dict"), py::arg("camera2_dict"), py::arg("max_iters") = 50,
          py::arg("weights") = Eigen::Vector4d(5.0e-4, 1.0, 5.0e-4, 1.0));

    // Stand-alone non-linear refinement
    m.def("refine_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const CameraPose &, const Camera &, const Camera &, const py::dict &>(
              &refine_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("initial_pose"), py::arg("camera1"), py::arg("camera2"),
          py::arg("bundle_options") = py::dict(), "Relative pose non-linear refinement.");
    m.def("refine_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const CameraPose &, const py::dict &, const py::dict &, const py::dict &>(
              &refine_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("initial_pose"), py::arg("camera1_dict"),
          py::arg("camera2_dict"), py::arg("bundle_options") = py::dict(), "Relative pose non-linear refinement.");

    m.def("refine_fundamental", &refine_fundamental_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("initial_F"), py::arg("bundle_options") = py::dict(), "Fundamental matrix non-linear refinement.");

    m.def("refine_generalized_relative_pose",
          py::overload_cast<const std::vector<PairwiseMatches> &, const CameraPose &, const std::vector<CameraPose> &,
                            const std::vector<Camera> &, const std::vector<CameraPose> &, const std::vector<Camera> &,
                            const py::dict &>(&refine_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("initial_pose"), py::arg("camera1_ext"), py::arg("cameras1"),
          py::arg("camera2_ext"), py::arg("cameras2"), py::arg("bundle_opt") = py::dict(),
          "Generalized relative pose non-linear refinement.");
    m.def("refine_generalized_relative_pose",
          py::overload_cast<const std::vector<PairwiseMatches> &, const CameraPose &, const std::vector<CameraPose> &,
                            const std::vector<py::dict> &, const std::vector<CameraPose> &,
                            const std::vector<py::dict> &, const py::dict &>(&refine_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("initial_pose"), py::arg("camera1_ext"), py::arg("camera1_dict"),
          py::arg("camera2_ext"), py::arg("camera2_dict"), py::arg("bundle_opt") = py::dict(),
          "Generalized relative pose non-linear refinement.");
}

} // namespace poselib
