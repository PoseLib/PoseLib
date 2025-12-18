#include "../../helpers.h"
#include "../../pybind11_extension.h"

#include <PoseLib/poselib.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace poselib {

static std::pair<CameraPose, py::dict> estimate_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
                                                               const std::vector<Eigen::Vector3d> &points3D,
                                                               const Camera &camera, const py::dict &ransac_opt_dict,
                                                               const py::dict &bundle_opt_dict,
                                                               const std::optional<CameraPose> &initial_pose) {
    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        ransac_opt.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_absolute_pose(points2D, points3D, camera, ransac_opt, bundle_opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);

    return std::make_pair(pose, output_dict);
}

static std::pair<CameraPose, py::dict> estimate_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
                                                               const std::vector<Eigen::Vector3d> &points3D,
                                                               const py::dict &camera_dict,
                                                               const py::dict &ransac_opt_dict,
                                                               const py::dict &bundle_opt_dict,
                                                               const std::optional<CameraPose> &initial_pose) {
    Camera camera = camera_from_dict(camera_dict);
    return estimate_absolute_pose_wrapper(points2D, points3D, camera, ransac_opt_dict, bundle_opt_dict, initial_pose);
}

static std::pair<CameraPose, py::dict> refine_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
                                                             const std::vector<Eigen::Vector3d> &points3D,
                                                             const CameraPose &initial_pose, const Camera &camera,
                                                             const py::dict &bundle_opt_dict) {

    // We normalize to improve numerics in the optimization
    const double scale = 1.0 / camera.focal();
    Camera norm_camera = camera;
    norm_camera.rescale(scale);

    std::vector<Eigen::Vector2d> points2D_scaled = points2D;
    for (size_t k = 0; k < points2D_scaled.size(); ++k) {
        points2D_scaled[k] *= scale;
    }

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);
    bundle_opt.loss_scale *= scale;

    CameraPose refined_pose = initial_pose;

    py::gil_scoped_release release;
    BundleStats stats = bundle_adjust(points2D_scaled, points3D, norm_camera, &refined_pose, bundle_opt);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_pose, output_dict);
}

static std::pair<CameraPose, py::dict> refine_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
                                                             const std::vector<Eigen::Vector3d> &points3D,
                                                             const CameraPose &initial_pose,
                                                             const py::dict &camera_dict,
                                                             const py::dict &bundle_opt_dict) {

    Camera camera = camera_from_dict(camera_dict);
    return refine_absolute_pose_wrapper(points2D, points3D, initial_pose, camera, bundle_opt_dict);
}

static std::pair<CameraPose, py::dict> estimate_absolute_pose_pnpl_wrapper(
    const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
    const std::vector<Eigen::Vector2d> &lines2D_1, const std::vector<Eigen::Vector2d> &lines2D_2,
    const std::vector<Eigen::Vector3d> &lines3D_1, const std::vector<Eigen::Vector3d> &lines3D_2, const Camera &camera,
    const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict, const std::optional<CameraPose> &initial_pose) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    py::gil_scoped_release release;

    std::vector<Line2D> lines2D;
    std::vector<Line3D> lines3D;
    lines2D.reserve(lines2D_1.size());
    lines3D.reserve(lines3D_1.size());
    for (size_t k = 0; k < lines2D_1.size(); ++k) {
        lines2D.emplace_back(lines2D_1[k], lines2D_2[k]);
        lines3D.emplace_back(lines3D_1[k], lines3D_2[k]);
    }

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        ransac_opt.score_initial_model = true;
    }
    std::vector<char> inlier_points_mask;
    std::vector<char> inlier_lines_mask;

    RansacStats stats = estimate_absolute_pose_pnpl(points2D, points3D, lines2D, lines3D, camera, ransac_opt,
                                                    bundle_opt, &pose, &inlier_points_mask, &inlier_lines_mask);

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_points_mask);
    output_dict["inliers_lines"] = convert_inlier_vector(inlier_lines_mask);
    return std::make_pair(pose, output_dict);
}

static std::pair<CameraPose, py::dict> estimate_absolute_pose_pnpl_wrapper(
    const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
    const std::vector<Eigen::Vector2d> &lines2D_1, const std::vector<Eigen::Vector2d> &lines2D_2,
    const std::vector<Eigen::Vector3d> &lines3D_1, const std::vector<Eigen::Vector3d> &lines3D_2,
    const py::dict &camera_dict, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict,
    const std::optional<CameraPose> &initial_pose) {

    Camera camera = camera_from_dict(camera_dict);
    return estimate_absolute_pose_pnpl_wrapper(points2D, points3D, lines2D_1, lines2D_2, lines3D_1, lines3D_2, camera,
                                               ransac_opt_dict, bundle_opt_dict, initial_pose);
}

static std::pair<CameraPose, py::dict> refine_absolute_pose_pnpl_wrapper(
    const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
    const std::vector<Eigen::Vector2d> &lines2D_1, const std::vector<Eigen::Vector2d> &lines2D_2,
    const std::vector<Eigen::Vector3d> &lines3D_1, const std::vector<Eigen::Vector3d> &lines3D_2,
    const CameraPose &initial_pose, const Camera &camera, const py::dict &bundle_opt_dict,
    const py::dict &line_bundle_opt_dict) {

    BundleOptions bundle_opt, line_bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);
    bundle_opt.loss_scale /= camera.focal();

    if (line_bundle_opt_dict.empty()) {
        line_bundle_opt = bundle_opt;
    } else {
        update_bundle_options(line_bundle_opt_dict, line_bundle_opt);
        line_bundle_opt.loss_scale /= camera.focal();
    }

    py::gil_scoped_release release;

    // Setup line objects
    std::vector<Line2D> lines2D;
    std::vector<Line3D> lines3D;
    lines2D.reserve(lines2D_1.size());
    lines3D.reserve(lines3D_1.size());
    for (size_t k = 0; k < lines2D_1.size(); ++k) {
        lines2D.emplace_back(lines2D_1[k], lines2D_2[k]);
        lines3D.emplace_back(lines3D_1[k], lines3D_2[k]);
    }

    // Calibrate points
    std::vector<Point2D> points2D_calib(points2D.size());
    for (size_t k = 0; k < points2D.size(); ++k) {
        camera.unproject(points2D[k], &points2D_calib[k]);
    }

    // Calibrate 2D line segments
    std::vector<Line2D> lines2D_calib(lines2D.size());
    for (size_t k = 0; k < lines2D.size(); ++k) {
        camera.unproject(lines2D[k].x1, &lines2D_calib[k].x1);
        camera.unproject(lines2D[k].x2, &lines2D_calib[k].x2);
    }

    CameraPose refined_pose = initial_pose;
    BundleStats stats =
        bundle_adjust(points2D_calib, points3D, lines2D_calib, lines3D, &refined_pose, bundle_opt, line_bundle_opt);

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_pose, output_dict);
}

static std::pair<CameraPose, py::dict> refine_absolute_pose_pnpl_wrapper(
    const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
    const std::vector<Eigen::Vector2d> &lines2D_1, const std::vector<Eigen::Vector2d> &lines2D_2,
    const std::vector<Eigen::Vector3d> &lines3D_1, const std::vector<Eigen::Vector3d> &lines3D_2,
    const CameraPose &initial_pose, const py::dict &camera_dict, const py::dict &bundle_opt_dict,
    const py::dict &line_bundle_opt_dict) {

    Camera camera = camera_from_dict(camera_dict);
    return refine_absolute_pose_pnpl_wrapper(points2D, points3D, lines2D_1, lines2D_2, lines3D_1, lines3D_2,
                                             initial_pose, camera, bundle_opt_dict, line_bundle_opt_dict);
}

static std::pair<CameraPose, py::dict> estimate_generalized_absolute_pose_wrapper(
    const std::vector<std::vector<Eigen::Vector2d>> &points2D,
    const std::vector<std::vector<Eigen::Vector3d>> &points3D, const std::vector<CameraPose> &camera_ext,
    const std::vector<Camera> &cameras, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict,
    const std::optional<CameraPose> &initial_pose) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        ransac_opt.score_initial_model = true;
    }
    std::vector<std::vector<char>> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_generalized_absolute_pose(points2D, points3D, camera_ext, cameras, ransac_opt,
                                                           bundle_opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vectors(inlier_mask);
    return std::make_pair(pose, output_dict);
}

static std::pair<CameraPose, py::dict> estimate_generalized_absolute_pose_wrapper(
    const std::vector<std::vector<Eigen::Vector2d>> &points2D,
    const std::vector<std::vector<Eigen::Vector3d>> &points3D, const std::vector<CameraPose> &camera_ext,
    const std::vector<py::dict> &camera_dicts, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict,
    const std::optional<CameraPose> &initial_pose) {

    std::vector<Camera> cameras;
    for (const py::dict &camera_dict : camera_dicts) {
        cameras.push_back(camera_from_dict(camera_dict));
    }

    return estimate_generalized_absolute_pose_wrapper(points2D, points3D, camera_ext, cameras, ransac_opt_dict,
                                                      bundle_opt_dict, initial_pose);
}

static std::pair<CameraPose, py::dict>
refine_generalized_absolute_pose_wrapper(const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                         const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                         const CameraPose &initial_pose, const std::vector<CameraPose> &camera_ext,
                                         const std::vector<Camera> &cameras, const py::dict &bundle_opt_dict) {

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose refined_pose = initial_pose;

    py::gil_scoped_release release;
    BundleStats stats = generalized_bundle_adjust(points2D, points3D, camera_ext, cameras, &refined_pose, bundle_opt);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_pose, output_dict);
}

static std::pair<CameraPose, py::dict>
refine_generalized_absolute_pose_wrapper(const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                         const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                         const CameraPose &initial_pose, const std::vector<CameraPose> &camera_ext,
                                         const std::vector<py::dict> &camera_dicts, const py::dict &bundle_opt_dict) {

    std::vector<Camera> cameras;
    for (const py::dict &camera_dict : camera_dicts) {
        cameras.push_back(camera_from_dict(camera_dict));
    }

    return refine_generalized_absolute_pose_wrapper(points2D, points3D, initial_pose, camera_ext, cameras,
                                                    bundle_opt_dict);
}

static std::pair<CameraPose, py::dict> estimate_1D_radial_absolute_pose_wrapper(
    const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
    const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict, const std::optional<CameraPose> &initial_pose) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        ransac_opt.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats =
        estimate_1D_radial_absolute_pose(points2D, points3D, ransac_opt, bundle_opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(pose, output_dict);
}

void register_absolute_pose(py::module &m) {
    // Robust estimators
    m.def("estimate_absolute_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const Camera &, const py::dict &, const py::dict &,
                            const std::optional<CameraPose> &>(&estimate_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("camera"), py::arg("ransac_opt") = py::dict(),
          py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
          "Absolute pose estimation with non-linear refinement.");
    m.def(
        "estimate_absolute_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &, const py::dict &,
                          const py::dict &, const py::dict &, const std::optional<CameraPose> &>(
            &estimate_absolute_pose_wrapper),
        py::arg("points2D"), py::arg("points3D"), py::arg("camera_dict"), py::arg("ransac_opt") = py::dict(),
        py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
        "Absolute pose estimation with non-linear refinement.");

    m.def("estimate_absolute_pose_pnpl",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const std::vector<Eigen::Vector3d> &, const std::vector<Eigen::Vector3d> &,
                            const Camera &, const py::dict &, const py::dict &,
                            const std::optional<CameraPose> &>(&estimate_absolute_pose_pnpl_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines3D_1"),
          py::arg("lines3D_2"), py::arg("camera"), py::arg("ransac_opt") = py::dict(),
          py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
          "Absolute pose estimation with non-linear refinement from points and lines.");
    m.def(
        "estimate_absolute_pose_pnpl",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                          const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                          const std::vector<Eigen::Vector3d> &, const std::vector<Eigen::Vector3d> &, const py::dict &,
                          const py::dict &, const py::dict &, const std::optional<CameraPose> &>(
            &estimate_absolute_pose_pnpl_wrapper),
        py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines3D_1"),
        py::arg("lines3D_2"), py::arg("camera_dict"), py::arg("ransac_opt") = py::dict(),
        py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
        "Absolute pose estimation with non-linear refinement from points and lines.");

    m.def("estimate_generalized_absolute_pose",
          py::overload_cast<const std::vector<std::vector<Eigen::Vector2d>> &,
                            const std::vector<std::vector<Eigen::Vector3d>> &, const std::vector<CameraPose> &,
                            const std::vector<Camera> &, const py::dict &, const py::dict &,
                            const std::optional<CameraPose> &>(
              &estimate_generalized_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("camera_ext"), py::arg("cameras"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
          "Generalized absolute pose estimation with non-linear refinement.");
    m.def("estimate_generalized_absolute_pose",
          py::overload_cast<const std::vector<std::vector<Eigen::Vector2d>> &,
                            const std::vector<std::vector<Eigen::Vector3d>> &, const std::vector<CameraPose> &,
                            const std::vector<py::dict> &, const py::dict &, const py::dict &,
                            const std::optional<CameraPose> &>(
              &estimate_generalized_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("camera_ext"), py::arg("camera_dicts"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), py::arg("initial_pose") = py::none(),
          "Generalized absolute pose estimation with non-linear refinement.");

    m.def("estimate_1D_radial_absolute_pose", &estimate_1D_radial_absolute_pose_wrapper, py::arg("points2D"),
          py::arg("points3D"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          py::arg("initial_pose") = py::none(),
          "Absolute pose estimation for the 1D radial camera model with non-linear refinement.");

    // Stand-alone non-linear refinement
    m.def("refine_absolute_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const CameraPose &, const Camera &, const py::dict &>(
              &refine_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("camera"),
          py::arg("bundle_options") = py::dict(), "Absolute pose non-linear refinement.");
    m.def("refine_absolute_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const CameraPose &, const py::dict &, const py::dict &>(
              &refine_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("camera_dict"),
          py::arg("bundle_options") = py::dict(), "Absolute pose non-linear refinement.");

    m.def("refine_absolute_pose_pnpl",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const std::vector<Eigen::Vector3d> &, const std::vector<Eigen::Vector3d> &,
                            const CameraPose &, const Camera &, const py::dict &, const py::dict &>(
              &refine_absolute_pose_pnpl_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines3D_1"),
          py::arg("lines3D_2"), py::arg("initial_pose"), py::arg("camera"), py::arg("bundle_opt") = py::dict(),
          py::arg("line_bundle_opt") = py::dict(), "Absolute pose non-linear refinement from points and lines.");
    m.def("refine_absolute_pose_pnpl",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const std::vector<Eigen::Vector3d> &, const std::vector<Eigen::Vector3d> &,
                            const CameraPose &, const py::dict &, const py::dict &, const py::dict &>(
              &refine_absolute_pose_pnpl_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines3D_1"),
          py::arg("lines3D_2"), py::arg("initial_pose"), py::arg("camera_dict"), py::arg("bundle_opt") = py::dict(),
          py::arg("line_bundle_opt") = py::dict(), "Absolute pose non-linear refinement from points and lines.");

    m.def("refine_generalized_absolute_pose",
          py::overload_cast<const std::vector<std::vector<Eigen::Vector2d>> &,
                            const std::vector<std::vector<Eigen::Vector3d>> &, const CameraPose &,
                            const std::vector<CameraPose> &, const std::vector<Camera> &,
                            const py::dict &>(&refine_generalized_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("camera_ext"), py::arg("cameras"),
          py::arg("bundle_opt") = py::dict(), "Generalized absolute pose non-linear refinement.");
    m.def("refine_generalized_absolute_pose",
          py::overload_cast<const std::vector<std::vector<Eigen::Vector2d>> &,
                            const std::vector<std::vector<Eigen::Vector3d>> &, const CameraPose &,
                            const std::vector<CameraPose> &, const std::vector<py::dict> &, const py::dict &>(
              &refine_generalized_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("camera_ext"),
          py::arg("camera_dicts"), py::arg("bundle_opt") = py::dict(),
          "Generalized absolute pose non-linear refinement.");
}

} // namespace poselib
