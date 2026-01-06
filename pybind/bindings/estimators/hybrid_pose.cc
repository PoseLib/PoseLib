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

std::pair<CameraPose, py::dict>
estimate_hybrid_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                             const std::vector<PairwiseMatches> &matches_2D_2D, const Camera &camera,
                             const std::vector<CameraPose> &map_ext, const std::vector<Camera> &map_cameras,
                             const py::dict &opt_dict,
                             const std::optional<CameraPose> &initial_pose) {

    HybridPoseOptions opt;
    update_hybrid_pose_options(opt_dict, opt);

    CameraPose pose;
    if (initial_pose.has_value()) {
        pose = initial_pose.value();
        opt.ransac.score_initial_model = true;
    }
    std::vector<char> inliers_mask_2d3d;
    std::vector<std::vector<char>> inliers_mask_2d2d;

    py::gil_scoped_release release;
    RansacStats stats = estimate_hybrid_pose(points2D, points3D, matches_2D_2D, camera, map_ext, map_cameras,
                                             opt, &pose, &inliers_mask_2d3d, &inliers_mask_2d2d);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inliers_mask_2d3d);
    output_dict["inliers_2D"] = convert_inlier_vectors(inliers_mask_2d2d);
    return std::make_pair(pose, output_dict);
}

std::pair<CameraPose, py::dict>
estimate_hybrid_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                             const std::vector<PairwiseMatches> &matches_2D_2D, const py::dict &camera_dict,
                             const std::vector<CameraPose> &map_ext, const std::vector<py::dict> &map_camera_dicts,
                             const py::dict &opt_dict,
                             const std::optional<CameraPose> &initial_pose) {

    Camera camera = camera_from_dict(camera_dict);
    std::vector<Camera> map_cameras;
    for (const py::dict &camera_dict : map_camera_dicts) {
        map_cameras.push_back(camera_from_dict(camera_dict));
    }

    return estimate_hybrid_pose_wrapper(points2D, points3D, matches_2D_2D, camera, map_ext, map_cameras,
                                        opt_dict, initial_pose);
}

} // namespace

void register_hybrid_pose(py::module &m) {
    m.def(
        "estimate_hybrid_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                          const std::vector<PairwiseMatches> &, const Camera &, const std::vector<CameraPose> &,
                          const std::vector<Camera> &, const py::dict &,
                          const std::optional<CameraPose> &>(&estimate_hybrid_pose_wrapper),
        py::arg("points2D"), py::arg("points3D"), py::arg("matches_2D_2D"), py::arg("camera"), py::arg("map_ext"),
        py::arg("map_cameras"), py::arg("opt") = py::dict(),
        py::arg("initial_pose") = py::none(),
        "Hybrid camera pose estimation (both 2D-3D and 2D-2D correspondences to the map) with non-linear refinement.");
    m.def(
        "estimate_hybrid_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                          const std::vector<PairwiseMatches> &, const py::dict &, const std::vector<CameraPose> &,
                          const std::vector<py::dict> &, const py::dict &,
                          const std::optional<CameraPose> &>(&estimate_hybrid_pose_wrapper),
        py::arg("points2D"), py::arg("points3D"), py::arg("matches_2D_2D"), py::arg("camera_dict"), py::arg("map_ext"),
        py::arg("map_camera_dicts"), py::arg("opt") = py::dict(),
        py::arg("initial_pose") = py::none(),
        "Hybrid camera pose estimation (both 2D-3D and 2D-2D correspondences to the map) with non-linear refinement.");
}

} // namespace poselib
