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

std::pair<Eigen::Matrix3d, py::dict> estimate_homography_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                                 const std::vector<Eigen::Vector2d> &points2D_2,
                                                                 const py::dict &opt_dict,
                                                                 const std::optional<Eigen::Matrix3d> &initial_H) {

    HomographyOptions opt;
    update_homography_options(opt_dict, opt);

    Eigen::Matrix3d H;
    if (initial_H.has_value()) {
        H = initial_H.value();
        opt.ransac.score_initial_model = true;
    }
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_homography(points2D_1, points2D_2, opt, &H, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(H, output_dict);
}

std::pair<Eigen::Matrix3d, py::dict> refine_homography_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                               const std::vector<Eigen::Vector2d> &points2D_2,
                                                               const Eigen::Matrix3d initial_H,
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

    Eigen::Matrix3d refined_H = T2 * initial_H * T1.inverse();
    BundleStats stats = refine_homography(x1_norm, x2_norm, &refined_H, bundle_opt_scaled);

    refined_H = T2.inverse() * refined_H * T1;
    refined_H /= refined_H.norm();

    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(refined_H, output_dict);
}

} // namespace

void register_homography(py::module &m) {
    m.def("estimate_homography", &estimate_homography_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("opt") = py::dict(), py::arg("initial_H") = py::none(),
          "Homography matrix estimation with non-linear refinement.");

    m.def("refine_homography", &refine_homography_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("initial_H"), py::arg("bundle_options") = py::dict(), "Homography non-linear refinement.");
}

} // namespace poselib
