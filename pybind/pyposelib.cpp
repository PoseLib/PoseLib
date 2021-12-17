#include <iostream>
#include <tuple>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>

#include <PoseLib/version.h>
#include <PoseLib/poselib.h>
#include "helpers.h"


namespace py = pybind11;

namespace pose_lib {


py::dict RansacOptions_wrapper(py::dict overwrite) {
    RansacOptions opt;
    update_ransac_options(overwrite, opt);
    py::dict result;
    write_to_dict(opt, result);
    return result;
}

py::dict BundleOptions_wrapper(py::dict overwrite) {
    BundleOptions opt;
    update_bundle_options(overwrite, opt);
    py::dict result;
    write_to_dict(opt, result);
    return result;
}


std::vector<CameraPose> p3p_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X){
    std::vector<CameraPose> output;
    p3p(x, X, &output);
    return output;
}

std::vector<CameraPose> gp3p_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X){
    std::vector<CameraPose> output;
    gp3p(p, x, X, &output);
    return output;
}

std::vector<CameraPose> gp4ps_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, bool filter_solutions=true){
    std::vector<CameraPose> output;
    gp4ps(p, x, X, &output, filter_solutions);
    return output;
}

std::vector<CameraPose> gp4ps_kukelova_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, bool filter_solutions=true){
    std::vector<CameraPose> output;
    gp4ps_kukelova(p, x, X, &output, filter_solutions);
    return output;
}

std::vector<CameraPose> gp4ps_camposeco_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X){
    std::vector<CameraPose> output;
    gp4ps_camposeco(p, x, X, &output);
    return output;
}

std::vector<CameraPose> p4pf_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, bool filter_solutions = true){
    std::vector<CameraPose> output;
    p4pf(x, X, &output, filter_solutions);
    return output;
}

std::vector<CameraPose> p2p2pl_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
           const std::vector<Eigen::Vector3d> &V){
    std::vector<CameraPose> output;
    p2p2pl(xp, Xp, x, X, V, &output);
    return output;
}

std::vector<CameraPose> p6lp_wrapper(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X){
    std::vector<CameraPose> output;
    p6lp(l, X, &output);
    return output;
}

std::vector<CameraPose> p5lp_radial_wrapper(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X){
    std::vector<CameraPose> output;
    p5lp_radial(l, X, &output);
    return output;
}

std::vector<CameraPose> p2p1ll_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
           const std::vector<Eigen::Vector3d> &V){
    std::vector<CameraPose> output;
    p2p1ll(xp, Xp, l, X, V, &output);
    return output;
}

std::vector<CameraPose> p1p2ll_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
           const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
           const std::vector<Eigen::Vector3d> &V){
    std::vector<CameraPose> output;
    p1p2ll(xp, Xp, l, X, V, &output);
    return output;
}

std::vector<CameraPose> p3ll_wrapper(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector3d> &V){
    std::vector<CameraPose> output;
    p3ll(l, X, V, &output);
    return output;
}

std::vector<CameraPose> up2p_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X){
    std::vector<CameraPose> output;
    up2p(x, X, &output);
    return output;
}

std::vector<CameraPose> ugp2p_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X){
    std::vector<CameraPose> output;
    ugp2p(p, x, X, &output);
    return output;
}

std::vector<CameraPose> ugp3ps_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
           const std::vector<Eigen::Vector3d> &X, bool filter_solutions = true){
    std::vector<CameraPose> output;
    ugp3ps(p, x, X, &output, filter_solutions);
    return output;
}

std::vector<CameraPose> up1p2pl_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
            const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
            const std::vector<Eigen::Vector3d> &V){
    std::vector<CameraPose> output;
    up1p2pl(xp, Xp, x, X, V, &output);
    return output;
}

std::vector<CameraPose> up4pl_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
          const std::vector<Eigen::Vector3d> &V){
    std::vector<CameraPose> output;
    up4pl(x, X, V, &output);
    return output;
}

std::vector<CameraPose> ugp4pl_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
           const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector3d> &V){
    std::vector<CameraPose> output;
    ugp4pl(p, x, X, V, &output);
    return output;
}

std::vector<Eigen::Matrix3d> essential_matrix_relpose_5pt_wrapper(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2){
    std::vector<Eigen::Matrix3d> essential_matrices;
    relpose_5pt(x1, x2, &essential_matrices);
    return essential_matrices;
}
std::vector<CameraPose> relpose_5pt_wrapper(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2){
    std::vector<CameraPose> output;
    relpose_5pt(x1, x2, &output);
    return output;
}

std::vector<CameraPose> relpose_8pt_wrapper(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2){
    std::vector<CameraPose> output;
    relpose_8pt(x1, x2, &output);
    return output;
}
Eigen::Matrix3d essential_matrix_8pt_wrapper(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2){
    Eigen::Matrix3d essential_matrix;
    essential_matrix_8pt(x1, x2, &essential_matrix);
    return essential_matrix;
}

std::vector<CameraPose> relpose_upright_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2){
    std::vector<CameraPose> output;
    relpose_upright_3pt(x1, x2, &output);
    return output;
}

std::vector<CameraPose> gen_relpose_upright_4pt_wrapper(const std::vector<Eigen::Vector3d> &p1, const std::vector<Eigen::Vector3d> &x1,
                            const std::vector<Eigen::Vector3d> &p2, const std::vector<Eigen::Vector3d> &x2){
    std::vector<CameraPose> output;
    gen_relpose_upright_4pt(p1, x1, p2, x2, &output);
    return output;
}

std::vector<CameraPose> relpose_upright_planar_2pt_wrapper(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2){
    std::vector<CameraPose> output;
    relpose_upright_planar_2pt(x1, x2, &output);
    return output;
}

std::vector<CameraPose> relpose_upright_planar_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2){
    std::vector<CameraPose> output;
    relpose_upright_planar_3pt(x1, x2, &output);
    return output;
}

std::pair<CameraPose, py::dict> estimate_absolute_pose_wrapper(
                                const std::vector<Eigen::Vector2d> points2D,
                                const std::vector<Eigen::Vector3d> points3D,
                                const py::dict &camera_dict,
                                const py::dict &ransac_opt_dict,
                                const py::dict &bundle_opt_dict){

    Camera camera = camera_from_dict(camera_dict);

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    std::vector<char> inlier_mask;

    RansacStats stats = estimate_absolute_pose(points2D, points3D, camera, ransac_opt, bundle_opt, &pose, &inlier_mask);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);

    return std::make_pair(pose, output_dict);
}


std::pair<CameraPose, py::dict> estimate_absolute_pose_pnpl_wrapper(
                                const std::vector<Eigen::Vector2d> points2D, const std::vector<Eigen::Vector3d> points3D,
                                const std::vector<Eigen::Vector2d> lines2D_1, const std::vector<Eigen::Vector2d> lines2D_2,
                                const std::vector<Eigen::Vector3d> lines3D_1, const std::vector<Eigen::Vector3d> lines3D_2,
                                const py::dict &camera_dict, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    Camera camera = camera_from_dict(camera_dict);

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    std::vector<Line2D> lines2D;
    std::vector<Line3D> lines3D;
    lines2D.reserve(lines2D_1.size());
    lines3D.reserve(lines3D_1.size());
    for(size_t k = 0; k < lines2D_1.size(); ++k) {
        lines2D.emplace_back(lines2D_1[k], lines2D_2[k]);
        lines3D.emplace_back(lines3D_1[k], lines3D_2[k]);
    }

    CameraPose pose;
    std::vector<char> inlier_points_mask;
    std::vector<char> inlier_lines_mask;

    RansacStats stats = estimate_absolute_pose_pnpl(points2D, points3D, lines2D, lines3D,
                         camera, ransac_opt, bundle_opt, &pose, &inlier_points_mask, &inlier_lines_mask);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_points_mask);
    output_dict["inliers_lines"] = convert_inlier_vector(inlier_lines_mask);
    return std::make_pair(pose, output_dict);
}


std::pair<CameraPose,py::dict> estimate_generalized_absolute_pose_wrapper(const std::vector<std::vector<Eigen::Vector2d>> points2D, const std::vector<std::vector<Eigen::Vector3d>> points3D,
                                const std::vector<CameraPose> &camera_ext, const std::vector<py::dict> &camera_dicts, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict){

    std::vector<Camera> cameras;
    for(const py::dict &camera_dict : camera_dicts) {
        cameras.push_back(camera_from_dict(camera_dict));
    }

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    std::vector<std::vector<char>> inlier_mask;

    RansacStats stats = estimate_generalized_absolute_pose(points2D, points3D, camera_ext, cameras, ransac_opt, bundle_opt, &pose, &inlier_mask);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vectors(inlier_mask);
    return std::make_pair(pose, output_dict);
}


std::pair<CameraPose,py::dict> estimate_relative_pose_wrapper(const std::vector<Eigen::Vector2d> points2D_1, const std::vector<Eigen::Vector2d> points2D_2,
                                const py::dict &camera1_dict, const py::dict &camera2_dict, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict){

    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    std::vector<char> inlier_mask;

    RansacStats stats = estimate_relative_pose(points2D_1, points2D_2, camera1, camera2, ransac_opt, bundle_opt, &pose, &inlier_mask);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(pose, output_dict);
}


std::pair<CameraPose,py::dict> refine_relative_pose_wrapper(const std::vector<Eigen::Vector2d> points2D_1, const std::vector<Eigen::Vector2d> points2D_2,
                                const CameraPose initial_pose, const py::dict &camera1_dict, const py::dict &camera2_dict, const py::dict &bundle_opt_dict){

    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    // Normalize image points
    std::vector<Eigen::Vector2d> x1_calib = points2D_1;
    std::vector<Eigen::Vector2d> x2_calib = points2D_2;

    for(size_t i = 0; i < x1_calib.size(); ++i) {
        camera1.unproject(x1_calib[i], &x1_calib[i]);
        camera2.unproject(x2_calib[i], &x2_calib[i]);
    }
    bundle_opt.loss_scale *= (1.0 / camera1.focal() + 1.0/camera2.focal()) * 0.5;


    CameraPose pose = initial_pose;
    BundleStats stats = refine_relpose(x1_calib, x2_calib, &pose, bundle_opt);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    return std::make_pair(pose, output_dict);
}


std::pair<Eigen::Matrix3d,py::dict> estimate_fundamental_wrapper(
                        const std::vector<Eigen::Vector2d> points2D_1,
                        const std::vector<Eigen::Vector2d> points2D_2,
                        const py::dict &ransac_opt_dict,
                        const py::dict &bundle_opt_dict){
    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    Eigen::Matrix3d F;
    std::vector<char> inlier_mask;

    RansacStats stats = estimate_fundamental(points2D_1, points2D_2, ransac_opt, bundle_opt, &F, &inlier_mask);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(F, output_dict);
}



std::pair<Eigen::Matrix3d,py::dict> estimate_homography_wrapper(
                                     const std::vector<Eigen::Vector2d> points2D_1,
                                     const std::vector<Eigen::Vector2d> points2D_2,
                                     const py::dict &ransac_opt_dict,
                                     const py::dict &bundle_opt_dict){

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    Eigen::Matrix3d H;
    std::vector<char> inlier_mask;

    RansacStats stats = estimate_homography(points2D_1, points2D_2, ransac_opt, bundle_opt, &H, &inlier_mask);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(H, output_dict);
}

std::pair<CameraPose,py::dict> estimate_generalized_relative_pose_wrapper(
                        const std::vector<PairwiseMatches> matches,
                        const std::vector<CameraPose> &camera1_ext, const std::vector<py::dict> &cameras1_dict,
                        const std::vector<CameraPose> &camera2_ext, const std::vector<py::dict> &cameras2_dict,
                        const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict){

    std::vector<Camera> cameras1, cameras2;
    for(const py::dict &camera_dict : cameras1_dict) {
        cameras1.push_back(camera_from_dict(camera_dict));
    }
    for(const py::dict &camera_dict : cameras2_dict) {
        cameras2.push_back(camera_from_dict(camera_dict));
    }

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    std::vector<std::vector<char>> inlier_mask;

    RansacStats stats = estimate_generalized_relative_pose(matches, camera1_ext, cameras1, camera2_ext, cameras2, ransac_opt, bundle_opt, &pose, &inlier_mask);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vectors(inlier_mask);
    return std::make_pair(pose, output_dict);
}



std::pair<CameraPose,py::dict> estimate_hybrid_pose_wrapper(
                        const std::vector<Eigen::Vector2d> points2D, const std::vector<Eigen::Vector3d> points3D,
                        const std::vector<PairwiseMatches> matches_2D_2D,
                        const py::dict &camera_dict,
                        const std::vector<CameraPose> &map_ext, const std::vector<py::dict> &map_camera_dicts,
                        const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict){

    Camera camera = camera_from_dict(camera_dict);
    std::vector<Camera> map_cameras;
    for(const py::dict &camera_dict : map_camera_dicts) {
        map_cameras.push_back(camera_from_dict(camera_dict));
    }

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    std::vector<char> inliers_mask_2d3d;
    std::vector<std::vector<char>> inliers_mask_2d2d;

    RansacStats stats = estimate_hybrid_pose(points2D, points3D, matches_2D_2D, camera, map_ext, map_cameras, ransac_opt, bundle_opt, &pose, &inliers_mask_2d3d, &inliers_mask_2d2d);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inliers_mask_2d3d);
    output_dict["inliers_2D"] = convert_inlier_vectors(inliers_mask_2d2d);
    return std::make_pair(pose, output_dict);
}



std::pair<CameraPose,py::dict> estimate_1D_radial_absolute_pose_wrapper(
                        const std::vector<Eigen::Vector2d> points2D,
                        const std::vector<Eigen::Vector3d> points3D,
                        const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict){

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    std::vector<char> inlier_mask;

    RansacStats stats = estimate_1D_radial_absolute_pose(points2D, points3D, ransac_opt, bundle_opt, &pose, &inlier_mask);

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);
    return std::make_pair(pose, output_dict);
}


}


PYBIND11_MODULE(poselib, m)
{
    py::class_<pose_lib::CameraPose>(m, "CameraPose")
            .def(py::init<>())
            .def_readwrite("R", &pose_lib::CameraPose::R)
            .def_readwrite("t", &pose_lib::CameraPose::t)
            .def_readwrite("alpha", &pose_lib::CameraPose::alpha)
            .def("__repr__",
                [](const pose_lib::CameraPose &a) {
                    return "[R: \n" + toString(a.R) + "\n" +
                            "t: \n" + toString(a.t) + "\n" +
                            "alpha: \n" + std::to_string(a.alpha) + "]\n";
                }
            );

    py::class_<pose_lib::PairwiseMatches>(m, "PairwiseMatches")
            .def(py::init<>())
            .def_readwrite("cam_id1", &pose_lib::PairwiseMatches::cam_id1)
            .def_readwrite("cam_id2", &pose_lib::PairwiseMatches::cam_id2)
            .def_readwrite("x1", &pose_lib::PairwiseMatches::x1)
            .def_readwrite("x2", &pose_lib::PairwiseMatches::x2)
            .def("__repr__",
                [](const pose_lib::PairwiseMatches &a) {
                    return "[cam_id1: " + std::to_string(a.cam_id1) + "\n" +
                            "cam_id2: " + std::to_string(a.cam_id2) + "\n" +
                            "x1: [2x" + std::to_string(a.x1.size()) + "]\n" +
                            "x2: [2x" + std::to_string(a.x2.size()) + "]]\n";
                }
            );

    m.doc() = "This library provides a collection of minimal solvers for camera pose estimation.";

    // Minimal solvers
    m.def("p3p", &pose_lib::p3p_wrapper, py::arg("x"), py::arg("X"));
    m.def("gp3p", &pose_lib::gp3p_wrapper, py::arg("p"), py::arg("x"), py::arg("X"));
    m.def("gp4ps", &pose_lib::gp4ps_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("filter_solutions"));
    m.def("gp4ps_kukelova", &pose_lib::gp4ps_kukelova_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("filter_solutions"));
    m.def("gp4ps_camposeco", &pose_lib::gp4ps_camposeco_wrapper, py::arg("p"), py::arg("x"), py::arg("X"));
    m.def("p4pf", &pose_lib::p4pf_wrapper, py::arg("x"), py::arg("X"), py::arg("filter_solutions"));
    m.def("p2p2pl", &pose_lib::p2p2pl_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("x"), py::arg("X"), py::arg("V"));
    m.def("p6lp", &pose_lib::p6lp_wrapper, py::arg("l"), py::arg("X"));
    m.def("p5lp_radial", &pose_lib::p5lp_radial_wrapper, py::arg("l"), py::arg("X"));
    m.def("p2p1ll", &pose_lib::p2p1ll_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"));
    m.def("p1p2ll", &pose_lib::p1p2ll_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"));
    m.def("p3ll", &pose_lib::p3ll_wrapper, py::arg("l"), py::arg("X"), py::arg("V"));
    m.def("up2p", &pose_lib::up2p_wrapper, py::arg("x"), py::arg("X"));
    m.def("ugp2p", &pose_lib::ugp2p_wrapper, py::arg("p"), py::arg("x"), py::arg("X"));
    m.def("ugp3ps", &pose_lib::ugp3ps_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("filter_solutions"));
    m.def("up1p2pl", &pose_lib::up1p2pl_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("x"), py::arg("X"), py::arg("V"));
    m.def("up4pl", &pose_lib::up4pl_wrapper, py::arg("x"), py::arg("X"), py::arg("V"));
    m.def("ugp4pl", &pose_lib::ugp4pl_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("V"));
    m.def("essential_matrix_5pt", &pose_lib::essential_matrix_relpose_5pt_wrapper, py::arg("x1"), py::arg("x2"));
    m.def("relpose_5pt", &pose_lib::relpose_5pt_wrapper, py::arg("x1"), py::arg("x2"));
    m.def("relpose_8pt", &pose_lib::relpose_8pt_wrapper, py::arg("x1"), py::arg("x2"));
    m.def("essential_matrix_8pt", &pose_lib::essential_matrix_8pt_wrapper, py::arg("x1"), py::arg("x2"));
    m.def("relpose_upright_3pt", &pose_lib::relpose_upright_3pt_wrapper, py::arg("x1"), py::arg("x2"));
    m.def("gen_relpose_upright_4pt", &pose_lib::gen_relpose_upright_4pt_wrapper, py::arg("p1"), py::arg("x1"), py::arg("p2"), py::arg("x2"));
    m.def("relpose_upright_planar_2pt", &pose_lib::relpose_upright_planar_2pt_wrapper, py::arg("x1"), py::arg("x2"));
    m.def("relpose_upright_planar_3pt", &pose_lib::relpose_upright_planar_3pt_wrapper, py::arg("x1"), py::arg("x2"));

    // Robust estimators
    m.def("estimate_absolute_pose", &pose_lib::estimate_absolute_pose_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("camera_dict"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),  "Absolute pose estimation with non-linear refinement.");
    m.def("estimate_absolute_pose_pnpl", &pose_lib::estimate_absolute_pose_pnpl_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"),py::arg("lines3D_1"), py::arg("lines3D_2"),py::arg("camera_dict"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),  "Absolute pose estimation with non-linear refinement from points and lines.");
    m.def("estimate_generalized_absolute_pose", &pose_lib::estimate_generalized_absolute_pose_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("camera_ext"), py::arg("camera_dicts"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),  "Generalized absolute pose estimation with non-linear refinement.");
    m.def("estimate_relative_pose", &pose_lib::estimate_relative_pose_wrapper, py::arg("points2D_1"), py::arg("points2D_2"), py::arg("camera1_dict"), py::arg("camera2_dict"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),  "Relative pose estimation with non-linear refinement.");
    m.def("estimate_fundamental", &pose_lib::estimate_fundamental_wrapper, py::arg("points2D_1"), py::arg("points2D_2"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), "Fundamental matrix estimation with non-linear refinement. Note: if you have known intrinsics you should use estimate_relative_pose instead!");
    m.def("estimate_homography", &pose_lib::estimate_homography_wrapper, py::arg("points2D_1"), py::arg("points2D_2"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), "Homography matrix estimation with non-linear refinement.");
    m.def("estimate_generalized_relative_pose", &pose_lib::estimate_generalized_relative_pose_wrapper, py::arg("matches"), py::arg("camera1_ext"), py::arg("camera1_dict"), py::arg("camera2_ext"), py::arg("camera2_dict"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),  "Generalized relative pose estimation with non-linear refinement.");
    m.def("estimate_hybrid_pose", &pose_lib::estimate_hybrid_pose_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("matches_2D_2D"), py::arg("camera_dict"), py::arg("map_ext"), py::arg("map_camera_dicts"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(), "Hybrid camera pose estimation (both 2D-3D and 2D-2D correspondences to the map) with non-linear refinement.");  
    m.def("estimate_1D_radial_absolute_pose", &pose_lib::estimate_1D_radial_absolute_pose_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),  "Absolute pose estimation for the 1D radial camera model with non-linear refinement.");

    // Stand-alone non-linear refinement
    m.def("refine_relative_pose", &pose_lib::refine_relative_pose_wrapper, py::arg("points2D_1"), py::arg("points2D_2"), py::arg("initial_pose"), py::arg("camera1_dict"), py::arg("camera2_dict"), py::arg("bundle_options") = py::dict(),  "Relative pose refinement with non-linear refinement.");

    m.def("RansacOptions", &pose_lib::RansacOptions_wrapper, py::arg("opt") = py::dict(), "Options for RANSAC.");
    m.def("BundleOptions", &pose_lib::BundleOptions_wrapper, py::arg("opt") = py::dict(), "Options for non-linear refinement.");

    m.attr("__version__") = std::string(POSELIB_VERSION);

}
