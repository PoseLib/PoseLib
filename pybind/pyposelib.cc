#include "helpers.h"
#include "pybind11_extension.h"

#include <PoseLib/poselib.h>
#include <iostream>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <tuple>

namespace py = pybind11;

namespace poselib {

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

std::vector<CameraPose> p3p_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    p3p(x, X, &output);
    return output;
}

std::vector<CameraPose> gp3p_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                                     const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    gp3p(p, x, X, &output);
    return output;
}

std::pair<std::vector<CameraPose>, std::vector<double>> gp4ps_wrapper(const std::vector<Eigen::Vector3d> &p,
                                                                      const std::vector<Eigen::Vector3d> &x,
                                                                      const std::vector<Eigen::Vector3d> &X,
                                                                      bool filter_solutions = true) {
    std::vector<CameraPose> output;
    std::vector<double> output_scales;
    gp4ps(p, x, X, &output, &output_scales, filter_solutions);
    return std::make_pair(output, output_scales);
}

std::pair<std::vector<CameraPose>, std::vector<double>> gp4ps_kukelova_wrapper(const std::vector<Eigen::Vector3d> &p,
                                                                               const std::vector<Eigen::Vector3d> &x,
                                                                               const std::vector<Eigen::Vector3d> &X,
                                                                               bool filter_solutions = true) {
    std::vector<CameraPose> output;
    std::vector<double> output_scales;
    gp4ps_kukelova(p, x, X, &output, &output_scales, filter_solutions);
    return std::make_pair(output, output_scales);
}

std::pair<std::vector<CameraPose>, std::vector<double>> gp4ps_camposeco_wrapper(const std::vector<Eigen::Vector3d> &p,
                                                                                const std::vector<Eigen::Vector3d> &x,
                                                                                const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    std::vector<double> output_scales;
    gp4ps_camposeco(p, x, X, &output, &output_scales);
    return std::make_pair(output, output_scales);
}

std::pair<std::vector<CameraPose>, std::vector<double>> p4pf_wrapper(const std::vector<Eigen::Vector2d> &x,
                                                                     const std::vector<Eigen::Vector3d> &X,
                                                                     bool filter_solutions = true) {
    std::vector<CameraPose> output;
    std::vector<double> output_focal;
    p4pf(x, X, &output, &output_focal, filter_solutions);
    return std::make_pair(output, output_focal);
}

std::vector<CameraPose> p2p2pl_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                                       const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
                                       const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    p2p2pl(xp, Xp, x, X, V, &output);
    return output;
}

std::vector<CameraPose> p6lp_wrapper(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    p6lp(l, X, &output);
    return output;
}

std::vector<CameraPose> p5lp_radial_wrapper(const std::vector<Eigen::Vector3d> &l,
                                            const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    p5lp_radial(l, X, &output);
    return output;
}

std::vector<CameraPose> p2p1ll_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                                       const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                                       const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    p2p1ll(xp, Xp, l, X, V, &output);
    return output;
}

std::vector<CameraPose> p1p2ll_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                                       const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                                       const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    p1p2ll(xp, Xp, l, X, V, &output);
    return output;
}

std::vector<CameraPose> p3ll_wrapper(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                                     const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    p3ll(l, X, V, &output);
    return output;
}

std::vector<CameraPose> up2p_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    up2p(x, X, &output);
    return output;
}

std::vector<CameraPose> ugp2p_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                                      const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    ugp2p(p, x, X, &output);
    return output;
}

std::pair<std::vector<CameraPose>, std::vector<double>> ugp3ps_wrapper(const std::vector<Eigen::Vector3d> &p,
                                                                       const std::vector<Eigen::Vector3d> &x,
                                                                       const std::vector<Eigen::Vector3d> &X,
                                                                       bool filter_solutions = true) {
    std::vector<CameraPose> output;
    std::vector<double> output_scales;
    ugp3ps(p, x, X, &output, &output_scales, filter_solutions);
    return std::make_pair(output, output_scales);
}

std::vector<CameraPose> up1p2pl_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                                        const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
                                        const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    up1p2pl(xp, Xp, x, X, V, &output);
    return output;
}

std::vector<CameraPose> up4pl_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
                                      const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    up4pl(x, X, V, &output);
    return output;
}

std::vector<CameraPose> ugp4pl_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                                       const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    ugp4pl(p, x, X, V, &output);
    return output;
}

std::vector<Eigen::Matrix3d> essential_matrix_relpose_5pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                                  const std::vector<Eigen::Vector3d> &x2) {
    std::vector<Eigen::Matrix3d> essential_matrices;
    relpose_5pt(x1, x2, &essential_matrices);
    return essential_matrices;
}
std::vector<CameraPose> relpose_5pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                            const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_5pt(x1, x2, &output);
    return output;
}
ImagePairVector shared_focal_relpose_6pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                 const std::vector<Eigen::Vector3d> &x2) {
    ImagePairVector output;
    relpose_6pt_shared_focal(x1, x2, &output);

    return output;
}
std::vector<CameraPose> relpose_8pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                            const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_8pt(x1, x2, &output);
    return output;
}
Eigen::Matrix3d essential_matrix_8pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                             const std::vector<Eigen::Vector3d> &x2) {
    Eigen::Matrix3d essential_matrix;
    essential_matrix_8pt(x1, x2, &essential_matrix);
    return essential_matrix;
}

std::vector<CameraPose> relpose_upright_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                    const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_upright_3pt(x1, x2, &output);
    return output;
}

std::vector<CameraPose> gen_relpose_upright_4pt_wrapper(const std::vector<Eigen::Vector3d> &p1,
                                                        const std::vector<Eigen::Vector3d> &x1,
                                                        const std::vector<Eigen::Vector3d> &p2,
                                                        const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    gen_relpose_upright_4pt(p1, x1, p2, x2, &output);
    return output;
}

std::vector<CameraPose> gen_relpose_6pt_wrapper(const std::vector<Eigen::Vector3d> &p1,
                                                const std::vector<Eigen::Vector3d> &x1,
                                                const std::vector<Eigen::Vector3d> &p2,
                                                const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    gen_relpose_6pt(p1, x1, p2, x2, &output);
    return output;
}

std::vector<CameraPose> relpose_upright_planar_2pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                           const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_upright_planar_2pt(x1, x2, &output);
    return output;
}

std::vector<CameraPose> relpose_upright_planar_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                           const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_upright_planar_3pt(x1, x2, &output);
    return output;
}

std::pair<CameraPose, py::dict> estimate_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
                                                               const std::vector<Eigen::Vector3d> &points3D,
                                                               const Camera &camera, const py::dict &ransac_opt_dict,
                                                               const py::dict &bundle_opt_dict) {
    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_absolute_pose(points2D, points3D, camera, ransac_opt, bundle_opt, &pose, &inlier_mask);
    py::gil_scoped_acquire acquire;

    py::dict output_dict;
    write_to_dict(stats, output_dict);
    output_dict["inliers"] = convert_inlier_vector(inlier_mask);

    return std::make_pair(pose, output_dict);
}

std::pair<CameraPose, py::dict> estimate_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
                                                               const std::vector<Eigen::Vector3d> &points3D,
                                                               const py::dict &camera_dict,
                                                               const py::dict &ransac_opt_dict,
                                                               const py::dict &bundle_opt_dict) {
    Camera camera = camera_from_dict(camera_dict);
    return estimate_absolute_pose_wrapper(points2D, points3D, camera, ransac_opt_dict, bundle_opt_dict);
}

std::pair<CameraPose, py::dict> refine_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
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

std::pair<CameraPose, py::dict> refine_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
                                                             const std::vector<Eigen::Vector3d> &points3D,
                                                             const CameraPose &initial_pose,
                                                             const py::dict &camera_dict,
                                                             const py::dict &bundle_opt_dict) {

    Camera camera = camera_from_dict(camera_dict);
    return refine_absolute_pose_wrapper(points2D, points3D, initial_pose, camera, bundle_opt_dict);
}

std::pair<CameraPose, py::dict> estimate_absolute_pose_pnpl_wrapper(
    const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
    const std::vector<Eigen::Vector2d> &lines2D_1, const std::vector<Eigen::Vector2d> &lines2D_2,
    const std::vector<Eigen::Vector3d> &lines3D_1, const std::vector<Eigen::Vector3d> &lines3D_2, const Camera &camera,
    const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

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

std::pair<CameraPose, py::dict> estimate_absolute_pose_pnpl_wrapper(
    const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
    const std::vector<Eigen::Vector2d> &lines2D_1, const std::vector<Eigen::Vector2d> &lines2D_2,
    const std::vector<Eigen::Vector3d> &lines3D_1, const std::vector<Eigen::Vector3d> &lines3D_2,
    const py::dict &camera_dict, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    Camera camera = camera_from_dict(camera_dict);
    return estimate_absolute_pose_pnpl_wrapper(points2D, points3D, lines2D_1, lines2D_2, lines3D_1, lines3D_2, camera,
                                               ransac_opt_dict, bundle_opt_dict);
}

std::pair<CameraPose, py::dict> refine_absolute_pose_pnpl_wrapper(
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

std::pair<CameraPose, py::dict> refine_absolute_pose_pnpl_wrapper(
    const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
    const std::vector<Eigen::Vector2d> &lines2D_1, const std::vector<Eigen::Vector2d> &lines2D_2,
    const std::vector<Eigen::Vector3d> &lines3D_1, const std::vector<Eigen::Vector3d> &lines3D_2,
    const CameraPose &initial_pose, const py::dict &camera_dict, const py::dict &bundle_opt_dict,
    const py::dict &line_bundle_opt_dict) {

    Camera camera = camera_from_dict(camera_dict);
    return refine_absolute_pose_pnpl_wrapper(points2D, points3D, lines2D_1, lines2D_2, lines3D_1, lines3D_2,
                                             initial_pose, camera, bundle_opt_dict, line_bundle_opt_dict);
}

std::pair<CameraPose, py::dict> estimate_generalized_absolute_pose_wrapper(
    const std::vector<std::vector<Eigen::Vector2d>> &points2D,
    const std::vector<std::vector<Eigen::Vector3d>> &points3D, const std::vector<CameraPose> &camera_ext,
    const std::vector<Camera> &cameras, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
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

std::pair<CameraPose, py::dict> estimate_generalized_absolute_pose_wrapper(
    const std::vector<std::vector<Eigen::Vector2d>> &points2D,
    const std::vector<std::vector<Eigen::Vector3d>> &points3D, const std::vector<CameraPose> &camera_ext,
    const std::vector<py::dict> &camera_dicts, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    std::vector<Camera> cameras;
    for (const py::dict &camera_dict : camera_dicts) {
        cameras.push_back(camera_from_dict(camera_dict));
    }

    return estimate_generalized_absolute_pose_wrapper(points2D, points3D, camera_ext, cameras, ransac_opt_dict,
                                                      bundle_opt_dict);
}

std::pair<CameraPose, py::dict>
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

std::pair<CameraPose, py::dict>
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

std::pair<CameraPose, py::dict> estimate_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                               const std::vector<Eigen::Vector2d> &points2D_2,
                                                               const Camera &camera1, const Camera &camera2,
                                                               const py::dict &ransac_opt_dict,
                                                               const py::dict &bundle_opt_dict) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
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

std::pair<CameraPose, py::dict> estimate_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                               const std::vector<Eigen::Vector2d> &points2D_2,
                                                               const py::dict &camera1_dict,
                                                               const py::dict &camera2_dict,
                                                               const py::dict &ransac_opt_dict,
                                                               const py::dict &bundle_opt_dict) {
    Camera camera1 = camera_from_dict(camera1_dict);
    Camera camera2 = camera_from_dict(camera2_dict);

    return estimate_relative_pose_wrapper(points2D_1, points2D_2, camera1, camera2, ransac_opt_dict, bundle_opt_dict);
}

std::pair<ImagePair, py::dict>
estimate_shared_focal_relative_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                            const std::vector<Eigen::Vector2d> &points2D_2, const Eigen::Vector2d &pp,
                                            const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    ImagePair image_pair;
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
                                                                  const py::dict &bundle_opt_dict) {
    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    Eigen::Matrix3d F;
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

std::pair<Eigen::Matrix3d, py::dict> estimate_homography_wrapper(const std::vector<Eigen::Vector2d> &points2D_1,
                                                                 const std::vector<Eigen::Vector2d> &points2D_2,
                                                                 const py::dict &ransac_opt_dict,
                                                                 const py::dict &bundle_opt_dict) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    Eigen::Matrix3d H;
    std::vector<char> inlier_mask;

    py::gil_scoped_release release;
    RansacStats stats = estimate_homography(points2D_1, points2D_2, ransac_opt, bundle_opt, &H, &inlier_mask);
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

std::pair<CameraPose, py::dict> estimate_generalized_relative_pose_wrapper(
    const std::vector<PairwiseMatches> &matches, const std::vector<CameraPose> &camera1_ext,
    const std::vector<Camera> &cameras1, const std::vector<CameraPose> &camera2_ext,
    const std::vector<Camera> &cameras2, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_epipolar_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
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
    const std::vector<py::dict> &cameras2_dict, const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    std::vector<Camera> cameras1, cameras2;
    for (const py::dict &camera_dict : cameras1_dict) {
        cameras1.push_back(camera_from_dict(camera_dict));
    }
    for (const py::dict &camera_dict : cameras2_dict) {
        cameras2.push_back(camera_from_dict(camera_dict));
    }

    return estimate_generalized_relative_pose_wrapper(matches, camera1_ext, cameras1, camera2_ext, cameras2,
                                                      ransac_opt_dict, bundle_opt_dict);
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

std::pair<CameraPose, py::dict>
estimate_hybrid_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D, const std::vector<Eigen::Vector3d> &points3D,
                             const std::vector<PairwiseMatches> &matches_2D_2D, const Camera &camera,
                             const std::vector<CameraPose> &map_ext, const std::vector<Camera> &map_cameras,
                             const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt; // TODO: figure out what do to here...
    bundle_opt.loss_scale = 0.25 * (ransac_opt.max_reproj_error + ransac_opt.max_epipolar_error);
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
    std::vector<char> inliers_mask_2d3d;
    std::vector<std::vector<char>> inliers_mask_2d2d;

    py::gil_scoped_release release;
    RansacStats stats = estimate_hybrid_pose(points2D, points3D, matches_2D_2D, camera, map_ext, map_cameras,
                                             ransac_opt, bundle_opt, &pose, &inliers_mask_2d3d, &inliers_mask_2d2d);
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
                             const py::dict &ransac_opt_dict, const py::dict &bundle_opt_dict) {

    Camera camera = camera_from_dict(camera_dict);
    std::vector<Camera> map_cameras;
    for (const py::dict &camera_dict : map_camera_dicts) {
        map_cameras.push_back(camera_from_dict(camera_dict));
    }

    return estimate_hybrid_pose_wrapper(points2D, points3D, matches_2D_2D, camera, map_ext, map_cameras,
                                        ransac_opt_dict, bundle_opt_dict);
}

std::pair<CameraPose, py::dict> estimate_1D_radial_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> &points2D,
                                                                         const std::vector<Eigen::Vector3d> &points3D,
                                                                         const py::dict &ransac_opt_dict,
                                                                         const py::dict &bundle_opt_dict) {

    RansacOptions ransac_opt;
    update_ransac_options(ransac_opt_dict, ransac_opt);

    BundleOptions bundle_opt;
    bundle_opt.loss_scale = 0.5 * ransac_opt.max_reproj_error;
    update_bundle_options(bundle_opt_dict, bundle_opt);

    CameraPose pose;
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

} // namespace poselib

PYBIND11_MODULE(poselib, m) {
    py::class_<poselib::CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def_readwrite("q", &poselib::CameraPose::q)
        .def_readwrite("t", &poselib::CameraPose::t)
        .def_property("R", &poselib::CameraPose::R,
                      [](poselib::CameraPose &self, Eigen::Matrix3d R_new) { self.q = poselib::rotmat_to_quat(R_new); })
        .def_property("Rt", &poselib::CameraPose::Rt,
                      [](poselib::CameraPose &self, Eigen::Matrix<double, 3, 4> Rt_new) {
                          self.q = poselib::rotmat_to_quat(Rt_new.leftCols<3>());
                          self.t = Rt_new.col(3);
                      })
        .def("center", &poselib::CameraPose::center, "Returns the camera center (c=-R^T*t).")
        .def("__repr__", [](const poselib::CameraPose &a) {
            return "[q: " + toString(a.q.transpose()) + ", " + "t: " + toString(a.t.transpose()) + "]";
        });

    py::class_<poselib::Camera>(m, "Camera")
        .def(py::init<>())
        .def_readwrite("model_id", &poselib::Camera::model_id)
        .def_readwrite("width", &poselib::Camera::width)
        .def_readwrite("height", &poselib::Camera::height)
        .def_readwrite("params", &poselib::Camera::params)
        .def("focal", &poselib::Camera::focal, "Returns the camera focal length.")
        .def("focal_x", &poselib::Camera::focal_x, "Returns the camera focal_x.")
        .def("focal_y", &poselib::Camera::focal_y, "Returns the camera focal_y.")
        .def("model_name", &poselib::Camera::model_name, "Returns the camera model name.")
        .def("prinicipal_point", &poselib::Camera::principal_point, "Returns the camera principal point.")
        .def("initialize_from_txt", &poselib::Camera::initialize_from_txt, "Initialize camera from a cameras.txt line")
        .def("project",
             [](poselib::Camera &self, std::vector<Eigen::Vector2d> &xp) {
                 std::vector<Eigen::Vector2d> x;
                 self.project(xp, &x);
                 return x;
             })
        .def("project_with_jac",
             [](poselib::Camera &self, std::vector<Eigen::Vector2d> &xp) {
                 std::vector<Eigen::Vector2d> x;
                 std::vector<Eigen::Matrix2d> jac;
                 self.project_with_jac(xp, &x, &jac);
                 return std::make_pair(x, jac);
             })
        .def("unproject",
             [](poselib::Camera &self, std::vector<Eigen::Vector2d> &x) {
                 std::vector<Eigen::Vector2d> xp;
                 self.unproject(x, &xp);
                 return xp;
             })
        .def("__repr__", [](const poselib::Camera &a) { return a.to_cameras_txt(); });

    py::class_<poselib::Image>(m, "Image")
        .def(py::init<>())
        .def_readwrite("camera", &poselib::Image::camera)
        .def_readwrite("pose", &poselib::Image::pose)
        .def("__repr__", [](const poselib::Image &a) {
            return "[pose q: " + toString(a.pose.q.transpose()) + ", t: " + toString(a.pose.t.transpose()) +
                   ", camera: " + a.camera.to_cameras_txt() + "]";
        });

    py::class_<poselib::ImagePair>(m, "ImagePair")
        .def(py::init<>())
        .def_readwrite("pose", &poselib::ImagePair::pose)
        .def_readwrite("camera1", &poselib::ImagePair::camera1)
        .def_readwrite("camera2", &poselib::ImagePair::camera2)
        .def("__repr__", [](const poselib::ImagePair &a) {
            return "[pose q: " + toString(a.pose.q.transpose()) + ", t: " + toString(a.pose.t.transpose()) +
                   ", camera1: " + a.camera1.to_cameras_txt() + ", camera2: " + a.camera2.to_cameras_txt() + "]";
        });

    py::class_<poselib::PairwiseMatches>(m, "PairwiseMatches")
        .def(py::init<>())
        .def_readwrite("cam_id1", &poselib::PairwiseMatches::cam_id1)
        .def_readwrite("cam_id2", &poselib::PairwiseMatches::cam_id2)
        .def_readwrite("x1", &poselib::PairwiseMatches::x1)
        .def_readwrite("x2", &poselib::PairwiseMatches::x2)
        .def("__repr__", [](const poselib::PairwiseMatches &a) {
            return "[cam_id1: " + std::to_string(a.cam_id1) + "\n" + "cam_id2: " + std::to_string(a.cam_id2) + "\n" +
                   "x1: [2x" + std::to_string(a.x1.size()) + "]\n" + "x2: [2x" + std::to_string(a.x2.size()) + "]]\n";
        });

    m.doc() = "This library provides a collection of minimal solvers for camera pose estimation.";

    // Minimal solvers
    m.def("p3p", &poselib::p3p_wrapper, py::arg("x"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("gp3p", &poselib::gp3p_wrapper, py::arg("p"), py::arg("x"), py::arg("X"),
          py::call_guard<py::gil_scoped_release>());
    m.def("gp4ps", &poselib::gp4ps_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("filter_solutions"),
          py::call_guard<py::gil_scoped_release>());
    m.def("gp4ps_kukelova", &poselib::gp4ps_kukelova_wrapper, py::arg("p"), py::arg("x"), py::arg("X"),
          py::arg("filter_solutions"), py::call_guard<py::gil_scoped_release>());
    m.def("gp4ps_camposeco", &poselib::gp4ps_camposeco_wrapper, py::arg("p"), py::arg("x"), py::arg("X"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p4pf", &poselib::p4pf_wrapper, py::arg("x"), py::arg("X"), py::arg("filter_solutions"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p2p2pl", &poselib::p2p2pl_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("x"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p6lp", &poselib::p6lp_wrapper, py::arg("l"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("p5lp_radial", &poselib::p5lp_radial_wrapper, py::arg("l"), py::arg("X"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p2p1ll", &poselib::p2p1ll_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p1p2ll", &poselib::p1p2ll_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p3ll", &poselib::p3ll_wrapper, py::arg("l"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("up2p", &poselib::up2p_wrapper, py::arg("x"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("ugp2p", &poselib::ugp2p_wrapper, py::arg("p"), py::arg("x"), py::arg("X"),
          py::call_guard<py::gil_scoped_release>());
    m.def("ugp3ps", &poselib::ugp3ps_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("filter_solutions"),
          py::call_guard<py::gil_scoped_release>());
    m.def("up1p2pl", &poselib::up1p2pl_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("x"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("up4pl", &poselib::up4pl_wrapper, py::arg("x"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("ugp4pl", &poselib::ugp4pl_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("essential_matrix_5pt", &poselib::essential_matrix_relpose_5pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("shared_focal_relpose_6pt", &poselib::shared_focal_relpose_6pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("relpose_5pt", &poselib::relpose_5pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("relpose_8pt", &poselib::relpose_8pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("essential_matrix_8pt", &poselib::essential_matrix_8pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("relpose_upright_3pt", &poselib::relpose_upright_3pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("gen_relpose_upright_4pt", &poselib::gen_relpose_upright_4pt_wrapper, py::arg("p1"), py::arg("x1"),
          py::arg("p2"), py::arg("x2"), py::call_guard<py::gil_scoped_release>());
    m.def("gen_relpose_6pt", &poselib::gen_relpose_6pt_wrapper, py::arg("p1"), py::arg("x1"), py::arg("p2"),
          py::arg("x2"), py::call_guard<py::gil_scoped_release>());
    m.def("relpose_upright_planar_2pt", &poselib::relpose_upright_planar_2pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("relpose_upright_planar_3pt", &poselib::relpose_upright_planar_3pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());

    // Robust estimators
    m.def("estimate_absolute_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const poselib::Camera &, const py::dict &, const py::dict &>(
              &poselib::estimate_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("camera"), py::arg("ransac_opt") = py::dict(),
          py::arg("bundle_opt") = py::dict(), "Absolute pose estimation with non-linear refinement.");
    m.def(
        "estimate_absolute_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &, const py::dict &,
                          const py::dict &, const py::dict &>(&poselib::estimate_absolute_pose_wrapper),
        py::arg("points2D"), py::arg("points3D"), py::arg("camera_dict"), py::arg("ransac_opt") = py::dict(),
        py::arg("bundle_opt") = py::dict(), "Absolute pose estimation with non-linear refinement.");

    m.def("estimate_absolute_pose_pnpl",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const std::vector<Eigen::Vector3d> &, const std::vector<Eigen::Vector3d> &,
                            const poselib::Camera &, const py::dict &, const py::dict &>(
              &poselib::estimate_absolute_pose_pnpl_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines3D_1"),
          py::arg("lines3D_2"), py::arg("camera"), py::arg("ransac_opt") = py::dict(),
          py::arg("bundle_opt") = py::dict(),
          "Absolute pose estimation with non-linear refinement from points and lines.");
    m.def(
        "estimate_absolute_pose_pnpl",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                          const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                          const std::vector<Eigen::Vector3d> &, const std::vector<Eigen::Vector3d> &, const py::dict &,
                          const py::dict &, const py::dict &>(&poselib::estimate_absolute_pose_pnpl_wrapper),
        py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines3D_1"),
        py::arg("lines3D_2"), py::arg("camera_dict"), py::arg("ransac_opt") = py::dict(),
        py::arg("bundle_opt") = py::dict(),
        "Absolute pose estimation with non-linear refinement from points and lines.");

    m.def("estimate_generalized_absolute_pose",
          py::overload_cast<const std::vector<std::vector<Eigen::Vector2d>> &,
                            const std::vector<std::vector<Eigen::Vector3d>> &, const std::vector<poselib::CameraPose> &,
                            const std::vector<poselib::Camera> &, const py::dict &, const py::dict &>(
              &poselib::estimate_generalized_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("camera_ext"), py::arg("cameras"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Generalized absolute pose estimation with non-linear refinement.");
    m.def("estimate_generalized_absolute_pose",
          py::overload_cast<const std::vector<std::vector<Eigen::Vector2d>> &,
                            const std::vector<std::vector<Eigen::Vector3d>> &, const std::vector<poselib::CameraPose> &,
                            const std::vector<py::dict> &, const py::dict &, const py::dict &>(
              &poselib::estimate_generalized_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("camera_ext"), py::arg("camera_dicts"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Generalized absolute pose estimation with non-linear refinement.");

    m.def("estimate_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const poselib::Camera &, const poselib::Camera &, const py::dict &, const py::dict &>(
              &poselib::estimate_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("camera1"), py::arg("camera2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Relative pose estimation with non-linear refinement.");
    m.def("estimate_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const py::dict &, const py::dict &, const py::dict &, const py::dict &>(
              &poselib::estimate_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("camera1_dict"), py::arg("camera2_dict"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Relative pose estimation with non-linear refinement.");

    m.def("estimate_shared_focal_relative_pose", &poselib::estimate_shared_focal_relative_pose_wrapper,
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("pp") = Eigen::Vector2d::Zero(),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Relative pose estimation with unknown equal focal lengths with non-linear refinement.");
    m.def("estimate_fundamental", &poselib::estimate_fundamental_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Fundamental matrix estimation with non-linear refinement. Note: if you have known intrinsics you should use "
          "estimate_relative_pose instead!");
    m.def("estimate_homography", &poselib::estimate_homography_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Homography matrix estimation with non-linear refinement.");

    m.def("estimate_generalized_relative_pose",
          py::overload_cast<const std::vector<poselib::PairwiseMatches> &, const std::vector<poselib::CameraPose> &,
                            const std::vector<poselib::Camera> &, const std::vector<poselib::CameraPose> &,
                            const std::vector<poselib::Camera> &, const py::dict &, const py::dict &>(
              &poselib::estimate_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("camera1_ext"), py::arg("cameras1"), py::arg("camera2_ext"), py::arg("cameras2"),
          py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Generalized relative pose estimation with non-linear refinement.");
    m.def("estimate_generalized_relative_pose",
          py::overload_cast<const std::vector<poselib::PairwiseMatches> &, const std::vector<poselib::CameraPose> &,
                            const std::vector<py::dict> &, const std::vector<poselib::CameraPose> &,
                            const std::vector<py::dict> &, const py::dict &, const py::dict &>(
              &poselib::estimate_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("camera1_ext"), py::arg("camera1_dict"), py::arg("camera2_ext"),
          py::arg("camera2_dict"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Generalized relative pose estimation with non-linear refinement.");

    m.def(
        "estimate_hybrid_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                          const std::vector<poselib::PairwiseMatches> &, const poselib::Camera &,
                          const std::vector<poselib::CameraPose> &, const std::vector<poselib::Camera> &,
                          const py::dict &, const py::dict &>(&poselib::estimate_hybrid_pose_wrapper),
        py::arg("points2D"), py::arg("points3D"), py::arg("matches_2D_2D"), py::arg("camera"), py::arg("map_ext"),
        py::arg("map_cameras"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
        "Hybrid camera pose estimation (both 2D-3D and 2D-2D correspondences to the map) with non-linear refinement.");
    m.def(
        "estimate_hybrid_pose",
        py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                          const std::vector<poselib::PairwiseMatches> &, const py::dict &,
                          const std::vector<poselib::CameraPose> &, const std::vector<py::dict> &, const py::dict &,
                          const py::dict &>(&poselib::estimate_hybrid_pose_wrapper),
        py::arg("points2D"), py::arg("points3D"), py::arg("matches_2D_2D"), py::arg("camera_dict"), py::arg("map_ext"),
        py::arg("map_camera_dicts"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
        "Hybrid camera pose estimation (both 2D-3D and 2D-2D correspondences to the map) with non-linear refinement.");

    m.def("estimate_1D_radial_absolute_pose", &poselib::estimate_1D_radial_absolute_pose_wrapper, py::arg("points2D"),
          py::arg("points3D"), py::arg("ransac_opt") = py::dict(), py::arg("bundle_opt") = py::dict(),
          "Absolute pose estimation for the 1D radial camera model with non-linear refinement.");

    m.def("motion_from_homography", &poselib::motion_from_homography_wrapper, py::arg("H"));
    m.def("focals_from_fundamental", &poselib::focals_from_fundamental, py::arg("F"), py::arg("pp1"), py::arg("pp2"));
    m.def("focals_from_fundamental_iterative", &poselib::focals_from_fundamental_iterative, py::arg("F"),
          py::arg("camera1"), py::arg("camera2"), py::arg("max_iters") = 50,
          py::arg("weights") = Eigen::Vector4d(5.0e-4, 1.0, 5.0e-4, 1.0));
    m.def("focals_from_fundamental_iterative", &poselib::focals_from_fundamental_iterative_wrapper, py::arg("F"),
          py::arg("camera1_dict"), py::arg("camera2_dict"), py::arg("max_iters") = 50,
          py::arg("weights") = Eigen::Vector4d(5.0e-4, 1.0, 5.0e-4, 1.0));

    // Stand-alone non-linear refinement
    m.def("refine_absolute_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const poselib::CameraPose &, const poselib::Camera &, const py::dict &>(
              &poselib::refine_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("camera"),
          py::arg("bundle_options") = py::dict(), "Absolute pose non-linear refinement.");
    m.def("refine_absolute_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const poselib::CameraPose &, const py::dict &, const py::dict &>(
              &poselib::refine_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("camera_dict"),
          py::arg("bundle_options") = py::dict(), "Absolute pose non-linear refinement.");

    m.def("refine_absolute_pose_pnpl",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const std::vector<Eigen::Vector3d> &, const std::vector<Eigen::Vector3d> &,
                            const poselib::CameraPose &, const poselib::Camera &, const py::dict &, const py::dict &>(
              &poselib::refine_absolute_pose_pnpl_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines3D_1"),
          py::arg("lines3D_2"), py::arg("initial_pose"), py::arg("camera"), py::arg("bundle_opt") = py::dict(),
          py::arg("line_bundle_opt") = py::dict(), "Absolute pose non-linear refinement from points and lines.");
    m.def("refine_absolute_pose_pnpl",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector3d> &,
                            const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const std::vector<Eigen::Vector3d> &, const std::vector<Eigen::Vector3d> &,
                            const poselib::CameraPose &, const py::dict &, const py::dict &, const py::dict &>(
              &poselib::refine_absolute_pose_pnpl_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("lines2D_1"), py::arg("lines2D_2"), py::arg("lines3D_1"),
          py::arg("lines3D_2"), py::arg("initial_pose"), py::arg("camera_dict"), py::arg("bundle_opt") = py::dict(),
          py::arg("line_bundle_opt") = py::dict(), "Absolute pose non-linear refinement from points and lines.");

    m.def("refine_generalized_absolute_pose",
          py::overload_cast<const std::vector<std::vector<Eigen::Vector2d>> &,
                            const std::vector<std::vector<Eigen::Vector3d>> &, const poselib::CameraPose &,
                            const std::vector<poselib::CameraPose> &, const std::vector<poselib::Camera> &,
                            const py::dict &>(&poselib::refine_generalized_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("camera_ext"), py::arg("cameras"),
          py::arg("bundle_opt") = py::dict(), "Generalized absolute pose non-linear refinement.");
    m.def("refine_generalized_absolute_pose",
          py::overload_cast<const std::vector<std::vector<Eigen::Vector2d>> &,
                            const std::vector<std::vector<Eigen::Vector3d>> &, const poselib::CameraPose &,
                            const std::vector<poselib::CameraPose> &, const std::vector<py::dict> &, const py::dict &>(
              &poselib::refine_generalized_absolute_pose_wrapper),
          py::arg("points2D"), py::arg("points3D"), py::arg("initial_pose"), py::arg("camera_ext"),
          py::arg("camera_dicts"), py::arg("bundle_opt") = py::dict(),
          "Generalized absolute pose non-linear refinement.");

    m.def("refine_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const poselib::CameraPose &, const poselib::Camera &, const poselib::Camera &,
                            const py::dict &>(&poselib::refine_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("initial_pose"), py::arg("camera1"), py::arg("camera2"),
          py::arg("bundle_options") = py::dict(), "Relative pose non-linear refinement.");
    m.def("refine_relative_pose",
          py::overload_cast<const std::vector<Eigen::Vector2d> &, const std::vector<Eigen::Vector2d> &,
                            const poselib::CameraPose &, const py::dict &, const py::dict &, const py::dict &>(
              &poselib::refine_relative_pose_wrapper),
          py::arg("points2D_1"), py::arg("points2D_2"), py::arg("initial_pose"), py::arg("camera1_dict"),
          py::arg("camera2_dict"), py::arg("bundle_options") = py::dict(), "Relative pose non-linear refinement.");

    m.def("refine_homography", &poselib::refine_homography_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("initial_H"), py::arg("bundle_options") = py::dict(), "Homography non-linear refinement.");

    m.def("refine_fundamental", &poselib::refine_fundamental_wrapper, py::arg("points2D_1"), py::arg("points2D_2"),
          py::arg("initial_F"), py::arg("bundle_options") = py::dict(), "Fundamental matrix non-linear refinement.");

    m.def("refine_generalized_relative_pose",
          py::overload_cast<const std::vector<poselib::PairwiseMatches> &, const poselib::CameraPose &,
                            const std::vector<poselib::CameraPose> &, const std::vector<poselib::Camera> &,
                            const std::vector<poselib::CameraPose> &, const std::vector<poselib::Camera> &,
                            const py::dict &>(&poselib::refine_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("initial_pose"), py::arg("camera1_ext"), py::arg("cameras1"),
          py::arg("camera2_ext"), py::arg("cameras2"), py::arg("bundle_opt") = py::dict(),
          "Generalized relative pose non-linear refinement.");
    m.def("refine_generalized_relative_pose",
          py::overload_cast<const std::vector<poselib::PairwiseMatches> &, const poselib::CameraPose &,
                            const std::vector<poselib::CameraPose> &, const std::vector<py::dict> &,
                            const std::vector<poselib::CameraPose> &, const std::vector<py::dict> &, const py::dict &>(
              &poselib::refine_generalized_relative_pose_wrapper),
          py::arg("matches"), py::arg("initial_pose"), py::arg("camera1_ext"), py::arg("camera1_dict"),
          py::arg("camera2_ext"), py::arg("camera2_dict"), py::arg("bundle_opt") = py::dict(),
          "Generalized relative pose non-linear refinement.");

    m.def("RansacOptions", &poselib::RansacOptions_wrapper, py::arg("opt") = py::dict(), "Options for RANSAC.");
    m.def("BundleOptions", &poselib::BundleOptions_wrapper, py::arg("opt") = py::dict(),
          "Options for non-linear refinement.");

    m.attr("__version__") = std::string(POSELIB_VERSION);
}
