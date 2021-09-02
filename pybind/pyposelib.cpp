#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <PoseLib/version.h>
#include <PoseLib/poselib.h>

static std::string toString(const Eigen::MatrixXd& mat){
    std::stringstream ss;
    ss << mat;
    return ss.str();
}


namespace py = pybind11;

namespace pose_lib {



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

py::dict estimate_absolute_pose_wrapper(const std::vector<Eigen::Vector2d> points2D, const std::vector<Eigen::Vector3d> points3D,
                                const py::dict &camera_dict,
                                const double max_reproj_error){
    
    Camera camera;
    camera.model_id = Camera::id_from_string(camera_dict["model"].cast<std::string>());    
    camera.width = camera_dict["width"].cast<size_t>();
    camera.height = camera_dict["height"].cast<size_t>();
    camera.params = camera_dict["params"].cast<std::vector<double>>();

    // Options chosen to be similar to pycolmap
    RansacOptions ransac_opt;
    ransac_opt.max_reproj_error = max_reproj_error;
    ransac_opt.min_iterations = 1000;
    ransac_opt.max_iterations = 100000;
    ransac_opt.success_prob = 0.9999;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::CAUCHY;
    bundle_opt.loss_scale = 1.0;
    bundle_opt.max_iterations = 100;

    CameraPose pose;
    std::vector<char> inlier_mask;

    RansacStats stats = estimate_absolute_pose(points2D, points3D, camera, ransac_opt, bundle_opt, &pose, &inlier_mask);

    if(stats.num_inliers == 0) {
        py::dict failure_dict;
        failure_dict["success"] = false;
        return failure_dict;
    }


    // Convert vector<char> to vector<bool>.
    std::vector<bool> inliers;
    for (auto it : inlier_mask) {
        if (it) {
            inliers.push_back(true);
        } else {
            inliers.push_back(false);
        }
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["success"] = true;
    success_dict["pose"] = pose;    
    success_dict["inliers"] = inliers;
    success_dict["num_inliers"] = stats.num_inliers;
    success_dict["iterations"] = stats.iterations;
    success_dict["inlier_ratio"] = stats.inlier_ratio;
    success_dict["refinements"] = stats.refinements;
    success_dict["model_score"] = stats.model_score;

    return success_dict;
}


py::dict estimate_generalized_absolute_pose_wrapper(const std::vector<std::vector<Eigen::Vector2d>> points2D, const std::vector<std::vector<Eigen::Vector3d>> points3D,
                                const std::vector<CameraPose> &camera_ext, const std::vector<py::dict> &camera_dicts, const double max_reproj_error){
    
    std::vector<Camera> cameras;
    for(py::dict camera_dict : camera_dicts) {
        cameras.emplace_back();
        cameras.back().model_id = Camera::id_from_string(camera_dict["model"].cast<std::string>());    
        cameras.back().width = camera_dict["width"].cast<size_t>();
        cameras.back().height = camera_dict["height"].cast<size_t>();
        cameras.back().params = camera_dict["params"].cast<std::vector<double>>();
    }

    // Options chosen to be similar to pycolmap
    RansacOptions ransac_opt;
    ransac_opt.max_reproj_error = max_reproj_error;
    ransac_opt.min_iterations = 1000;
    ransac_opt.max_iterations = 100000;
    ransac_opt.success_prob = 0.9999;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::CAUCHY;
    bundle_opt.loss_scale = 1.0;
    bundle_opt.max_iterations = 1000;

    CameraPose pose;
    std::vector<std::vector<char>> inlier_mask;

    RansacStats stats = estimate_generalized_absolute_pose(points2D, points3D, camera_ext, cameras, ransac_opt, bundle_opt, &pose, &inlier_mask);

    if(stats.num_inliers == 0) {
        py::dict failure_dict;
        failure_dict["success"] = false;
        return failure_dict;
    }

    // Convert vector<char> to vector<bool>.
    std::vector<std::vector<bool>> inliers(inlier_mask.size());
    for(size_t cam_k = 0; cam_k < inlier_mask.size(); ++cam_k) {
        inliers[cam_k].resize(inlier_mask[cam_k].size());
        for(size_t pt_k = 0; pt_k < inlier_mask[cam_k].size(); ++pt_k) {
            inliers[cam_k][pt_k] = static_cast<bool>(inlier_mask[cam_k][pt_k]);
        }
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["success"] = true;
    success_dict["pose"] = pose;    
    success_dict["inliers"] = inliers;
    success_dict["num_inliers"] = stats.num_inliers;
    success_dict["iterations"] = stats.iterations;
    success_dict["inlier_ratio"] = stats.inlier_ratio;
    success_dict["refinements"] = stats.refinements;
    success_dict["model_score"] = stats.model_score;

    return success_dict;
}


py::dict estimate_relative_pose_wrapper(const std::vector<Eigen::Vector2d> points2D_1, const std::vector<Eigen::Vector2d> points2D_2,
                                const py::dict &camera1_dict, const py::dict &camera2_dict, const double max_reproj_error){
    
    Camera camera1;
    camera1.model_id = Camera::id_from_string(camera1_dict["model"].cast<std::string>());    
    camera1.width = camera1_dict["width"].cast<size_t>();
    camera1.height = camera1_dict["height"].cast<size_t>();
    camera1.params = camera1_dict["params"].cast<std::vector<double>>();
    
    Camera camera2;
    camera2.model_id = Camera::id_from_string(camera2_dict["model"].cast<std::string>());    
    camera2.width = camera2_dict["width"].cast<size_t>();
    camera2.height = camera2_dict["height"].cast<size_t>();
    camera2.params = camera2_dict["params"].cast<std::vector<double>>();
    
    // Options chosen to be similar to pycolmap
    RansacOptions ransac_opt;
    ransac_opt.max_epipolar_error = max_reproj_error;
    ransac_opt.min_iterations = 1000;
    ransac_opt.max_iterations = 100000;
    ransac_opt.success_prob = 0.9999;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::CAUCHY;
    bundle_opt.loss_scale = max_reproj_error * 0.5;
    bundle_opt.max_iterations = 1000;

    CameraPose pose;
    std::vector<char> inlier_mask;

    RansacStats stats = estimate_relative_pose(points2D_1, points2D_2, camera1, camera2, ransac_opt, bundle_opt, &pose, &inlier_mask);

    if(stats.num_inliers == 0) {
        py::dict failure_dict;
        failure_dict["success"] = false;
        return failure_dict;
    }

    // Convert vector<char> to vector<bool>.
    std::vector<bool> inliers(inlier_mask.size());    
    for(size_t pt_k = 0; pt_k < inlier_mask.size(); ++pt_k) {
        inliers[pt_k] = static_cast<bool>(inlier_mask[pt_k]);
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["success"] = true;
    success_dict["pose"] = pose;    
    success_dict["inliers"] = inliers;
    success_dict["num_inliers"] = stats.num_inliers;
    success_dict["iterations"] = stats.iterations;
    success_dict["inlier_ratio"] = stats.inlier_ratio;
    success_dict["refinements"] = stats.refinements;
    success_dict["model_score"] = stats.model_score;

    return success_dict;
}


py::dict estimate_fundamental_wrapper(const std::vector<Eigen::Vector2d> points2D_1, const std::vector<Eigen::Vector2d> points2D_2, const double max_epipolar_error){
       
    // Options chosen to be similar to pycolmap
    RansacOptions ransac_opt;
    ransac_opt.max_epipolar_error = max_epipolar_error;
    ransac_opt.min_iterations = 1000;
    ransac_opt.max_iterations = 100000;
    ransac_opt.success_prob = 0.9999;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::CAUCHY;
    bundle_opt.loss_scale = max_epipolar_error * 0.5;
    bundle_opt.max_iterations = 1000;

    Eigen::Matrix3d F;
    std::vector<char> inlier_mask;

    RansacStats stats = estimate_fundamental(points2D_1, points2D_2, ransac_opt, bundle_opt, &F, &inlier_mask);

    if(stats.num_inliers == 0) {
        py::dict failure_dict;
        failure_dict["success"] = false;
        return failure_dict;
    }

    // Convert vector<char> to vector<bool>.
    std::vector<bool> inliers(inlier_mask.size());    
    for(size_t pt_k = 0; pt_k < inlier_mask.size(); ++pt_k) {
        inliers[pt_k] = static_cast<bool>(inlier_mask[pt_k]);
    }

    // Success output dictionary.
    py::dict success_dict;
    success_dict["success"] = true;
    success_dict["F"] = F;    
    success_dict["inliers"] = inliers;
    success_dict["num_inliers"] = stats.num_inliers;
    success_dict["iterations"] = stats.iterations;
    success_dict["inlier_ratio"] = stats.inlier_ratio;
    success_dict["refinements"] = stats.refinements;
    success_dict["model_score"] = stats.model_score;

    return success_dict;
}

py::dict estimate_generalized_relative_pose_wrapper(const std::vector<PairwiseMatches> matches,
                        const std::vector<CameraPose> &camera1_ext, const std::vector<py::dict> &cameras1_dict,
                        const std::vector<CameraPose> &camera2_ext, const std::vector<py::dict> &cameras2_dict, const double max_reproj_error){
    
    std::vector<Camera> cameras1, cameras2;
    for(size_t k = 0; k < cameras1_dict.size(); ++k) {
        cameras1.emplace_back();
        cameras1.back().model_id = Camera::id_from_string(cameras1_dict[k]["model"].cast<std::string>());    
        cameras1.back().width = cameras1_dict[k]["width"].cast<size_t>();
        cameras1.back().height = cameras1_dict[k]["height"].cast<size_t>();
        cameras1.back().params = cameras1_dict[k]["params"].cast<std::vector<double>>();
    }
    
    for(size_t k = 0; k < cameras2_dict.size(); ++k) {
        cameras2.emplace_back();
        cameras2.back().model_id = Camera::id_from_string(cameras2_dict[k]["model"].cast<std::string>());    
        cameras2.back().width = cameras2_dict[k]["width"].cast<size_t>();
        cameras2.back().height = cameras2_dict[k]["height"].cast<size_t>();
        cameras2.back().params = cameras2_dict[k]["params"].cast<std::vector<double>>();        
    }
    
    // Options chosen to be similar to pycolmap
    RansacOptions ransac_opt;
    ransac_opt.max_epipolar_error = max_reproj_error;
    ransac_opt.min_iterations = 1000;
    ransac_opt.max_iterations = 100000;
    ransac_opt.success_prob = 0.9999;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::CAUCHY;
    bundle_opt.loss_scale = max_reproj_error * 0.5;
    bundle_opt.max_iterations = 1000;

    CameraPose pose;
    std::vector<std::vector<char>> inlier_mask;

    RansacStats stats = estimate_generalized_relative_pose(matches, camera1_ext, cameras1, camera2_ext, cameras2, ransac_opt, bundle_opt, &pose, &inlier_mask);

    if(stats.num_inliers == 0) {
        py::dict failure_dict;
        failure_dict["success"] = false;
        return failure_dict;
    }

    
    // Convert vector<char> to vector<bool>.
    std::vector<std::vector<bool>> inliers(inlier_mask.size());    
    for(size_t match_k = 0; match_k < inliers.size(); ++match_k) {
        inliers[match_k].resize(inlier_mask[match_k].size());
        for(size_t pt_k = 0; pt_k < inlier_mask[match_k].size(); ++pt_k) {
            inliers[match_k][pt_k] = static_cast<bool>(inlier_mask[match_k][pt_k]);
        }
    }


    // Success output dictionary.
    py::dict success_dict;
    success_dict["success"] = true;
    success_dict["pose"] = pose;        
    success_dict["inliers"] = inliers;
    success_dict["num_inliers"] = stats.num_inliers;
    success_dict["iterations"] = stats.iterations;
    success_dict["inlier_ratio"] = stats.inlier_ratio;
    success_dict["refinements"] = stats.refinements;
    success_dict["model_score"] = stats.model_score;
    
    return success_dict;
}



py::dict estimate_hybrid_pose_wrapper(
                        const std::vector<Eigen::Vector2d> points2D, const std::vector<Eigen::Vector3d> points3D,
                        const std::vector<PairwiseMatches> matches_2D_2D,
                        const py::dict &camera_dict, 
                        const std::vector<CameraPose> &map_ext, const std::vector<py::dict> &map_camera_dicts,
                        const double max_reproj_error, const double max_epipolar_error){
    
    Camera camera;
    camera.model_id = Camera::id_from_string(camera_dict["model"].cast<std::string>());    
    camera.width = camera_dict["width"].cast<size_t>();
    camera.height = camera_dict["height"].cast<size_t>();
    camera.params = camera_dict["params"].cast<std::vector<double>>();

    std::vector<Camera> map_cameras;
    for(size_t k = 0; k < map_camera_dicts.size(); ++k) {
        map_cameras.emplace_back();
        map_cameras.back().model_id = Camera::id_from_string(map_camera_dicts[k]["model"].cast<std::string>());    
        map_cameras.back().width = map_camera_dicts[k]["width"].cast<size_t>();
        map_cameras.back().height = map_camera_dicts[k]["height"].cast<size_t>();
        map_cameras.back().params = map_camera_dicts[k]["params"].cast<std::vector<double>>();
    }
    
    // Options chosen to be similar to pycolmap
    RansacOptions ransac_opt;
    ransac_opt.max_reproj_error = max_reproj_error;
    ransac_opt.max_epipolar_error = max_epipolar_error;
    ransac_opt.min_iterations = 1000;
    ransac_opt.max_iterations = 100000;
    ransac_opt.success_prob = 0.9999;

    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::CAUCHY;
    bundle_opt.loss_scale = max_reproj_error * 0.5;
    bundle_opt.max_iterations = 1000;

    CameraPose pose;
    
    std::vector<char> inliers_mask_2d3d;
    std::vector<std::vector<char>> inliers_mask_2d2d;

    RansacStats stats = estimate_hybrid_pose(points2D, points3D, matches_2D_2D, camera, map_ext, map_cameras, ransac_opt, bundle_opt, &pose, &inliers_mask_2d3d, &inliers_mask_2d2d);

    if(stats.num_inliers == 0) {
        py::dict failure_dict;
        failure_dict["success"] = false;
        return failure_dict;
    }

    
    // Convert vector<char> to vector<bool>.
    std::vector<std::vector<bool>> inliers_2d2d(inliers_mask_2d2d.size());    
    for(size_t match_k = 0; match_k < inliers_mask_2d2d.size(); ++match_k) {
        inliers_2d2d[match_k].resize(inliers_mask_2d2d[match_k].size());
        for(size_t pt_k = 0; pt_k < inliers_mask_2d2d[match_k].size(); ++pt_k) {
            inliers_2d2d[match_k][pt_k] = static_cast<bool>(inliers_mask_2d2d[match_k][pt_k]);
        }
    }

    std::vector<bool> inliers_2d3d(inliers_mask_2d3d.size());    
    for(size_t pt_k = 0; pt_k < inliers_mask_2d3d.size(); ++pt_k) {
        inliers_2d3d[pt_k] = static_cast<bool>(inliers_mask_2d3d[pt_k]);
    }


    // Success output dictionary.
    py::dict success_dict;
    success_dict["success"] = true;
    success_dict["pose"] = pose;        
    success_dict["inliers"] = inliers_2d3d;
    success_dict["inliers_2D"] = inliers_2d2d;
    success_dict["num_inliers"] = stats.num_inliers;
    success_dict["iterations"] = stats.iterations;
    success_dict["inlier_ratio"] = stats.inlier_ratio;
    success_dict["refinements"] = stats.refinements;
    success_dict["model_score"] = stats.model_score;
    
    return success_dict;
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
  m.def("estimate_absolute_pose", &pose_lib::estimate_absolute_pose_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("camera_dict"), py::arg("max_reproj_error") = 12.0,  "Absolute pose estimation with non-linear refinement.");
  m.def("estimate_generalized_absolute_pose", &pose_lib::estimate_generalized_absolute_pose_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("camera_ext"), py::arg("camera_dicts"), py::arg("max_reproj_error") = 12.0,  "Generalized absolute pose estimation with non-linear refinement.");
  m.def("estimate_relative_pose", &pose_lib::estimate_relative_pose_wrapper, py::arg("points2D_1"), py::arg("points2D_2"), py::arg("camera1_dict"), py::arg("camera2_dict"), py::arg("max_epipolar_error") = 2.0,  "Relative pose estimation with non-linear refinement.");  
  m.def("estimate_fundamental", &pose_lib::estimate_fundamental_wrapper, py::arg("points2D_1"), py::arg("points2D_2"), py::arg("max_epipolar_error") = 2.0, "Fundamental matrix estimation with non-linear refinement. Note: if you have known intrinsics you should use estimate_relative_pose instead!");  
  m.def("estimate_generalized_relative_pose", &pose_lib::estimate_generalized_relative_pose_wrapper, py::arg("matches"), py::arg("camera1_ext"), py::arg("camera1_dict"), py::arg("camera2_ext"), py::arg("camera2_dict"), py::arg("max_epipolar_error") = 2.0,  "Generalized relative pose estimation with non-linear refinement.");  
  m.def("estimate_hybrid_pose", &pose_lib::estimate_hybrid_pose_wrapper, py::arg("points2D"), py::arg("points3D"), py::arg("matches_2D_2D"), py::arg("camera_dict"), py::arg("map_ext"), py::arg("map_camera_dicts"), py::arg("max_reproj_error") = 12.0, py::arg("max_epipolar_error") = 2.0, "Hybrid camera pose estimation (both 2D-3D and 2D-2D correspondences to the map) with non-linear refinement.");  
  m.attr("__version__") = std::string(POSELIB_VERSION);
  
}
