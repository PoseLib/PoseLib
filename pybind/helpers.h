#ifndef POSELIB_PYBIND_HELPERS_H_
#define POSELIB_PYBIND_HELPERS_H_
#include "pybind11_extension.h"

#include <PoseLib/poselib.h>
#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

static std::string toString(const Eigen::MatrixXd &mat) {
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

namespace poselib {
template <typename T> void update(const py::dict &input, const std::string &name, T &value) {
    if (input.contains(name)) {
        value = input[name.c_str()].cast<T>();
    }
}
template <> void update(const py::dict &input, const std::string &name, bool &value) {
    if (input.contains(name)) {
        py::object input_value = input[name.c_str()];
        value = (py::str(input_value).is(py::str(Py_True)));
    }
}

void update_ransac_options(const py::dict &input, RansacOptions &ransac_opt) {
    update(input, "max_iterations", ransac_opt.max_iterations);
    update(input, "min_iterations", ransac_opt.min_iterations);
    update(input, "dyn_num_trials_mult", ransac_opt.dyn_num_trials_mult);
    update(input, "success_prob", ransac_opt.success_prob);
    update(input, "max_reproj_error", ransac_opt.max_reproj_error);
    update(input, "max_epipolar_error", ransac_opt.max_epipolar_error);
    update(input, "seed", ransac_opt.seed);
    update(input, "progressive_sampling", ransac_opt.progressive_sampling);
    update(input, "max_prosac_iterations", ransac_opt.max_prosac_iterations);
    update(input, "real_focal_check", ransac_opt.real_focal_check);
}

void update_bundle_options(const py::dict &input, BundleOptions &bundle_opt) {
    update(input, "max_iterations", bundle_opt.max_iterations);
    update(input, "loss_scale", bundle_opt.loss_scale);
    update(input, "gradient_tol", bundle_opt.gradient_tol);
    update(input, "step_tol", bundle_opt.step_tol);
    update(input, "initial_lambda", bundle_opt.initial_lambda);
    update(input, "min_lambda", bundle_opt.min_lambda);
    update(input, "max_lambda", bundle_opt.max_lambda);
    update(input, "verbose", bundle_opt.verbose);
    if (input.contains("loss_type")) {
        std::string loss_type = input["loss_type"].cast<std::string>();
        for (char &c : loss_type)
            c = std::toupper(c);
        if (loss_type == "TRIVIAL") {
            bundle_opt.loss_type = BundleOptions::LossType::TRIVIAL;
        } else if (loss_type == "TRUNCATED") {
            bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
        } else if (loss_type == "HUBER") {
            bundle_opt.loss_type = BundleOptions::LossType::HUBER;
        } else if (loss_type == "CAUCHY") {
            bundle_opt.loss_type = BundleOptions::LossType::CAUCHY;
        } else if (loss_type == "TRUNCATED_LE_ZACH") {
            bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED_LE_ZACH;
        }
    }
}

void write_to_dict(const RansacOptions &ransac_opt, py::dict &dict) {
    dict["max_iterations"] = ransac_opt.max_iterations;
    dict["min_iterations"] = ransac_opt.min_iterations;
    dict["dyn_num_trials_mult"] = ransac_opt.dyn_num_trials_mult;
    dict["success_prob"] = ransac_opt.success_prob;
    dict["max_reproj_error"] = ransac_opt.max_reproj_error;
    dict["max_epipolar_error"] = ransac_opt.max_epipolar_error;
    dict["seed"] = ransac_opt.seed;
    dict["progressive_sampling"] = ransac_opt.progressive_sampling;
    dict["max_prosac_iterations"] = ransac_opt.max_prosac_iterations;
    dict["real_focal_check"] = ransac_opt.real_focal_check;
}

void write_to_dict(const BundleOptions &bundle_opt, py::dict &dict) {
    dict["max_iterations"] = bundle_opt.max_iterations;
    dict["loss_scale"] = bundle_opt.loss_scale;
    switch (bundle_opt.loss_type) {
    default:
    case BundleOptions::LossType::TRIVIAL:
        dict["loss_type"] = "TRIVIAL";
        break;
    case BundleOptions::LossType::TRUNCATED:
        dict["loss_type"] = "TRUNCATED";
        break;
    case BundleOptions::LossType::HUBER:
        dict["loss_type"] = "HUBER";
        break;
    case BundleOptions::LossType::CAUCHY:
        dict["loss_type"] = "CAUCHY";
        break;
    case BundleOptions::LossType::TRUNCATED_LE_ZACH:
        dict["loss_type"] = "TRUNCATED_LE_ZACH";
        break;
    }
    dict["gradient_tol"] = bundle_opt.gradient_tol;
    dict["step_tol"] = bundle_opt.step_tol;
    dict["initial_lambda"] = bundle_opt.initial_lambda;
    dict["min_lambda"] = bundle_opt.min_lambda;
    dict["max_lambda"] = bundle_opt.max_lambda;
    dict["verbose"] = bundle_opt.verbose;
    ;
}

void write_to_dict(const BundleStats &stats, py::dict &dict) {
    dict["iterations"] = stats.iterations;
    dict["cost"] = stats.cost;
    dict["initial_cost"] = stats.initial_cost;
    dict["invalid_steps"] = stats.invalid_steps;
    dict["grad_norm"] = stats.grad_norm;
    dict["step_norm"] = stats.step_norm;
    dict["lambda"] = stats.lambda;
}

void write_to_dict(const RansacStats &stats, py::dict &dict) {
    dict["refinements"] = stats.refinements;
    dict["iterations"] = stats.iterations;
    dict["num_inliers"] = stats.num_inliers;
    dict["inlier_ratio"] = stats.inlier_ratio;
    dict["model_score"] = stats.model_score;
}

Camera camera_from_dict(const py::dict &camera_dict) {
    Camera camera;
    camera.model_id = Camera::id_from_string(camera_dict["model"].cast<std::string>());

    update(camera_dict, "width", camera.width);
    update(camera_dict, "height", camera.height);

    camera.params = camera_dict["params"].cast<std::vector<double>>();
    return camera;
}

std::vector<bool> convert_inlier_vector(const std::vector<char> &inliers) {
    std::vector<bool> inliers_bool(inliers.size());
    for (size_t k = 0; k < inliers.size(); ++k) {
        inliers_bool[k] = static_cast<bool>(inliers[k]);
    }
    return inliers_bool;
}

std::vector<std::vector<bool>> convert_inlier_vectors(const std::vector<std::vector<char>> &inliers) {
    std::vector<std::vector<bool>> inliers_bool(inliers.size());
    for (size_t cam_k = 0; cam_k < inliers.size(); ++cam_k) {
        inliers_bool[cam_k].resize(inliers[cam_k].size());
        for (size_t pt_k = 0; pt_k < inliers[cam_k].size(); ++pt_k) {
            inliers_bool[cam_k][pt_k] = static_cast<bool>(inliers[cam_k][pt_k]);
        }
    }
    return inliers_bool;
}

} // namespace poselib

#endif