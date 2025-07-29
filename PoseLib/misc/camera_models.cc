// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "camera_models.h"

#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>

namespace poselib {

static const double EPSILON = 1e-12;
static const double UNDIST_TOL = 1e-10;
static const size_t UNDIST_MAX_ITER = 100;

///////////////////////////////////////////////////////////////////
// Camera - base class storing ID

Camera::Camera() : Camera(-1, {}, 0, 0) { init_params(); }

Camera::Camera(int id) : Camera(id, {}, 0, 0) { init_params(); }

Camera::Camera(const std::string &model_name, const std::vector<double> &p, int w, int h)
    : Camera(id_from_string(model_name), p, w, h) {
    init_params();
}

Camera::Camera(int id, const std::vector<double> &p, int w, int h) {
    model_id = id;
    params = p;
    width = w;
    height = h;
    init_params();
}
Camera::Camera(const std::string &init_txt) {
    size_t id = id_from_string(init_txt);
    if (id == CameraModelId::INVALID) {
        initialize_from_txt(init_txt);
    } else {
        model_id = id;
        width = 0;
        height = 0;
    }
    init_params();
}

void Camera::init_params() {
    bool init_params = params.size() == 0;

#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        params.resize(Model::num_params, 0.0);                                                                         \
        if (init_params) {                                                                                             \
            if (width > 0 && height > 0) {                                                                             \
                set_focal(max_dim() * 1.2);                                                                            \
                set_principal_point(width / 2.0, height / 2.0);                                                        \
            } else {                                                                                                   \
                set_focal(1.0);                                                                                        \
                set_principal_point(0.0, 0.0);                                                                         \
            }                                                                                                          \
        }                                                                                                              \
        break;
    switch (model_id) { SWITCH_CAMERA_MODELS }

#undef SWITCH_CAMERA_MODEL_CASE
}

void Camera::set_focal(double f) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        for (size_t k : Model::focal_idx) {                                                                            \
            params[k] = f;                                                                                             \
        }                                                                                                              \
        break;

    switch (model_id) { SWITCH_CAMERA_MODELS }

#undef SWITCH_CAMERA_MODEL_CASE
}

void Camera::set_principal_point(double cx, double cy) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        if (Model::principal_point_idx.size() == 2) {                                                                  \
            params[Model::principal_point_idx[0]] = cx;                                                                \
            params[Model::principal_point_idx[1]] = cy;                                                                \
        }                                                                                                              \
        break;

    switch (model_id) { SWITCH_CAMERA_MODELS }

#undef SWITCH_CAMERA_MODEL_CASE
}

int Camera::id_from_string(const std::string &model_name) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    if (model_name == Model::to_string()) {                                                                            \
        return Model::model_id;                                                                                        \
    }
    SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE

    return CameraModelId::INVALID;
}

std::string Camera::name_from_id(int model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        return Model::to_string();
    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        return "INVALID_MODEL";
    }
#undef SWITCH_CAMERA_MODEL_CASE
}

// Projection and distortion
void Camera::project(const Eigen::Vector3d &x, Eigen::Vector2d *xp) const {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        Model::project(params, x, xp);                                                                                 \
        break;
    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        throw std::runtime_error("NYI");
    }
#undef SWITCH_CAMERA_MODEL_CASE
}
void Camera::project_with_jac(const Eigen::Vector3d &x, Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                              Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) const {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        Model::project_with_jac(params, x, xp, jac, jac_params);                                                       \
        break;
    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        throw std::runtime_error("NYI");
    }
#undef SWITCH_CAMERA_MODEL_CASE
}
void Camera::unproject(const Eigen::Vector2d &xp, Eigen::Vector3d *x) const {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        Model::unproject(params, xp, x);                                                                               \
        break;
    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        throw std::runtime_error("NYI");
    }
#undef SWITCH_CAMERA_MODEL_CASE
}

void Camera::unproject_with_jac(const Eigen::Vector2d &xp, Eigen::Vector3d *x, Eigen::Matrix<double, 3, 2> *jac,
                                Eigen::Matrix<double, 3, Eigen::Dynamic> *jac_params) const {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        Model::unproject_with_jac(params, xp, x, jac, jac_params);                                                     \
        break;
    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        throw std::runtime_error("NYI");
    }
#undef SWITCH_CAMERA_MODEL_CASE
}

void Camera::project(const std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector2d> *xp) const {
    xp->resize(x.size());
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        for (size_t i = 0; i < x.size(); ++i) {                                                                        \
            Model::project(params, x[i], &((*xp)[i]));                                                                 \
        }                                                                                                              \
        break;

    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        throw std::runtime_error("NYI");
    }
#undef SWITCH_CAMERA_MODEL_CASE
}
void Camera::project_with_jac(const std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector2d> *xp,
                              std::vector<Eigen::Matrix<double, 2, 3>> *jac,
                              std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic>> *jac_params) const {
    xp->resize(x.size());
    if (jac)
        jac->resize(x.size());
    if (jac_params)
        jac_params->resize(x.size());
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        if (jac_params) {                                                                                              \
            jac_params->resize(x.size());                                                                              \
            for (size_t i = 0; i < x.size(); ++i) {                                                                    \
                Model::project_with_jac(params, x[i], &((*xp)[i]), &((*jac)[i]), &((*jac_params)[i]));                 \
            }                                                                                                          \
        } else {                                                                                                       \
            for (size_t i = 0; i < x.size(); ++i) {                                                                    \
                Model::project_with_jac(params, x[i], &((*xp)[i]), &((*jac)[i]));                                      \
            }                                                                                                          \
        }                                                                                                              \
        break;

    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        throw std::runtime_error("NYI");
    }
#undef SWITCH_CAMERA_MODEL_CASE
}

void Camera::unproject(const std::vector<Eigen::Vector2d> &xp, std::vector<Eigen::Vector3d> *x) const {
    x->resize(xp.size());
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        for (size_t i = 0; i < xp.size(); ++i) {                                                                       \
            Model::unproject(params, xp[i], &((*x)[i]));                                                               \
        }                                                                                                              \
        break;

    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        throw std::runtime_error("NYI");
    }
#undef SWITCH_CAMERA_MODEL_CASE
}

void Camera::unproject_with_jac(const std::vector<Eigen::Vector2d> &xp, std::vector<Eigen::Vector3d> *x,
                                std::vector<Eigen::Matrix<double, 3, 2>> *jac,
                                std::vector<Eigen::Matrix<double, 3, Eigen::Dynamic>> *jac_params) const {
    x->resize(xp.size());
    if (jac)
        jac->resize(xp.size());
    if (jac_params)
        jac_params->resize(xp.size());
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        if (jac_params) {                                                                                              \
            jac_params->resize(xp.size());                                                                             \
            for (size_t i = 0; i < xp.size(); ++i) {                                                                   \
                Model::unproject_with_jac(params, xp[i], &((*x)[i]), &((*jac)[i]), &((*jac_params)[i]));               \
            }                                                                                                          \
        } else {                                                                                                       \
            for (size_t i = 0; i < xp.size(); ++i) {                                                                   \
                Model::unproject_with_jac(params, xp[i], &((*x)[i]), &((*jac)[i]));                                    \
            }                                                                                                          \
        }                                                                                                              \
        break;

    switch (model_id) {
        SWITCH_CAMERA_MODELS

    default:
        throw std::runtime_error("NYI");
    }
#undef SWITCH_CAMERA_MODEL_CASE
}

std::string Camera::model_name() const { return name_from_id(model_id); }

double Camera::focal() const {
    if (params.empty()) {
        return 1.0; // empty camera assumed to be identity
    }

    double focal = 0.0;
    switch (model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        for (size_t idx : Model::focal_idx)                                                                            \
            focal += params.at(idx) / Model::focal_idx.size();                                                         \
        break;

        SWITCH_CAMERA_MODELS
    }
#undef SWITCH_CAMERA_MODEL_CASE
    return focal;
}

double Camera::focal_x() const {
    if (params.empty()) {
        return 1.0; // empty camera assumed to be identity
    }

    switch (model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        if (Model::focal_idx.size() == 0) {                                                                            \
            return 0.0;                                                                                                \
        } else {                                                                                                       \
            return params.at(Model::focal_idx[0]);                                                                     \
        }

        SWITCH_CAMERA_MODELS
    }
#undef SWITCH_CAMERA_MODEL_CASE
    return -1.0;
}
double Camera::focal_y() const {
    if (params.empty()) {
        return 1.0; // empty camera assumed to be identity
    }

    switch (model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        if (Model::focal_idx.size() == 0) {                                                                            \
            return 0.0;                                                                                                \
        } else if (Model::focal_idx.size() > 1) {                                                                      \
            return params.at(Model::focal_idx[1]);                                                                     \
        } else {                                                                                                       \
            return params.at(Model::focal_idx[0]);                                                                     \
        }

        SWITCH_CAMERA_MODELS
    }
#undef SWITCH_CAMERA_MODEL_CASE
    return -1.0;
}

Eigen::Vector2d Camera::principal_point() const {
    if (params.empty()) {
        return Eigen::Vector2d(0.0, 0.0);
    }
    switch (model_id) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        if (Model::principal_point_idx.size() == 0) {                                                                  \
            return Eigen::Vector2d(0.0, 0.0);                                                                          \
        } else {                                                                                                       \
            return Eigen::Vector2d(params.at(Model::principal_point_idx[0]),                                           \
                                   params.at(Model::principal_point_idx[1]));                                          \
        }

        SWITCH_CAMERA_MODELS
    }
#undef SWITCH_CAMERA_MODEL_CASE
    return Eigen::Vector2d(-1.0, -1.0);
}

Eigen::Matrix3d Camera::calib_matrix() const {
    Eigen::Matrix3d K;
    K.setIdentity();
    K(0, 0) = focal_x();
    K(1, 1) = focal_y();
    K.block<2, 1>(0, 2) = principal_point();
    return K;
}

// Update the camera parameters such that the projections are rescaled
void Camera::rescale(double scale) {
    if (params.empty()) {
        return;
    }

#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        for (size_t idx : Model::focal_idx)                                                                            \
            params.at(idx) *= scale;                                                                                   \
        for (size_t idx : Model::principal_point_idx)                                                                  \
            params.at(idx) *= scale;                                                                                   \
        break;

    switch (model_id) { SWITCH_CAMERA_MODELS }
#undef SWITCH_CAMERA_MODEL_CASE
}

#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    void Model::unproject_with_jac(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x,   \
                                   Eigen::Matrix<double, 3, 2> *jac_point,                                             \
                                   Eigen::Matrix<double, 3, Eigen::Dynamic> *jac_params) {                             \
        unproject(params, xp, x);                                                                                      \
        if (!jac_point and !jac_params) {                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        Eigen::Matrix<double, 2, 3> jac_proj;                                                                          \
        Eigen::Matrix<double, 2, Eigen::Dynamic> jac_proj_params;                                                      \
                                                                                                                       \
        Eigen::Vector2d xp_proj;                                                                                       \
        Eigen::Matrix2d B_inv;                                                                                         \
        if (jac_params)                                                                                                \
            Model::project_with_jac(params, *x, &xp_proj, &jac_proj, &jac_proj_params);                                \
        else                                                                                                           \
            Model::project_with_jac(params, *x, &xp_proj, &jac_proj);                                                  \
                                                                                                                       \
        Eigen::Matrix2d B = jac_proj * jac_proj.transpose();                                                           \
        double det = B(0, 0) * B(1, 1) - B(0, 1) * B(1, 0);                                                            \
        B_inv << B(1, 1), -B(0, 1), -B(1, 0), B(0, 0);                                                                 \
        B_inv /= det;                                                                                                  \
                                                                                                                       \
        if (jac_point) {                                                                                               \
            *jac_point = jac_proj.transpose() * B_inv;                                                                 \
            if (jac_params) {                                                                                          \
                *jac_params = -*jac_point * jac_proj_params;                                                           \
            }                                                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
        if (jac_params) {                                                                                              \
            *jac_params = -jac_proj.transpose() * B_inv * jac_proj_params;                                             \
        }                                                                                                              \
    }

SWITCH_CAMERA_MODELS_DEFAULT_UNPROJECT_WITH_JAC

#undef SWITCH_CAMERA_MODEL_CASE

int Camera::initialize_from_txt(const std::string &line) {
    std::stringstream ss(line);

    // We might not have a camea id
    int camera_id;
    if (std::isdigit(line[0])) {
        ss >> camera_id;
    } else {
        camera_id = -1;
    }

    // Read the model
    std::string model_name;
    ss >> model_name;
    model_id = id_from_string(model_name);
    if (model_id == -1) {
        return -1;
    }

    // Read sizes
    double d;
    ss >> d;
    width = d;
    ss >> d;
    height = d;

    // Read parameters
    params.clear();
    double param;
    while (ss >> param) {
        params.push_back(param);
    }

    return camera_id;
}
std::string Camera::to_cameras_txt(int camera_id) const {
    std::stringstream ss;
    if (camera_id != -1) {
        ss << camera_id << " ";
    }
    ss << model_name();
    ss << " " << width;
    ss << " " << height;
    ss << std::setprecision(16);
    for (double d : params) {
        ss << " " << d;
    }
    return ss.str();
}

std::vector<size_t> Camera::get_param_refinement_idx(const BundleOptions &opt) {
    std::vector<size_t> idx;
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        if (opt.refine_focal_length)                                                                                   \
            idx.insert(idx.end(), Model::focal_idx.begin(), Model::focal_idx.end());                                   \
        if (opt.refine_principal_point)                                                                                \
            idx.insert(idx.end(), Model::principal_point_idx.begin(), Model::principal_point_idx.end());               \
        if (opt.refine_extra_params)                                                                                   \
            idx.insert(idx.end(), Model::extra_idx.begin(), Model::extra_idx.end());                                   \
        break;

    switch (model_id) { SWITCH_CAMERA_MODELS }
#undef SWITCH_CAMERA_MODEL_CASE
    return idx;
}

std::string Camera::params_info() const {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        return Model::params_info();                                                                                   \
        break;

    switch (model_id) { SWITCH_CAMERA_MODELS }
#undef SWITCH_CAMERA_MODEL_CASE
    return "INVALID_MODEL";
}

//  xp = f * d(r) * x
//  J = f * d'(r) * Jr + f * d(r)
// r = |x|, Jr = x / |x|

// Solves
//   rd = (1+k1 * r*r) * r
double undistort_poly1(double k1, double rd) {
    // f  = k1 * r^3 + r + 1 - rd = 0
    // fp = 3 * k1 * r^2 + 1
    double r = rd;
    for (size_t iter = 0; iter < UNDIST_MAX_ITER; ++iter) {
        double r2 = r * r;
        double f = k1 * r2 * r + r - rd;
        if (std::abs(f) < UNDIST_TOL) {
            break;
        }
        double fp = 3.0 * k1 * r2 + 1.0;
        r = r - f / fp;
    }
    return r;
}

// Solves
//   rd = (1+ k1 * r^2 + k2 * r^4) * r
double undistort_poly2(double k1, double k2, double rd) {
    // f  = k2 * r^5 + k1 * r^3 + r + 1 - rd = 0
    // fp = 5 * k2 * r^4 + 3 * k1 * r^2 + 1
    double r = rd;
    for (size_t iter = 0; iter < UNDIST_MAX_ITER; ++iter) {
        double r2 = r * r;
        double f = k2 * r2 * r2 * r + k1 * r2 * r + r - rd;
        if (std::abs(f) < UNDIST_TOL) {
            break;
        }
        double fp = 5.0 * k2 * r2 * r2 + 3.0 * k1 * r2 + 1.0;
        r = r - f / fp;
    }
    return r;
}

double undistort_theta_poly_newton(const std::vector<double> &params, int k0, double rd, double &theta) {
    double f, fp;
    for (size_t iter = 0; iter < UNDIST_MAX_ITER; iter++) {
        f = 1.0;
        fp = 1.0;
        double theta2 = theta * theta;
        double theta2k = theta2;
        double factor = 3.0;
        for (size_t k = k0; k < params.size(); ++k) {
            const double t = theta2k * params[k];
            f += t;
            fp += factor * t;
            theta2k *= theta2;
            factor += 2.0;
        }
        f *= theta;
        f -= rd;

        if (std::abs(f) < UNDIST_TOL) {
            return std::abs(f);
        }
        theta = theta - f / fp;
    }
    return std::abs(f);
}

double undistort_theta_poly(const std::vector<double> &params, int k0, double rd) {
    // try zero-init first
    double theta = 0;
    double res = undistort_theta_poly_newton(params, k0, rd, theta);
    if (res > UNDIST_TOL || theta < 0) {
        // If this fails try to initialize with rho (first order approx.)
        theta = rd;
        res = undistort_theta_poly_newton(params, k0, rd, theta);

        if (res > UNDIST_TOL || theta < 0) {
            // try once more
            theta = 0.5 * rd;
            res = undistort_theta_poly_newton(params, k0, rd, theta);

            if (res > UNDIST_TOL || theta < 0) {
                // try once more
                theta = 1.5 * rd;
                res = undistort_theta_poly_newton(params, k0, rd, theta);
                // if this does not work, just fail silently... yay
            }
        }
    }
    return theta;
}

///////////////////////////////////////////////////////////////////
// Pinhole camera
// params = fx, fy, cx, cy

void PinholeCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    (*xp)(0) = params[0] * x(0) / x(2) + params[2];
    (*xp)(1) = params[1] * x(1) / x(2) + params[3];
}
void PinholeCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                          Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                          Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const double inv_z = 1.0 / x(2);
    const double px = params[0] * x(0) * inv_z;
    const double py = params[1] * x(1) * inv_z;
    (*xp)(0) = px + params[2];
    (*xp)(1) = py + params[3];
    if (jac != nullptr) {
        (*jac)(0, 0) = params[0] * inv_z;
        (*jac)(0, 1) = 0.0;
        (*jac)(0, 2) = -px * inv_z;
        (*jac)(1, 0) = 0.0;
        (*jac)(1, 1) = params[1] * inv_z;
        (*jac)(1, 2) = -py * inv_z;
    }
    if (jac_params != nullptr) {
        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = x(0) * inv_z;
        (*jac_params)(1, 0) = 0.0;
        (*jac_params)(0, 1) = 0.0;
        (*jac_params)(1, 1) = x(1) * inv_z;
        (*jac_params)(0, 2) = 1.0;
        (*jac_params)(1, 2) = 0.0;
        (*jac_params)(0, 3) = 0.0;
        (*jac_params)(1, 3) = 1.0;
    }
}
void PinholeCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    (*x)(0) = (xp(0) - params[2]) / params[0];
    (*x)(1) = (xp(1) - params[3]) / params[1];
    (*x)(2) = 1.0;
    x->normalize();
}
const size_t PinholeCameraModel::num_params = 4;
const std::string PinholeCameraModel::params_info() { return "fx, fy, cx, cy"; };
const std::vector<size_t> PinholeCameraModel::focal_idx = {0, 1};
const std::vector<size_t> PinholeCameraModel::principal_point_idx = {2, 3};
const std::vector<size_t> PinholeCameraModel::extra_idx = {};

///////////////////////////////////////////////////////////////////
// Simple Pinhole camera
// params = f, cx, cy

void SimplePinholeCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x,
                                       Eigen::Vector2d *xp) {
    (*xp)(0) = params[0] * x(0) / x(2) + params[1];
    (*xp)(1) = params[0] * x(1) / x(2) + params[2];
}
void SimplePinholeCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                                Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                                Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const double inv_z = 1.0 / x(2);
    const double px = params[0] * x(0) * inv_z;
    const double py = params[0] * x(1) * inv_z;
    (*xp)(0) = px + params[1];
    (*xp)(1) = py + params[2];

    if (jac) {
        (*jac)(0, 0) = params[0] * inv_z;
        (*jac)(0, 1) = 0.0;
        (*jac)(0, 2) = -px * inv_z;

        (*jac)(1, 0) = 0.0;
        (*jac)(1, 1) = params[0] * inv_z;
        (*jac)(1, 2) = -py * inv_z;
    }
    if (jac_params) {
        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = x(0) * inv_z;
        (*jac_params)(1, 0) = x(1) * inv_z;
        (*jac_params)(0, 1) = 1.0;
        (*jac_params)(1, 1) = 0.0;
        (*jac_params)(0, 2) = 0.0;
        (*jac_params)(1, 2) = 1.0;
    }
}
void SimplePinholeCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                         Eigen::Vector3d *x) {
    (*x)(0) = (xp(0) - params[1]) / params[0];
    (*x)(1) = (xp(1) - params[2]) / params[0];
    (*x)(2) = 1.0;
    x->normalize();
}

const size_t SimplePinholeCameraModel::num_params = 3;
const std::string SimplePinholeCameraModel::params_info() { return "f, cx, cy"; };
const std::vector<size_t> SimplePinholeCameraModel::focal_idx = {0};
const std::vector<size_t> SimplePinholeCameraModel::principal_point_idx = {1, 2};
const std::vector<size_t> SimplePinholeCameraModel::extra_idx = {};

///////////////////////////////////////////////////////////////////
// Radial camera
// params = f, cx, cy, k1, k2

void RadialCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    const Eigen::Vector2d xph = x.hnormalized();
    const double r2 = xph.squaredNorm();
    const double alpha = (1.0 + params[3] * r2 + params[4] * r2 * r2);
    (*xp)(0) = params[0] * alpha * xph(0) + params[1];
    (*xp)(1) = params[0] * alpha * xph(1) + params[2];
}
void RadialCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                         Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                         Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const double inv_z = 1.0 / x(2);
    const double px = x(0) * inv_z;
    const double py = x(1) * inv_z;
    const double r2 = px * px + py * py;
    const double alpha = (1.0 + params[3] * r2 + params[4] * r2 * r2);
    const double alphap = (2.0 * params[3] + 4.0 * params[4] * r2);
    if (jac) {
        Eigen::Matrix2d jac_d;
        jac_d(0, 0) = (alphap * px * px + alpha) * params[0];
        jac_d(0, 1) = (alphap * px * py) * params[0];
        jac_d(1, 0) = jac_d(0, 1);
        jac_d(1, 1) = (alphap * py * py + alpha) * params[0];

        (*jac)(0, 0) = inv_z;
        (*jac)(0, 1) = 0;
        (*jac)(0, 2) = -px * inv_z;
        (*jac)(1, 0) = 0;
        (*jac)(1, 1) = inv_z;
        (*jac)(1, 2) = -py * inv_z;
        *jac = jac_d * (*jac);
    }
    if (jac_params) {
        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = alpha * px;
        (*jac_params)(1, 0) = alpha * py;

        (*jac_params)(0, 1) = 1.0;
        (*jac_params)(1, 1) = 0.0;

        (*jac_params)(0, 2) = 0.0;
        (*jac_params)(1, 2) = 1.0;

        (*jac_params)(0, 3) = params[0] * r2 * px;
        (*jac_params)(1, 3) = params[0] * r2 * py;

        (*jac_params)(0, 4) = params[0] * r2 * r2 * px;
        (*jac_params)(1, 4) = params[0] * r2 * r2 * py;
    }

    (*xp)(0) = params[0] * alpha * px + params[1];
    (*xp)(1) = params[0] * alpha * py + params[2];
}
void RadialCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    (*x)(0) = (xp(0) - params[1]) / params[0];
    (*x)(1) = (xp(1) - params[2]) / params[0];
    (*x)(2) = 0.0;
    double r0 = x->norm();
    if (std::abs(r0) > 1e-8) {
        double r = undistort_poly2(params[3], params[4], r0);
        (*x) *= r / r0;
    } // we are very close to the principal axis (r/r0 = 1)
    (*x)(2) = 1.0;
    x->normalize();
}

const size_t RadialCameraModel::num_params = 5;
const std::string RadialCameraModel::params_info() { return "f, cx, cy, k1, k2"; };
const std::vector<size_t> RadialCameraModel::focal_idx = {0};
const std::vector<size_t> RadialCameraModel::principal_point_idx = {1, 2};
const std::vector<size_t> RadialCameraModel::extra_idx = {3, 4};

///////////////////////////////////////////////////////////////////
// Simple Radial camera
// params = f, cx, cy, k1

void SimpleRadialCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x,
                                      Eigen::Vector2d *xp) {
    const double inv_z = 1.0 / x(2);
    const double px = x(0) * inv_z;
    const double py = x(1) * inv_z;
    const double r2 = px * px + py * py;
    const double alpha = (1.0 + params[3] * r2);
    (*xp)(0) = params[0] * alpha * px + params[1];
    (*xp)(1) = params[0] * alpha * py + params[2];
}
void SimpleRadialCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                               Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                               Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const double inv_z = 1.0 / x(2);
    const double px = x(0) * inv_z;
    const double py = x(1) * inv_z;
    const double r2 = px * px + py * py;
    const double alpha = (1.0 + params[3] * r2);

    Eigen::Matrix2d jac_d;
    jac_d(0, 0) = (2.0 * params[3] * px * px + alpha) * params[0];
    jac_d(0, 1) = (2.0 * params[3] * px * py) * params[0];
    jac_d(1, 0) = jac_d(0, 1);
    jac_d(1, 1) = (2.0 * params[3] * py * py + alpha) * params[0];

    if (jac) {
        (*jac)(0, 0) = inv_z;
        (*jac)(0, 1) = 0;
        (*jac)(0, 2) = -px * inv_z;
        (*jac)(1, 0) = 0;
        (*jac)(1, 1) = inv_z;
        (*jac)(1, 2) = -py * inv_z;
        *jac = jac_d * (*jac);
    }

    if (jac_params) {
        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = alpha * px;
        (*jac_params)(1, 0) = alpha * py;

        (*jac_params)(0, 1) = 1.0;
        (*jac_params)(1, 1) = 0.0;

        (*jac_params)(0, 2) = 0.0;
        (*jac_params)(1, 2) = 1.0;

        (*jac_params)(0, 3) = params[0] * r2 * px;
        (*jac_params)(1, 3) = params[0] * r2 * py;
    }

    (*xp)(0) = params[0] * alpha * px + params[1];
    (*xp)(1) = params[0] * alpha * py + params[2];
}
void SimpleRadialCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                        Eigen::Vector3d *x) {
    (*x)(0) = (xp(0) - params[1]) / params[0];
    (*x)(1) = (xp(1) - params[2]) / params[0];
    (*x)(2) = 0.0;
    double r0 = x->norm();
    if (std::abs(r0) > 1e-8) {
        double r = undistort_poly1(params[3], r0);
        (*x) *= r / r0;
    } // else we are very close to distortion center (r / r0 = 1)
    (*x)(2) = 1.0;
    x->normalize();
}

const size_t SimpleRadialCameraModel::num_params = 4;
const std::string SimpleRadialCameraModel::params_info() { return "f, cx, cy, k"; };
const std::vector<size_t> SimpleRadialCameraModel::focal_idx = {0};
const std::vector<size_t> SimpleRadialCameraModel::principal_point_idx = {1, 2};
const std::vector<size_t> SimpleRadialCameraModel::extra_idx = {3};

///////////////////////////////////////////////////////////////////
// OpenCV camera
//   params = fx, fy, cx, cy, k1, k2, p1, p2

void compute_opencv_distortion(double k1, double k2, double p1, double p2, const Eigen::Vector2d &x,
                               Eigen::Vector2d &xp) {
    const double u = x(0);
    const double v = x(1);
    const double u2 = u * u;
    const double uv = u * v;
    const double v2 = v * v;
    const double r2 = u * u + v * v;
    const double alpha = 1.0 + k1 * r2 + k2 * r2 * r2;
    xp(0) = alpha * u + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2);
    xp(1) = alpha * v + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2);
}

void compute_opencv_distortion_jac(double k1, double k2, double p1, double p2, const Eigen::Vector2d &x,
                                   Eigen::Vector2d &xp, Eigen::Matrix2d &jac,
                                   Eigen::Matrix<double, 2, 4> *jacp = nullptr) {
    const double u = x(0);
    const double v = x(1);
    const double u2 = u * u;
    const double uv = u * v;
    const double v2 = v * v;
    const double r2 = u * u + v * v;
    jac(0, 0) = k2 * r2 * r2 + 6 * p2 * u + 2 * p1 * v + u * (2 * k1 * u + 4 * k2 * u * r2) + k1 * r2 + 1.0;
    jac(0, 1) = 2 * p1 * u + 2 * p2 * v + v * (2 * k1 * u + 4 * k2 * u * r2);
    jac(1, 0) = 2 * p1 * u + 2 * p2 * v + u * (2 * k1 * v + 4 * k2 * v * r2);
    jac(1, 1) = k2 * r2 * r2 + 2 * p2 * u + 6 * p1 * v + v * (2 * k1 * v + 4 * k2 * v * r2) + k1 * r2 + 1.0;

    const double alpha = 1.0 + k1 * r2 + k2 * r2 * r2;
    xp(0) = alpha * u + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2);
    xp(1) = alpha * v + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2);

    if (jacp) {
        (*jacp)(0, 0) = r2 * u;
        (*jacp)(1, 0) = r2 * v;

        (*jacp)(0, 1) = r2 * r2 * u;
        (*jacp)(1, 1) = r2 * r2 * v;

        (*jacp)(0, 2) = 2.0 * uv;
        (*jacp)(1, 2) = (r2 + 2.0 * v2);

        (*jacp)(0, 3) = (r2 + 2.0 * u2);
        (*jacp)(1, 3) = 2.0 * uv;
    }
}

void OpenCVCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    Eigen::Vector2d x0(x(0) / x(2), x(1) / x(2));
    compute_opencv_distortion(params[4], params[5], params[6], params[7], x0, *xp);
    (*xp)(0) = params[0] * (*xp)(0) + params[2];
    (*xp)(1) = params[1] * (*xp)(1) + params[3];
}

Eigen::Vector2d undistort_opencv(double k1, double k2, double p1, double p2, const Eigen::Vector2d &xp) {
    Eigen::Vector2d x = xp;
    Eigen::Vector2d xd;
    Eigen::Matrix2d jac;
    static const double lambda = 1e-8;
    for (size_t iter = 0; iter < UNDIST_MAX_ITER; ++iter) {
        compute_opencv_distortion_jac(k1, k2, p1, p2, x, xd, jac);
        jac(0, 0) += lambda;
        jac(1, 1) += lambda;
        Eigen::Vector2d res = xd - xp;

        if (res.norm() < UNDIST_TOL) {
            break;
        }

        x = x - jac.inverse() * res;
    }
    return x;
}

void OpenCVCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                         Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                         Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    Eigen::Vector2d x0(x(0) / x(2), x(1) / x(2));
    Eigen::Matrix<double, 2, 2> jac0;
    Eigen::Matrix<double, 2, 4> jac1;
    compute_opencv_distortion_jac(params[4], params[5], params[6], params[7], x0, *xp, jac0, &jac1);
    if (jac) {
        *jac << 1.0 / x(2), 0.0, -x0(0) / x(2), 0.0, 1.0 / x(2), -x0(1) / x(2);
        *jac = jac0 * (*jac);
        jac->row(0) *= params[0];
        jac->row(1) *= params[1];
    }
    if (jac_params) {
        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = (*xp)(0);
        (*jac_params)(1, 0) = 0.0;

        (*jac_params)(0, 1) = 0.0;
        (*jac_params)(1, 1) = (*xp)(1);

        (*jac_params)(0, 2) = 1.0;
        (*jac_params)(1, 2) = 0.0;

        (*jac_params)(0, 3) = 0.0;
        (*jac_params)(1, 3) = 1.0;

        jac_params->block<1, 4>(0, 4) = params[0] * jac1.row(0);
        jac_params->block<1, 4>(1, 4) = params[1] * jac1.row(1);
    }
    (*xp)(0) = params[0] * (*xp)(0) + params[2];
    (*xp)(1) = params[1] * (*xp)(1) + params[3];
}
void OpenCVCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    Eigen::Vector2d xp0;
    xp0 << (xp(0) - params[2]) / params[0], (xp(1) - params[3]) / params[1];
    Eigen::Vector2d x0;
    x0 = undistort_opencv(params[4], params[5], params[6], params[7], xp0);
    *x << x0(0), x0(1), 1.0;
    x->normalize();
}

const size_t OpenCVCameraModel::num_params = 8;
const std::string OpenCVCameraModel::params_info() { return "fx, fy, cx, cy, k1, k2, p1, p2"; };
const std::vector<size_t> OpenCVCameraModel::focal_idx = {0, 1};
const std::vector<size_t> OpenCVCameraModel::principal_point_idx = {2, 3};
const std::vector<size_t> OpenCVCameraModel::extra_idx = {4, 5, 6, 7};

///////////////////////////////////////////////////////////////////
// OpenCV Fisheye camera
//   params = fx, fy, cx, cy, k1, k2, k3, k4

void OpenCVFisheyeCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x,
                                       Eigen::Vector2d *xp) {
    double rho = x.topRows<2>().norm();

    if (rho > 1e-8) {
        double theta = std::atan2(rho, x(2));
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;
        double theta6 = theta2 * theta4;
        double theta8 = theta2 * theta6;

        double rd = theta * (1.0 + theta2 * params[4] + theta4 * params[5] + theta6 * params[6] + theta8 * params[7]);
        const double inv_r = 1.0 / rho;
        (*xp)(0) = params[0] * x(0) * inv_r * rd + params[2];
        (*xp)(1) = params[1] * x(1) * inv_r * rd + params[3];
    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[2];
        (*xp)(1) = params[1] * x(1) + params[3];
    }
}

void OpenCVFisheyeCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                                Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                                Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    double rho = x.topRows<2>().norm();

    if (rho > 1e-8) {
        double theta = std::atan2(rho, x(2));
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;
        double theta6 = theta2 * theta4;
        double theta8 = theta2 * theta6;

        double rd = theta * (1.0 + theta2 * params[4] + theta4 * params[5] + theta6 * params[6] + theta8 * params[7]);
        const double inv_r = 1.0 / rho;

        double drho_dx = x(0) / rho;
        double drho_dy = x(1) / rho;

        double rho_z2 = rho * rho + x(2) * x(2);
        double dtheta_drho = x(2) / rho_z2;
        double dtheta_dz = -rho / rho_z2;

        double drd_dtheta = (1.0 + 3.0 * theta2 * params[4] + 5.0 * theta4 * params[5] + 7.0 * theta6 * params[6] +
                             9.0 * theta8 * params[7]);
        double drd_dx = drd_dtheta * dtheta_drho * drho_dx;
        double drd_dy = drd_dtheta * dtheta_drho * drho_dy;
        double drd_dz = drd_dtheta * dtheta_dz;

        double dinv_r_drho = -1.0 / (rho * rho);
        double dinv_r_dx = dinv_r_drho * drho_dx;
        double dinv_r_dy = dinv_r_drho * drho_dy;

        (*xp)(0) = params[0] * x(0) * inv_r * rd + params[2];
        (*xp)(1) = params[1] * x(1) * inv_r * rd + params[3];

        if (jac) {
            (*jac)(0, 0) = params[0] * (inv_r * rd + x(0) * dinv_r_dx * rd + x(0) * inv_r * drd_dx);
            (*jac)(0, 1) = params[0] * x(0) * (dinv_r_dy * rd + inv_r * drd_dy);
            (*jac)(0, 2) = params[0] * x(0) * inv_r * drd_dz;
            (*jac)(1, 0) = params[1] * x(1) * (dinv_r_dx * rd + inv_r * drd_dx);
            (*jac)(1, 1) = params[1] * (inv_r * rd + x(1) * dinv_r_dy * rd + x(1) * inv_r * drd_dy);
            (*jac)(1, 2) = params[1] * x(1) * inv_r * drd_dz;
        }

        if (jac_params) {
            jac_params->resize(2, num_params);
            (*jac_params)(0, 0) = x(0) * inv_r * rd;
            (*jac_params)(1, 0) = 0.0;

            (*jac_params)(0, 1) = 0.0;
            (*jac_params)(1, 1) = x(1) * inv_r * rd;

            (*jac_params)(0, 2) = 1.0;
            (*jac_params)(1, 2) = 0.0;

            (*jac_params)(0, 3) = 0.0;
            (*jac_params)(1, 3) = 1.0;

            (*jac_params)(0, 4) = params[0] * x(0) * inv_r * theta * theta2;
            (*jac_params)(1, 4) = params[1] * x(1) * inv_r * theta * theta2;

            (*jac_params)(0, 5) = params[0] * x(0) * inv_r * theta * theta4;
            (*jac_params)(1, 5) = params[1] * x(1) * inv_r * theta * theta4;

            (*jac_params)(0, 6) = params[0] * x(0) * inv_r * theta * theta6;
            (*jac_params)(1, 6) = params[1] * x(1) * inv_r * theta * theta6;

            (*jac_params)(0, 7) = params[0] * x(0) * inv_r * theta * theta8;
            (*jac_params)(1, 7) = params[1] * x(1) * inv_r * theta * theta8;
        }
    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[2];
        (*xp)(1) = params[1] * x(1) + params[3];
        if (jac) {
            (*jac)(0, 0) = params[0];
            (*jac)(0, 1) = 0.0;
            (*jac)(0, 2) = 0.0;
            (*jac)(1, 0) = 0.0;
            (*jac)(1, 1) = params[1];
            (*jac)(1, 2) = 0.0;
        }
        if (jac_params) {
            jac_params->resize(2, num_params);
            jac_params->setZero();
            (*jac_params)(0, 0) = x(0);
            (*jac_params)(1, 1) = x(1);
            (*jac_params)(0, 2) = 1.0;
            (*jac_params)(1, 3) = 1.0;
        }
    }
}

double opencv_fisheye_newton(const std::vector<double> &params, double rd, double &theta) {
    double f;
    for (size_t iter = 0; iter < UNDIST_MAX_ITER; iter++) {
        const double theta2 = theta * theta;
        const double theta4 = theta2 * theta2;
        const double theta6 = theta2 * theta4;
        const double theta8 = theta2 * theta6;
        f = theta * (1.0 + theta2 * params[4] + theta4 * params[5] + theta6 * params[6] + theta8 * params[7]) - rd;
        if (std::abs(f) < UNDIST_TOL) {
            return std::abs(f);
        }
        const double fp = (1.0 + 3.0 * theta2 * params[4] + 5.0 * theta4 * params[5] + 7.0 * theta6 * params[6] +
                           9.0 * theta8 * params[7]);
        theta = theta - f / fp;
    }
    return std::abs(f);
}

void OpenCVFisheyeCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                         Eigen::Vector3d *x) {
    const double px = (xp(0) - params[2]) / params[0];
    const double py = (xp(1) - params[3]) / params[1];
    const double rd = std::sqrt(px * px + py * py);

    if (rd > 1e-8) {
        double theta = undistort_theta_poly(params, 4, rd);

        (*x)(0) = px / rd;
        (*x)(1) = py / rd;

        if (std::abs(theta - M_PI_2) > 1e-8) {
            (*x)(2) = 1.0 / std::tan(theta);
        } else {
            (*x)(2) = 0.0;
        }
    } else {
        (*x)(0) = px;
        (*x)(1) = py;
        (*x)(2) = std::sqrt(1 - rd * rd);
    }

    x->normalize();
}

const size_t OpenCVFisheyeCameraModel::num_params = 8;
const std::string OpenCVFisheyeCameraModel::params_info() { return "fx, fy, cx, cy, k1, k2, k3, k4"; };
const std::vector<size_t> OpenCVFisheyeCameraModel::focal_idx = {0, 1};
const std::vector<size_t> OpenCVFisheyeCameraModel::principal_point_idx = {2, 3};
const std::vector<size_t> OpenCVFisheyeCameraModel::extra_idx = {4, 5, 6, 7};

///////////////////////////////////////////////////////////////////
// Full OpenCV camera
//   params = fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

void compute_full_opencv_distortion(double k1, double k2, double p1, double p2, double k3, double k4, double k5,
                                    double k6, const Eigen::Vector2d &x, Eigen::Vector2d &xp) {
    const double u = x(0);
    const double v = x(1);
    const double u2 = u * u;
    const double uv = u * v;
    const double v2 = v * v;
    const double r2 = u * u + v * v;
    const double r4 = r2 * r2;
    const double r6 = r2 * r4;
    const double alpha = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
    xp(0) = alpha * u + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2);
    xp(1) = alpha * v + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2);
}

void compute_full_opencv_distortion_jac(double k1, double k2, double p1, double p2, double k3, double k4, double k5,
                                        double k6, const Eigen::Vector2d &x, Eigen::Vector2d &xp, Eigen::Matrix2d &jac,
                                        Eigen::Matrix<double, 2, 8> *jacp = nullptr) {
    const double u = x(0);
    const double v = x(1);
    const double u2 = u * u;
    const double uv = u * v;
    const double v2 = v * v;
    const double r2 = u * u + v * v;
    const double r4 = r2 * r2;
    const double r6 = r2 * r4;

    const double nn = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
    const double dd = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
    const double nn_r = 2.0 * k1 + 4.0 * k2 * r2 + 6.0 * k3 * r4;
    const double dd_r = 2.0 * k4 + 4.0 * k5 * r2 + 6.0 * k6 * r4;
    const double dd2 = dd * dd;

    jac(0, 0) = 6 * p2 * u + 2 * p1 * v + nn / dd + (u2 * nn_r) / dd - (nn * u2 * dd_r) / dd2;
    jac(0, 1) = 2 * p1 * u + 2 * p2 * v + (uv * nn_r) / dd - (nn * uv * dd_r) / dd2;
    jac(1, 0) = jac(0, 1);
    // jac(1,0) = 2*p1*u + 2*p2*v + (uv*nn_r)/dd - (nn*uv*dd_r)/dd^2;
    jac(1, 1) = 2 * p2 * u + 6 * p1 * v + nn / dd + (v2 * nn_r) / dd - (nn * v2 * dd_r) / dd2;

    const double alpha = nn / dd;
    xp(0) = alpha * u + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2);
    xp(1) = alpha * v + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2);

    if (jacp) {
        // k1
        (*jacp)(0, 0) = r2 / dd * u;
        (*jacp)(1, 0) = r2 / dd * v;

        // k2
        (*jacp)(0, 1) = r4 / dd * u;
        (*jacp)(1, 1) = r4 / dd * v;

        // p1
        (*jacp)(0, 2) = 2.0 * uv;
        (*jacp)(1, 2) = r2 + 2.0 * v2;

        // p2
        (*jacp)(0, 3) = r2 + 2.0 * u2;
        (*jacp)(1, 3) = 2.0 * uv;

        // k3
        (*jacp)(0, 4) = r6 / dd * u;
        (*jacp)(1, 4) = r6 / dd * v;

        // k4
        (*jacp)(0, 5) = -nn / dd2 * r2 * u;
        (*jacp)(1, 5) = -nn / dd2 * r2 * v;

        // k5
        (*jacp)(0, 6) = -nn / dd2 * r4 * u;
        (*jacp)(1, 6) = -nn / dd2 * r4 * v;

        // k6
        (*jacp)(0, 7) = -nn / dd2 * r6 * u;
        (*jacp)(1, 7) = -nn / dd2 * r6 * v;
    }
}

void FullOpenCVCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    Eigen::Vector2d x0(x(0) / x(2), x(1) / x(2));
    compute_full_opencv_distortion(params[4], params[5], params[6], params[7], params[8], params[9], params[10],
                                   params[11], x0, *xp);
    (*xp)(0) = params[0] * (*xp)(0) + params[2];
    (*xp)(1) = params[1] * (*xp)(1) + params[3];
}

Eigen::Vector2d undistort_full_opencv(double k1, double k2, double p1, double p2, double k3, double k4, double k5,
                                      double k6, const Eigen::Vector2d &xp) {
    Eigen::Vector2d x = xp;
    Eigen::Vector2d xd;
    Eigen::Matrix2d jac;
    static const double lambda = 1e-8;
    for (size_t iter = 0; iter < UNDIST_MAX_ITER; ++iter) {
        compute_full_opencv_distortion_jac(k1, k2, p1, p2, k3, k4, k5, k6, x, xd, jac);
        jac(0, 0) += lambda;
        jac(1, 1) += lambda;
        Eigen::Vector2d res = xd - xp;

        if (res.norm() < UNDIST_TOL) {
            break;
        }

        x = x - jac.inverse() * res;
    }
    return x;
}

void FullOpenCVCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                             Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                             Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    Eigen::Vector2d x0(x(0) / x(2), x(1) / x(2));
    Eigen::Matrix<double, 2, 2> jac0;
    Eigen::Matrix<double, 2, 8> jac1;

    compute_full_opencv_distortion_jac(params[4], params[5], params[6], params[7], params[8], params[9], params[10],
                                       params[11], x0, *xp, jac0, &jac1);
    if (jac) {
        *jac << 1.0 / x(2), 0.0, -x0(0) / x(2), 0.0, 1.0 / x(2), -x0(1) / x(2);
        *jac = jac0 * (*jac);
        jac->row(0) *= params[0];
        jac->row(1) *= params[1];
    }
    if (jac_params) {
        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = (*xp)(0);
        (*jac_params)(1, 0) = 0.0;

        (*jac_params)(0, 1) = 0.0;
        (*jac_params)(1, 1) = (*xp)(1);

        (*jac_params)(0, 2) = 1.0;
        (*jac_params)(1, 2) = 0.0;

        (*jac_params)(0, 3) = 0.0;
        (*jac_params)(1, 3) = 1.0;

        jac_params->block<1, 8>(0, 4) = params[0] * jac1.row(0);
        jac_params->block<1, 8>(1, 4) = params[1] * jac1.row(1);
    }
    (*xp)(0) = params[0] * (*xp)(0) + params[2];
    (*xp)(1) = params[1] * (*xp)(1) + params[3];
}
void FullOpenCVCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                      Eigen::Vector3d *x) {
    Eigen::Vector2d xp0;
    xp0 << (xp(0) - params[2]) / params[0], (xp(1) - params[3]) / params[1];
    Eigen::Vector2d x0;
    x0 = undistort_full_opencv(params[4], params[5], params[6], params[7], params[8], params[9], params[10], params[11],
                               xp0);
    *x << x0(0), x0(1), 1.0;
    x->normalize();
}

const size_t FullOpenCVCameraModel::num_params = 12;
const std::string FullOpenCVCameraModel::params_info() { return "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6"; };
const std::vector<size_t> FullOpenCVCameraModel::focal_idx = {0, 1};
const std::vector<size_t> FullOpenCVCameraModel::principal_point_idx = {2, 3};
const std::vector<size_t> FullOpenCVCameraModel::extra_idx = {4, 5, 6, 7, 8, 9, 10, 11};

///////////////////////////////////////////////////////////////////
// FOV camera
//   params = fx, fy, cx, cy, omega

void FOVCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    const double r = x.topRows<2>().norm();
    const double z = x(2);
    const double w = params[4];
    const double tan_wh = std::tan(w / 2.0);
    double factor;
    if (std::abs(w) < EPSILON) {
        const double w2 = w * w;
        const double z2 = z * z;
        factor = (-4 * r * r * w2 + w2 * z2 + 12 * z2) / (12.0 * z2 * z);
    } else if (std::abs(r) < EPSILON) {
        factor = (2 * tan_wh * (3 * z * z - 4 * r * r * tan_wh * tan_wh)) / (3 * w * z * z * z);
    } else {
        factor = std::atan2(2.0 * r * tan_wh, z) / r / w;
    }
    *xp = factor * x.topRows<2>();
    (*xp)(0) = params[0] * (*xp)(0) + params[2];
    (*xp)(1) = params[1] * (*xp)(1) + params[3];
}

void FOVCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp,
                                      Eigen::Matrix<double, 2, 3> *jac,
                                      Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const Eigen::Vector2d v = x.topRows<2>();
    const double r = v.norm();
    const double r2 = r * r;
    const double z = x(2);
    const double z2 = z * z;
    const double z3 = z2 * z;
    const double w = params[4];
    const double w2 = w * w;
    const double tan_wh = std::tan(w / 2.0);
    double factor;
    double dfactor_dw;

    if (std::abs(w) < EPSILON) {
        factor = (-4.0 * r * r * w2 + w2 * z2 + 12.0 * z2) / (12.0 * z2 * z);
        dfactor_dw = (-8.0 * r * r * w + 2.0 * w * z2) / (12.0 * z2 * z);

        const double dfactor_dr = -r * w2 / (3 * z2 * z);
        const double dfactor_dz =
            (2 * z * (w2 + 12.0)) / (12.0 * z2 * z) - (-4 * r * r * w2 + w2 * z2 + 12 * z2) / (4.0 * z2 * z2);

        if (jac) {
            (*jac).block<2, 2>(0, 0) =
                factor * Eigen::Matrix2d::Identity() + dfactor_dr * v * v.transpose() / (r + EPSILON);
            (*jac).col(2) = dfactor_dz * v;
        }

    } else if (std::abs(r) < EPSILON) {
        const double tan_wh2 = tan_wh * tan_wh;
        factor = (2 * tan_wh * (3 * z2 - 4 * r * r * tan_wh2)) / (3 * w * z2 * z);
        dfactor_dw = ((tan_wh2 + 1.0) * (z2 - 4 * r2 * tan_wh2)) / (w * z3) -
                     (2 * tan_wh * z2 - (8 * r2 * tan_wh2 * tan_wh) / 3) / (w2 * z3);

        const double dfactor_dr = -(16 * r * tan_wh * tan_wh2) / (3 * w * z2 * z);
        const double dfactor_dz = (2 * tan_wh * (4 * r2 * tan_wh2 - z2)) / (w * z2 * z2);

        if (jac) {
            (*jac).block<2, 2>(0, 0) =
                factor * Eigen::Matrix2d::Identity() + dfactor_dr * v * v.transpose() / (r + EPSILON);
            (*jac).col(2) = dfactor_dz * v;
        }
    } else {

        const double tan_wh2 = tan_wh * tan_wh;
        const double phi = std::atan2(2.0 * r * tan_wh, z) / w;
        factor = phi / r;

        const double denom = w * (z2 + 4 * r2 * tan_wh2);
        const double dphi_dr = 2 * z * tan_wh / denom;
        const double dphi_dz = -2 * r * tan_wh / denom;
        const double dphi_dw =
            (r * z * (tan_wh2 + 1.0)) / (w * (z * z + 4 * r * r * tan_wh2)) - std::atan2(2 * r * tan_wh, z) / (w * w);

        const double dfactor_dr = dphi_dr / r - phi / r2;
        const double dfactor_dz = dphi_dz / r;

        // d atan2(y, x) = (-y, x) / (x^2 + y^2)
        dfactor_dw = dphi_dw / r;

        if (jac) {
            (*jac).block<2, 2>(0, 0) = factor * Eigen::Matrix2d::Identity() + dfactor_dr * v * v.transpose() / r;
            (*jac).col(2) = dfactor_dz * v;
        }
    }

    if (jac) {
        jac->row(0) *= params[0];
        jac->row(1) *= params[1];
    }
    *xp = factor * v;
    if (jac_params) {
        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = (*xp)(0);
        (*jac_params)(1, 0) = 0.0;

        (*jac_params)(0, 1) = 0.0;
        (*jac_params)(1, 1) = (*xp)(1);

        (*jac_params)(0, 2) = 1.0;
        (*jac_params)(1, 2) = 0.0;

        (*jac_params)(0, 3) = 0.0;
        (*jac_params)(1, 3) = 1.0;

        (*jac_params)(0, 4) = params[0] * dfactor_dw * v(0);
        (*jac_params)(1, 4) = params[1] * dfactor_dw * v(1);
    }
    (*xp)(0) = params[0] * (*xp)(0) + params[2];
    (*xp)(1) = params[1] * (*xp)(1) + params[3];
}
void FOVCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    Eigen::Vector2d xp0;
    xp0 << (xp(0) - params[2]) / params[0], (xp(1) - params[3]) / params[1];
    const double r = xp0.norm();
    const double w = params[4];
    double a;
    if (std::abs(w) < EPSILON) {
        const double w2 = w * w;
        a = 1 - w / 12.0 - (r * r * w2) / 6.0;
    } else if (std::abs(r) < EPSILON) {
        const double w2 = w * w;
        a = (-r * r * w * w2 + 6 * w) / (12.0 * std::tan(w / 2.0));
    } else {
        a = std::sin(r * w) / 2.0 / r / std::tan(w / 2.0);
    }
    (*x) << xp0(0) * a, xp0(1) * a, std::cos(r * w);
    x->normalize();
}

const size_t FOVCameraModel::num_params = 5;
const std::string FOVCameraModel::params_info() { return "fx, fy, cx, cy, omega"; };
const std::vector<size_t> FOVCameraModel::focal_idx = {0, 1};
const std::vector<size_t> FOVCameraModel::principal_point_idx = {2, 3};
const std::vector<size_t> FOVCameraModel::extra_idx = {4};

///////////////////////////////////////////////////////////////////
// Simple Radial Fisheye
//   params = fx, fy, cx, cy, omega

void SimpleRadialFisheyeCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x,
                                             Eigen::Vector2d *xp) {
    double rho = x.topRows<2>().norm();
    if (rho > 1e-8) {
        double theta = std::atan2(rho, x(2));
        double theta2 = theta * theta;

        double rd = theta * (1.0 + theta2 * params[3]);
        const double inv_r = 1.0 / rho;

        (*xp)(0) = params[0] * x(0) * inv_r * rd + params[1];
        (*xp)(1) = params[0] * x(1) * inv_r * rd + params[2];

    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[1];
        (*xp)(1) = params[0] * x(1) + params[2];
    }
}

void SimpleRadialFisheyeCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                                      Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                                      Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    double rho = x.topRows<2>().norm();

    if (rho > 1e-8) {
        double theta = std::atan2(rho, x(2));
        double theta2 = theta * theta;

        double rd = theta * (1.0 + theta2 * params[3]);
        const double inv_r = 1.0 / rho;

        double drho_dx = x(0) / rho;
        double drho_dy = x(1) / rho;

        double rho_z2 = rho * rho + x(2) * x(2);
        double dtheta_drho = x(2) / rho_z2;
        double dtheta_dz = -rho / rho_z2;

        double drd_dtheta = (1.0 + 3.0 * theta2 * params[3]);
        double drd_dx = drd_dtheta * dtheta_drho * drho_dx;
        double drd_dy = drd_dtheta * dtheta_drho * drho_dy;
        double drd_dz = drd_dtheta * dtheta_dz;

        double dinv_r_drho = -1.0 / (rho * rho);
        double dinv_r_dx = dinv_r_drho * drho_dx;
        double dinv_r_dy = dinv_r_drho * drho_dy;

        (*xp)(0) = params[0] * x(0) * inv_r * rd + params[1];
        (*xp)(1) = params[0] * x(1) * inv_r * rd + params[2];

        if (jac) {
            (*jac)(0, 0) = params[0] * (inv_r * rd + x(0) * dinv_r_dx * rd + x(0) * inv_r * drd_dx);
            (*jac)(0, 1) = params[0] * x(0) * (dinv_r_dy * rd + inv_r * drd_dy);
            (*jac)(0, 2) = params[0] * x(0) * inv_r * drd_dz;
            (*jac)(1, 0) = params[0] * x(1) * (dinv_r_dx * rd + inv_r * drd_dx);
            (*jac)(1, 1) = params[0] * (inv_r * rd + x(1) * dinv_r_dy * rd + x(1) * inv_r * drd_dy);
            (*jac)(1, 2) = params[0] * x(1) * inv_r * drd_dz;
        }

        if (jac_params) {
            jac_params->resize(2, num_params);
            (*jac_params)(0, 0) = x(0) * inv_r * rd;
            (*jac_params)(1, 0) = x(1) * inv_r * rd;

            (*jac_params)(0, 1) = 1.0;
            (*jac_params)(1, 1) = 0.0;

            (*jac_params)(0, 2) = 0.0;
            (*jac_params)(1, 2) = 1.0;

            (*jac_params)(0, 3) = params[0] * x(0) * inv_r * theta * theta2;
            (*jac_params)(1, 3) = params[0] * x(1) * inv_r * theta * theta2;
        }
    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[1];
        (*xp)(1) = params[0] * x(1) + params[2];
        if (jac) {
            (*jac)(0, 0) = params[0];
            (*jac)(0, 1) = 0.0;
            (*jac)(0, 2) = 0.0;
            (*jac)(1, 0) = 0.0;
            (*jac)(1, 1) = params[0];
            (*jac)(1, 2) = 0.0;
        }
        if (jac_params) {
            jac_params->resize(2, num_params);
            jac_params->setZero();
        }
    }
}

double simple_radial_fisheye_newton(const std::vector<double> &params, double rd, double &theta) {
    double f;
    for (size_t iter = 0; iter < UNDIST_MAX_ITER; iter++) {
        const double theta2 = theta * theta;
        f = theta * (1.0 + theta2 * params[3]) - rd;
        if (std::abs(f) < UNDIST_TOL) {
            return std::abs(f);
        }
        const double fp = (1.0 + 3.0 * theta2 * params[4]);
        theta = theta - f / fp;
    }
    return std::abs(f);
}

void SimpleRadialFisheyeCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                               Eigen::Vector3d *x) {
    const double px = (xp(0) - params[1]) / params[0];
    const double py = (xp(1) - params[2]) / params[0];
    const double rd = std::sqrt(px * px + py * py);

    if (rd > 1e-8) {
        double theta = undistort_theta_poly(params, 3, rd);

        (*x)(0) = px / rd;
        (*x)(1) = py / rd;

        if (std::abs(theta - M_PI_2) > 1e-8) {
            (*x)(2) = 1.0 / std::tan(theta);
        } else {
            (*x)(2) = 0.0;
        }
    } else {
        (*x)(0) = px;
        (*x)(1) = py;
        (*x)(2) = std::sqrt(1 - rd * rd);
    }

    x->normalize();
}

const size_t SimpleRadialFisheyeCameraModel::num_params = 4;
const std::string SimpleRadialFisheyeCameraModel::params_info() { return "f, cx, cy, k"; };
const std::vector<size_t> SimpleRadialFisheyeCameraModel::focal_idx = {0};
const std::vector<size_t> SimpleRadialFisheyeCameraModel::principal_point_idx = {1, 2};
const std::vector<size_t> SimpleRadialFisheyeCameraModel::extra_idx = {3};

///////////////////////////////////////////////////////////////////
//  Radial Fisheye
//   params = f, cx, cy, k1, k2

void RadialFisheyeCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x,
                                       Eigen::Vector2d *xp) {
    double rho = x.topRows<2>().norm();
    if (rho > 1e-8) {
        double theta = std::atan2(rho, x(2));
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;

        double rd = theta * (1.0 + theta2 * params[3] + theta4 * params[4]);
        const double inv_r = 1.0 / rho;

        (*xp)(0) = params[0] * x(0) * inv_r * rd + params[1];
        (*xp)(1) = params[0] * x(1) * inv_r * rd + params[2];

    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[1];
        (*xp)(1) = params[0] * x(1) + params[2];
    }
}

void RadialFisheyeCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                                Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                                Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    double rho = x.topRows<2>().norm();

    if (rho > 1e-8) {
        double theta = std::atan2(rho, x(2));
        double theta2 = theta * theta;
        double theta4 = theta2 * theta2;

        double rd = theta * (1.0 + theta2 * params[3] + theta4 * params[4]);
        const double inv_r = 1.0 / rho;

        double drho_dx = x(0) / rho;
        double drho_dy = x(1) / rho;

        double rho_z2 = rho * rho + x(2) * x(2);
        double dtheta_drho = x(2) / rho_z2;
        double dtheta_dz = -rho / rho_z2;

        double drd_dtheta = (1.0 + 3.0 * theta2 * params[3] + 5.0 * theta4 * params[4]);
        double drd_dx = drd_dtheta * dtheta_drho * drho_dx;
        double drd_dy = drd_dtheta * dtheta_drho * drho_dy;
        double drd_dz = drd_dtheta * dtheta_dz;

        double dinv_r_drho = -1.0 / (rho * rho);
        double dinv_r_dx = dinv_r_drho * drho_dx;
        double dinv_r_dy = dinv_r_drho * drho_dy;

        (*xp)(0) = params[0] * x(0) * inv_r * rd + params[1];
        (*xp)(1) = params[0] * x(1) * inv_r * rd + params[2];

        if (jac) {
            (*jac)(0, 0) = params[0] * (inv_r * rd + x(0) * dinv_r_dx * rd + x(0) * inv_r * drd_dx);
            (*jac)(0, 1) = params[0] * x(0) * (dinv_r_dy * rd + inv_r * drd_dy);
            (*jac)(0, 2) = params[0] * x(0) * inv_r * drd_dz;
            (*jac)(1, 0) = params[0] * x(1) * (dinv_r_dx * rd + inv_r * drd_dx);
            (*jac)(1, 1) = params[0] * (inv_r * rd + x(1) * dinv_r_dy * rd + x(1) * inv_r * drd_dy);
            (*jac)(1, 2) = params[0] * x(1) * inv_r * drd_dz;
        }

        if (jac_params) {
            jac_params->resize(2, num_params);
            (*jac_params)(0, 0) = x(0) * inv_r * rd;
            (*jac_params)(1, 0) = x(1) * inv_r * rd;

            (*jac_params)(0, 1) = 1.0;
            (*jac_params)(1, 1) = 0.0;

            (*jac_params)(0, 2) = 0.0;
            (*jac_params)(1, 2) = 1.0;

            (*jac_params)(0, 3) = params[0] * x(0) * inv_r * theta * theta2;
            (*jac_params)(1, 3) = params[0] * x(1) * inv_r * theta * theta2;

            (*jac_params)(0, 4) = params[0] * x(0) * inv_r * theta * theta4;
            (*jac_params)(1, 4) = params[0] * x(1) * inv_r * theta * theta4;
        }
    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[1];
        (*xp)(1) = params[0] * x(1) + params[2];
        if (jac) {
            (*jac)(0, 0) = params[0];
            (*jac)(0, 1) = 0.0;
            (*jac)(0, 2) = 0.0;
            (*jac)(1, 0) = 0.0;
            (*jac)(1, 1) = params[0];
            (*jac)(1, 2) = 0.0;
        }
        if (jac_params) {
            jac_params->resize(2, num_params);
            jac_params->setZero();
        }
    }
}

void RadialFisheyeCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                         Eigen::Vector3d *x) {
    const double px = (xp(0) - params[1]) / params[0];
    const double py = (xp(1) - params[2]) / params[0];
    const double rd = std::sqrt(px * px + py * py);

    if (rd > 1e-8) {
        double theta = undistort_theta_poly(params, 3, rd);

        (*x)(0) = px / rd;
        (*x)(1) = py / rd;

        if (std::abs(theta - M_PI_2) > 1e-8) {
            (*x)(2) = 1.0 / std::tan(theta);
        } else {
            (*x)(2) = 0.0;
        }
    } else {
        (*x)(0) = px;
        (*x)(1) = py;
        (*x)(2) = std::sqrt(1 - rd * rd);
    }

    x->normalize();
}

const size_t RadialFisheyeCameraModel::num_params = 5;
const std::string RadialFisheyeCameraModel::params_info() { return "f, cx, cy, k1, k2"; };
const std::vector<size_t> RadialFisheyeCameraModel::focal_idx = {0};
const std::vector<size_t> RadialFisheyeCameraModel::principal_point_idx = {1, 2};
const std::vector<size_t> RadialFisheyeCameraModel::extra_idx = {3, 4};

///////////////////////////////////////////////////////////////////
// Thin Prism Fisheye Camera model
//   params = fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1

void compute_thinprismfisheye_distortion(const std::vector<double> &params, const Eigen::Vector2d &x,
                                         Eigen::Vector2d &xp) {
    const double k1 = params[4];
    const double k2 = params[5];
    const double p1 = params[6];
    const double p2 = params[7];
    const double k3 = params[8];
    const double k4 = params[9];
    const double sx1 = params[10];
    const double sy1 = params[11];

    const double u = x(0);
    const double v = x(1);
    const double u2 = u * u;
    const double uv = u * v;
    const double v2 = v * v;
    const double r2 = u * u + v * v;
    const double r4 = r2 * r2;
    const double r6 = r2 * r4;
    const double r8 = r4 * r4;
    const double alpha = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    xp(0) = alpha * u + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2;
    xp(1) = alpha * v + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2;
}

void compute_thinprismfisheye_distortion_jac(const std::vector<double> &params, const Eigen::Vector2d &x,
                                             Eigen::Vector2d &xp, Eigen::Matrix2d &jac,
                                             Eigen::Matrix<double, 2, 8> *jacp = nullptr) {
    const double k1 = params[4];
    const double k2 = params[5];
    const double p1 = params[6];
    const double p2 = params[7];
    const double k3 = params[8];
    const double k4 = params[9];
    const double sx1 = params[10];
    const double sy1 = params[11];

    const double u = x(0);
    const double v = x(1);
    const double u2 = u * u;
    const double uv = u * v;
    const double v2 = v * v;
    const double r2 = u * u + v * v;
    const double r4 = r2 * r2;
    const double r6 = r2 * r4;
    const double r8 = r4 * r4;
    const double alpha = 1.0 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
    xp(0) = alpha * u + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2) + sx1 * r2;
    xp(1) = alpha * v + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2) + sy1 * r2;

    jac(0, 0) = 1.0 + u * (2 * k1 * u + 4 * k2 * u * r2 + 6 * k3 * u * r4 + 8 * k4 * u * r6) + k2 * r4 + k3 * r6 +
                k4 * r8 + 6 * p2 * u + 2 * p1 * v + 2 * sx1 * u + k1 * r2;
    jac(0, 1) =
        u * (2 * k1 * v + 4 * k2 * v * r2 + 6 * k3 * v * r4 + 8 * k4 * v * r6) + 2 * p1 * u + 2 * p2 * v + 2 * sx1 * v;
    jac(1, 0) =
        v * (2 * k1 * u + 4 * k2 * u * r2 + 6 * k3 * u * r4 + 8 * k4 * u * r6) + 2 * p1 * u + 2 * p2 * v + 2 * sy1 * u;
    jac(1, 1) = 1.0 + v * (2 * k1 * v + 4 * k2 * v * r2 + 6 * k3 * v * r4 + 8 * k4 * v * r6) + k2 * r4 + k3 * r6 +
                k4 * r8 + 2 * p2 * u + 6 * p1 * v + 2 * sy1 * v + k1 * r2;

    if (jacp) {
        // k1
        (*jacp)(0, 0) = r2 * u;
        (*jacp)(1, 0) = r2 * v;

        // k2
        (*jacp)(0, 1) = r4 * u;
        (*jacp)(1, 1) = r4 * v;

        // p1
        (*jacp)(0, 2) = 2.0 * uv;
        (*jacp)(1, 2) = r2 + 2.0 * v2;

        // p2
        (*jacp)(0, 3) = r2 + 2.0 * u2;
        (*jacp)(1, 3) = 2.0 * uv;

        // k3
        (*jacp)(0, 4) = r6 * u;
        (*jacp)(1, 4) = r6 * v;

        // k4
        (*jacp)(0, 5) = r8 * u;
        (*jacp)(1, 5) = r8 * v;

        // sx1
        (*jacp)(0, 6) = r2;
        (*jacp)(1, 6) = 0.0;

        // sy1
        (*jacp)(0, 7) = 0.0;
        (*jacp)(1, 7) = r2;
    }
}

void ThinPrismFisheyeCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x,
                                          Eigen::Vector2d *xp) {
    double rho = x.topRows<2>().norm();
    if (rho > 1e-8) {
        double theta = std::atan2(rho, x(2));
        const double inv_r = 1.0 / rho;

        Eigen::Vector2d xp0;
        xp0(0) = x(0) * theta * inv_r;
        xp0(1) = x(1) * theta * inv_r;

        compute_thinprismfisheye_distortion(params, xp0, *xp);

        (*xp)(0) = params[0] * (*xp)(0) + params[2];
        (*xp)(1) = params[1] * (*xp)(1) + params[3];

    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[2];
        (*xp)(1) = params[1] * x(1) + params[3];
    }
}

Eigen::Vector2d undistort_thinprismfisheye(const std::vector<double> &params, const Eigen::Vector2d &xp) {
    Eigen::Vector2d x = xp;
    Eigen::Vector2d xd;
    Eigen::Matrix2d jac;
    Eigen::Vector2d res;
    static const double lambda = 1e-8;
    for (size_t iter = 0; iter < UNDIST_MAX_ITER; ++iter) {
        compute_thinprismfisheye_distortion_jac(params, x, xd, jac);
        jac(0, 0) += lambda;
        jac(1, 1) += lambda;
        res = xd - xp;

        if (res.norm() < UNDIST_TOL) {
            break;
        }

        x = x - jac.inverse() * res;
    }
    return x;
}

void ThinPrismFisheyeCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                                   Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                                   Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    Eigen::Matrix<double, 2, 2> jac0;
    Eigen::Matrix<double, 2, 8> jac1;

    double rho = x.topRows<2>().norm();
    if (rho > 1e-8) {
        double theta = std::atan2(rho, x(2));
        const double inv_r = 1.0 / rho;

        Eigen::Vector2d xp0;
        xp0(0) = x(0) * theta * inv_r;
        xp0(1) = x(1) * theta * inv_r;

        compute_thinprismfisheye_distortion_jac(params, xp0, *xp, jac0, &jac1);

        if (jac) {
            double drho_dx = x(0) / rho;
            double drho_dy = x(1) / rho;

            double rho_z2 = rho * rho + x(2) * x(2);
            double dtheta_drho = x(2) / rho_z2;
            double dtheta_dx = dtheta_drho * drho_dx;
            double dtheta_dy = dtheta_drho * drho_dy;
            double dtheta_dz = -rho / rho_z2;

            double dinv_r_drho = -1.0 / (rho * rho);
            double dinv_r_dx = dinv_r_drho * drho_dx;
            double dinv_r_dy = dinv_r_drho * drho_dy;
            // double dinv_r_dz = 0.0;

            (*jac)(0, 0) = theta * inv_r + x(0) * (dtheta_dx * inv_r + theta * dinv_r_dx);
            (*jac)(0, 1) = x(0) * (dtheta_dy * inv_r + theta * dinv_r_dy);
            (*jac)(0, 2) = x(0) * dtheta_dz * inv_r;

            (*jac)(1, 0) = x(1) * (dtheta_dx * inv_r + theta * dinv_r_dx);
            (*jac)(1, 1) = theta * inv_r + x(1) * (dtheta_dy * inv_r + theta * dinv_r_dy);
            (*jac)(1, 2) = x(1) * dtheta_dz * inv_r;

            (*jac) = jac0 * (*jac);
            jac->row(0) *= params[0];
            jac->row(1) *= params[1];
        }

        if (jac_params) {
            jac_params->resize(2, num_params);
            (*jac_params)(0, 0) = (*xp)(0);
            (*jac_params)(1, 0) = 0.0;

            (*jac_params)(0, 1) = 0.0;
            (*jac_params)(1, 1) = (*xp)(1);

            (*jac_params)(0, 2) = 1.0;
            (*jac_params)(1, 2) = 0.0;

            (*jac_params)(0, 3) = 0.0;
            (*jac_params)(1, 3) = 1.0;

            jac_params->block<1, 8>(0, 4) = params[0] * jac1.row(0);
            jac_params->block<1, 8>(1, 4) = params[1] * jac1.row(1);
        }

        (*xp)(0) = params[0] * (*xp)(0) + params[2];
        (*xp)(1) = params[1] * (*xp)(1) + params[3];
    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[2];
        (*xp)(1) = params[1] * x(1) + params[3];

        if (jac) {
            (*jac)(0, 0) = params[0];
            (*jac)(0, 1) = 0.0;
            (*jac)(0, 2) = 0.0;
            (*jac)(1, 0) = 0.0;
            (*jac)(1, 1) = params[1];
            (*jac)(1, 2) = 0.0;
        }
        if (jac_params) {
            jac_params->resize(2, num_params);
            jac_params->setZero();
            (*jac_params)(0, 0) = x(0);
            (*jac_params)(1, 1) = x(1);
            (*jac_params)(0, 2) = 1.0;
            (*jac_params)(1, 3) = 1.0;
        }
    }
}
void ThinPrismFisheyeCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                            Eigen::Vector3d *x) {
    const double px = (xp(0) - params[2]) / params[0];
    const double py = (xp(1) - params[3]) / params[1];

    Eigen::Vector2d xp_undist = undistort_thinprismfisheye(params, Eigen::Vector2d(px, py));
    const double theta = xp_undist.norm();
    double t_cos_t = theta * std::cos(theta);

    if (t_cos_t > 1e-8) {
        double scale = std::sin(theta) / t_cos_t;
        (*x)(0) = xp_undist(0) * scale;
        (*x)(1) = xp_undist(1) * scale;

        if (std::abs(theta - M_PI_2) > 1e-8) {
            (*x)(2) = 1.0;
        } else {
            (*x)(2) = 0.0;
        }
    } else {
        (*x)(0) = xp_undist(0);
        (*x)(1) = xp_undist(1);
        (*x)(2) = std::sqrt(1 - theta * theta);
    }
    x->normalize();
}

const size_t ThinPrismFisheyeCameraModel::num_params = 12;
const std::string ThinPrismFisheyeCameraModel::params_info() {
    return "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1";
};
const std::vector<size_t> ThinPrismFisheyeCameraModel::focal_idx = {0, 1};
const std::vector<size_t> ThinPrismFisheyeCameraModel::principal_point_idx = {2, 3};
const std::vector<size_t> ThinPrismFisheyeCameraModel::extra_idx = {4, 5, 6, 7, 8, 9, 10, 11};

///////////////////////////////////////////////////////////////////
// 1D Radial camera model
// Note that this does not project onto 2D point, but rather to a direction
// so project(X) will go to a unit 2D vector pointing from the center towards X
// Note also that project and unproject are not consistent!
// params = cx, cy

void Radial1DCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    // project([X,Y,Z]) = [X,Y] / sqrt(X^2 + Y^2)
    const double nrm = std::max(x.topRows<2>().norm(), 1e-8);
    (*xp)[0] = x(0) / nrm;
    (*xp)[1] = x(1) / nrm;
}

void Radial1DCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                           Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                           Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const double nrm = std::max(x.topRows<2>().norm(), 1e-8);
    const Eigen::Vector2d v = x.topRows<2>() / nrm;
    (*xp)[0] = v(0);
    (*xp)[1] = v(1);

    // jacobian(x / |x|) = I / |x| - x*x' / |x|^3 = (I - v*v') / |x|
    // v = x / |x|
    if (jac) {
        jac->block<2, 2>(0, 0) = (Eigen::Matrix2d::Identity() - (v * v.transpose())) / nrm;
        jac->col(2).setZero();
    }
    if (jac_params) {
        jac_params->resize(2, num_params);
        jac_params->setZero();
    }
}

void Radial1DCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    (*x)[0] = xp(0) - params[0];
    (*x)[1] = xp(1) - params[1];
    (*x)[2] = 1.0;
    x->normalize();
}

const size_t Radial1DCameraModel::num_params = 2;
const std::string Radial1DCameraModel::params_info() { return "cx, cy"; };
const std::vector<size_t> Radial1DCameraModel::focal_idx = {};
const std::vector<size_t> Radial1DCameraModel::principal_point_idx = {0, 1};
const std::vector<size_t> Radial1DCameraModel::extra_idx = {};

///////////////////////////////////////////////////////////////////
// Spherical camera - a 360 camera model mapping latitude and longitude to x and y
// params = w, h

void SphericalCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    // X = [sin(theta) * cos(phi); sin(phi); cos(theta) * cos(phi)]
    const double nrm = x.norm();
    const double theta = std::atan2(x[0], x[2]);
    const double phi = std::asin(std::clamp(x[1] / nrm, -1.0, 1.0));

    (*xp)[0] = (theta + M_PI) / (2 * M_PI) * params[0];
    (*xp)[1] = (phi + M_PI_2) / M_PI * params[1];
}

void SphericalCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                            Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                            Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const double w = params[0], h = params[1];
    const double nrm = x.norm();
    const Eigen::Vector3d v = x / nrm;
    const double v0 = v[0], v1 = v[1], v2 = v[2];

    const double theta = std::atan2(x[0], x[2]);
    const double phi = std::asin(std::clamp(v[1], -1.0, 1.0));
    (*xp)[0] = (theta + M_PI) / (2 * M_PI) * w;
    (*xp)[1] = (phi + M_PI_2) / M_PI * h;

    if (jac) {
        const Eigen::Matrix3d dnorm = (Eigen::Matrix3d::Identity() - (v * v.transpose())) / nrm;
        const double v02_nrm = v0 * v0 + v2 * v2;
        const double sqrt_1_minus_v1_sq = std::sqrt(std::max(1.0 - v1 * v1, 1e-8));
        Eigen::Matrix<double, 2, 3> jac_norm;
        jac_norm << w / (2 * M_PI) * v2 / v02_nrm, 0.0, -w / (2 * M_PI) * v0 / v02_nrm, 0.0,
            h / M_PI / sqrt_1_minus_v1_sq, 0.0;
        (*jac) = jac_norm * dnorm;
    }
    if (jac_params) {
        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = (theta + M_PI) / (2 * M_PI);
        (*jac_params)(1, 0) = (theta + M_PI) / (2 * M_PI);

        (*jac_params)(0, 1) = (phi + M_PI_2) / M_PI;
        (*jac_params)(1, 1) = (phi + M_PI_2) / M_PI;
    }
}

void SphericalCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    const double theta = xp[0] * (2 * M_PI) / params[0] - M_PI;
    const double phi = xp[1] * M_PI / params[1] - M_PI_2;
    const double cos_phi = std::cos(phi);

    (*x)[0] = std::sin(theta) * cos_phi;
    (*x)[1] = std::sin(phi);
    (*x)[2] = std::cos(theta) * cos_phi;
    x->normalize();
}

const size_t SphericalCameraModel::num_params = 2;
const std::string SphericalCameraModel::params_info() { return "w, h"; };
const std::vector<size_t> SphericalCameraModel::focal_idx = {};
const std::vector<size_t> SphericalCameraModel::principal_point_idx = {};
const std::vector<size_t> SphericalCameraModel::extra_idx = {};

///////////////////////////////////////////////////////////////////
// Division - One parameter division model from Fitzgibbon
// params = fx, fy, cx, cy, k

void DivisionCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    // (xp, 1+k*|xp|^2) ~= (x(1:2), x3)
    // (1+k*|xp|^2) * x(1:2) = x3*xp
    // rho = |x(1:2)|, r = |xp|
    // rho + k*r2 * rho = x3 * r
    // rho*k*r2 - x3 * r + rho = 0

    const double rho = x.topRows<2>().norm();
    const double disc2 = x(2) * x(2) - 4.0 * rho * rho * params[4];
    if (disc2 < 0) {
        (*xp).setZero();
        return;
    }
    double sq = std::sqrt(disc2);
    const double r = 2.0 / (x(2) + sq);

    (*xp)[0] = params[0] * r * x(0) + params[2];
    (*xp)[1] = params[1] * r * x(1) + params[3];
}

void DivisionCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                           Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                           Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const double k = params[4];
    const double rho = x.topRows<2>().norm();
    const double disc2 = x(2) * x(2) - 4.0 * rho * rho * k;
    if (disc2 < 0) {
        (*xp).setZero();
        if (jac) {
            jac->setZero();
        }
        if (jac_params) {
            jac_params->resize(2, num_params);
            jac_params->setZero();
        }
        return;
    }
    const double sq = std::sqrt(disc2);
    const double den = x(2) + sq;
    const double r = 2.0 / den;

    const double xp0 = r * x(0);
    const double xp1 = r * x(1);
    (*xp)[0] = params[0] * xp0 + params[2];
    (*xp)[1] = params[1] * xp1 + params[3];

    if (jac) {
        const Eigen::Vector3d ddisc2_dd(-8.0 * k * x(0), -8.0 * k * x(1), 2.0 * x(2));
        const double dsq_ddisc2 = 0.5 / sq;
        const double dr_dden = -2.0 / (den * den);
        const Eigen::Vector3d dr_dd = dr_dden * (dsq_ddisc2 * ddisc2_dd + Eigen::Vector3d(0.0, 0.0, 1.0));

        jac->row(0) = x(0) * dr_dd;
        jac->row(1) = x(1) * dr_dd;
        (*jac)(0, 0) += r;
        (*jac)(1, 1) += r;
        jac->row(0) *= params[0];
        jac->row(1) *= params[1];
    }
    if (jac_params) {
        double dr_dk = 4.0 * rho * rho / (den * den * sq);

        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = xp0;
        (*jac_params)(1, 0) = 0.0;

        (*jac_params)(0, 1) = 0.0;
        (*jac_params)(1, 1) = xp1;

        (*jac_params)(0, 2) = 1.0;
        (*jac_params)(1, 2) = 0.0;

        (*jac_params)(0, 3) = 0.0;
        (*jac_params)(1, 3) = 1.0;

        (*jac_params)(0, 4) = params[0] * x(0) * dr_dk;
        (*jac_params)(1, 4) = params[1] * x(1) * dr_dk;
    }
}

void DivisionCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    const double x0 = (xp(0) - params[2]) / params[0];
    const double y0 = (xp(1) - params[3]) / params[1];
    const double r2 = x0 * x0 + y0 * y0;

    (*x)[0] = x0;
    (*x)[1] = y0;
    (*x)[2] = 1.0 + params[4] * r2;
    x->normalize();
}

void DivisionCameraModel::unproject_with_jac(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                             Eigen::Vector3d *x, Eigen::Matrix<double, 3, 2> *jac,
                                             Eigen::Matrix<double, 3, Eigen::Dynamic> *jac_p) {
    const double x0 = (xp(0) - params[2]) / params[0];
    const double y0 = (xp(1) - params[3]) / params[1];
    const double r2 = x0 * x0 + y0 * y0;

    (*x)[0] = x0;
    (*x)[1] = y0;
    (*x)[2] = 1.0 + params[4] * r2;
    double inv_norm = 1.0 / x->norm();

    (*x) *= inv_norm;

    if (jac) {
        (*jac)(0, 0) = params[0] * (1 - (*x)(0) * (*x)(0)) + 2 * (*x)(0) * (*x)(2) * params[4] * (params[2] - xp[0]);
        (*jac)(0, 1) = (*x)(0) * (-params[1] * (*x)(1) + 2 * (*x)(2) * params[4] * (params[3] - xp[1]));
        (*jac)(1, 0) = (*x)(1) * (-params[0] * (*x)(0) + 2 * (*x)(2) * params[4] * (params[2] - xp[0]));
        (*jac)(1, 1) = params[1] * (1 - (*x)(1) * (*x)(1)) + 2 * (*x)(1) * (*x)(2) * params[4] * (params[3] - xp[1]);
        (*jac)(2, 0) = -params[0] * (*x)(0) * (*x)(2) + 2 * params[4] * (params[2] - xp[0]) * ((*x)(2) * (*x)(2) - 1);
        (*jac)(2, 1) = -params[1] * (*x)(1) * (*x)(2) + 2 * params[4] * (params[3] - xp[1]) * ((*x)(2) * (*x)(2) - 1);
        jac->col(0) *= inv_norm / (params[0] * params[0]);
        jac->col(1) *= inv_norm / (params[1] * params[1]);
    }

    if (jac_p) {
        jac_p->resize(3, 5);
        (*jac_p)(0, 0) = (params[2] - xp[0]) * (-params[0] * ((*x)(0) * (*x)(0) - 1) +
                                                2 * (*x)(0) * (*x)(2) * params[4] * (params[2] - xp[0]));
        (*jac_p)(0, 1) =
            (*x)(0) * (params[3] - xp[1]) * (-params[1] * (*x)(1) + 2 * (*x)(2) * params[4] * (params[3] - xp[1]));
        (*jac_p)(0, 2) = params[0] * ((*x)(0) * (*x)(0) - 1) - 2 * (*x)(0) * (*x)(2) * params[4] * (params[2] - xp[0]);
        (*jac_p)(0, 3) = (*x)(0) * (params[1] * (*x)(1) - 2 * (*x)(2) * params[4] * (params[3] - xp[1]));
        (*jac_p)(0, 4) = -(*x)(0) * (*x)(2) * r2;
        (*jac_p)(1, 0) =
            (*x)(1) * (params[2] - xp[0]) * (-params[0] * (*x)(0) + 2 * (*x)(2) * params[4] * (params[2] - xp[0]));
        (*jac_p)(1, 1) = (params[3] - xp[1]) * (-params[1] * ((*x)(1) * (*x)(1) - 1) +
                                                2 * (*x)(1) * (*x)(2) * params[4] * (params[3] - xp[1]));
        (*jac_p)(1, 2) = (*x)(1) * (params[0] * (*x)(0) - 2 * (*x)(2) * params[4] * (params[2] - xp[0]));
        (*jac_p)(1, 3) = params[1] * ((*x)(1) * (*x)(1) - 1) - 2 * (*x)(1) * (*x)(2) * params[4] * (params[3] - xp[1]);
        (*jac_p)(1, 4) = -(*x)(1) * (*x)(2) * r2;
        (*jac_p)(2, 0) = (params[2] - xp[0]) * (-params[0] * (*x)(0) * (*x)(2) +
                                                2 * params[4] * (params[2] - xp[0]) * ((*x)(1) * (*x)(1) - 1));
        (*jac_p)(2, 1) = (params[3] - xp[1]) * (-params[1] * (*x)(1) * (*x)(2) +
                                                2 * params[4] * (params[3] - xp[1]) * ((*x)(1) * (*x)(1) - 1));
        (*jac_p)(2, 2) = params[0] * (*x)(0) * (*x)(2) - 2 * params[4] * (params[2] - xp[0]) * ((*x)(1) * (*x)(1) - 1);
        (*jac_p)(2, 3) = params[1] * (*x)(1) * (*x)(2) - 2 * params[4] * (params[3] - xp[1]) * ((*x)(1) * (*x)(1) - 1);
        (*jac_p)(2, 4) = r2 * (1 - (*x)(2) * (*x)(2));
        jac_p->col(0) /= std::pow(params[0], 3);
        jac_p->col(1) /= std::pow(params[1], 3);
        jac_p->col(2) /= std::pow(params[0], 2);
        jac_p->col(3) /= std::pow(params[1], 2);
        *jac_p *= inv_norm;
    }
}

const size_t DivisionCameraModel::num_params = 5;
const std::string DivisionCameraModel::params_info() { return "fx, fy, cx, cy, k"; };
const std::vector<size_t> DivisionCameraModel::focal_idx = {0, 1};
const std::vector<size_t> DivisionCameraModel::principal_point_idx = {2, 3};
const std::vector<size_t> DivisionCameraModel::extra_idx = {4};

///////////////////////////////////////////////////////////////////
// Simple Division - One parameter division model from Fitzgibbon
// params = f, cx, cy, k

void SimpleDivisionCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x,
                                        Eigen::Vector2d *xp) {
    // (xp, 1+k*|xp|^2) ~= (x(1:2), x3)
    // (1+k*|xp|^2) * x(1:2) = x3*xp
    // rho = |x(1:2)|, r = |xp|
    // rho + k*r2 * rho = x3 * r
    // rho*k*r2 - x3 * r + rho = 0

    const double rho = x.topRows<2>().norm();
    const double disc2 = x(2) * x(2) - 4.0 * rho * rho * params[3];
    if (disc2 < 0) {
        (*xp).setZero();
        return;
    }
    double sq = std::sqrt(disc2);
    const double r = 2.0 / (x(2) + sq);

    (*xp)[0] = params[0] * r * x(0) + params[1];
    (*xp)[1] = params[0] * r * x(1) + params[2];
}

void SimpleDivisionCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                                 Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                                                 Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    const double k = params[3];
    const double rho = x.topRows<2>().norm();
    const double disc2 = x(2) * x(2) - 4.0 * rho * rho * k;
    if (disc2 < 0) {
        (*xp).setZero();
        if (jac) {
            jac->setZero();
        }
        if (jac_params) {
            jac_params->resize(2, num_params);
            jac_params->setZero();
        }
        return;
    }
    const double sq = std::sqrt(disc2);
    const double den = x(2) + sq;
    const double r = 2.0 / den;

    const double xp0 = r * x(0);
    const double xp1 = r * x(1);
    (*xp)[0] = params[0] * xp0 + params[1];
    (*xp)[1] = params[0] * xp1 + params[2];

    if (jac) {
        const Eigen::Vector3d ddisc2_dd(-8.0 * k * x(0), -8.0 * k * x(1), 2.0 * x(2));
        const double dsq_ddisc2 = 0.5 / sq;
        const double dr_dden = -2.0 / (den * den);
        const Eigen::Vector3d dr_dd = dr_dden * (dsq_ddisc2 * ddisc2_dd + Eigen::Vector3d(0.0, 0.0, 1.0));

        jac->row(0) = x(0) * dr_dd;
        jac->row(1) = x(1) * dr_dd;
        (*jac)(0, 0) += r;
        (*jac)(1, 1) += r;
        jac->row(0) *= params[0];
        jac->row(1) *= params[0];
    }
    if (jac_params) {
        double dr_dk = 4.0 * rho * rho / (den * den * sq);

        jac_params->resize(2, num_params);
        (*jac_params)(0, 0) = xp0;
        (*jac_params)(1, 0) = xp1;

        (*jac_params)(0, 1) = 1.0;
        (*jac_params)(1, 1) = 0.0;

        (*jac_params)(0, 2) = 0.0;
        (*jac_params)(1, 2) = 1.0;

        (*jac_params)(0, 3) = params[0] * x(0) * dr_dk;
        (*jac_params)(1, 3) = params[0] * x(1) * dr_dk;
    }
}

void SimpleDivisionCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                          Eigen::Vector3d *x) {
    const double x0 = (xp(0) - params[1]) / params[0];
    const double y0 = (xp(1) - params[2]) / params[0];
    const double r2 = x0 * x0 + y0 * y0;

    (*x)[0] = x0;
    (*x)[1] = y0;
    (*x)[2] = 1.0 + params[3] * r2;
    x->normalize();
}

void SimpleDivisionCameraModel::unproject_with_jac(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                                   Eigen::Vector3d *x, Eigen::Matrix<double, 3, 2> *jac,
                                                   Eigen::Matrix<double, 3, Eigen::Dynamic> *jac_p) {
    const double x0 = (xp(0) - params[1]) / params[0];
    const double y0 = (xp(1) - params[2]) / params[0];
    const double r2 = x0 * x0 + y0 * y0;

    (*x)[0] = x0;
    (*x)[1] = y0;
    (*x)[2] = 1.0 + params[3] * r2;
    double inv_norm = 1.0 / x->norm();

    (*x) *= inv_norm;

    if (jac) {
        (*jac)(0, 0) = params[0] * (1 - (*x)(0) * (*x)(0)) + 2 * (*x)(0) * (*x)(2) * params[3] * (params[1] - xp[0]);
        (*jac)(0, 1) = (*x)(0) * (-params[0] * (*x)(1) + 2 * (*x)(2) * params[3] * (params[2] - xp[1]));
        (*jac)(1, 0) = (*x)(1) * (-params[0] * (*x)(0) + 2 * (*x)(2) * params[3] * (params[1] - xp[0]));
        (*jac)(1, 1) = params[0] * (1 - (*x)(1) * (*x)(1)) + 2 * (*x)(1) * (*x)(2) * params[3] * (params[2] - xp[1]);
        (*jac)(2, 0) = -params[0] * (*x)(0) * (*x)(2) + 2 * params[3] * (params[1] - xp[0]) * ((*x)(2) * (*x)(2) - 1);
        (*jac)(2, 1) = -params[0] * (*x)(1) * (*x)(2) + 2 * params[3] * (params[2] - xp[1]) * ((*x)(2) * (*x)(2) - 1);
        jac->col(0) *= inv_norm / (params[0] * params[0]);
        jac->col(1) *= inv_norm / (params[0] * params[0]);
    }

    if (jac_p) {
        jac_p->resize(3, 4);
        (*jac_p)(0, 0) = (params[1] - xp[0]) * (-params[0] * ((*x)(0) * (*x)(0) - 1) +
                                                2 * (*x)(0) * (*x)(2) * params[3] * (params[1] - xp[0]));
        (*jac_p)(0, 0) +=
            (*x)(0) * (params[2] - xp[1]) * (-params[0] * (*x)(1) + 2 * (*x)(2) * params[3] * (params[2] - xp[1]));
        (*jac_p)(0, 1) = params[0] * ((*x)(0) * (*x)(0) - 1) - 2 * (*x)(0) * (*x)(2) * params[3] * (params[1] - xp[0]);
        (*jac_p)(0, 2) = (*x)(0) * (params[0] * (*x)(1) - 2 * (*x)(2) * params[3] * (params[2] - xp[1]));
        (*jac_p)(0, 3) = -(*x)(0) * (*x)(2) * r2;
        (*jac_p)(1, 0) =
            (*x)(1) * (params[1] - xp[0]) * (-params[0] * (*x)(0) + 2 * (*x)(2) * params[3] * (params[1] - xp[0]));
        (*jac_p)(1, 0) += (params[2] - xp[1]) * (-params[0] * ((*x)(1) * (*x)(1) - 1) +
                                                 2 * (*x)(1) * (*x)(2) * params[3] * (params[2] - xp[1]));
        (*jac_p)(1, 1) = (*x)(1) * (params[0] * (*x)(0) - 2 * (*x)(2) * params[3] * (params[1] - xp[0]));
        (*jac_p)(1, 2) = params[0] * ((*x)(1) * (*x)(1) - 1) - 2 * (*x)(1) * (*x)(2) * params[3] * (params[2] - xp[1]);
        (*jac_p)(1, 3) = -(*x)(1) * (*x)(2) * r2;
        (*jac_p)(2, 0) = (params[1] - xp[0]) * (-params[0] * (*x)(0) * (*x)(2) +
                                                2 * params[3] * (params[1] - xp[0]) * ((*x)(1) * (*x)(1) - 1));
        (*jac_p)(2, 0) += (params[2] - xp[1]) * (-params[0] * (*x)(1) * (*x)(2) +
                                                 2 * params[3] * (params[2] - xp[1]) * ((*x)(1) * (*x)(1) - 1));
        (*jac_p)(2, 1) = params[0] * (*x)(0) * (*x)(2) - 2 * params[3] * (params[1] - xp[0]) * ((*x)(1) * (*x)(1) - 1);
        (*jac_p)(2, 2) = params[0] * (*x)(1) * (*x)(2) - 2 * params[3] * (params[2] - xp[1]) * ((*x)(1) * (*x)(1) - 1);
        (*jac_p)(2, 3) = r2 * (1 - (*x)(2) * (*x)(2));
        jac_p->col(0) /= std::pow(params[0], 3);
        jac_p->col(1) /= std::pow(params[0], 2);
        jac_p->col(2) /= std::pow(params[0], 2);
        *jac_p *= inv_norm;
    }
}

const size_t SimpleDivisionCameraModel::num_params = 4;
const std::string SimpleDivisionCameraModel::params_info() { return "f, cx, cy, k"; };
const std::vector<size_t> SimpleDivisionCameraModel::focal_idx = {0};
const std::vector<size_t> SimpleDivisionCameraModel::principal_point_idx = {1, 2};
const std::vector<size_t> SimpleDivisionCameraModel::extra_idx = {3};

///////////////////////////////////////////////////////////////////
// Null camera - this is used as a dummy value in various places
// This is equivalent to a pinhole camera with identity K matrix
// params = {}

void NullCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    *xp = x.hnormalized();
}
void NullCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp,
                                       Eigen::Matrix<double, 2, 3> *jac,
                                       Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params) {
    *xp = x.hnormalized();
    const double z_inv = 1.0 / x(2);
    if (jac) {
        *jac << z_inv, 0.0, -(*xp)(0) * z_inv, 0.0, z_inv, -(*xp)(1) * z_inv;
    }
    if (jac_params) {
        jac_params->resize(2, num_params);
        jac_params->setZero();
    }
}
void NullCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    *x = xp.homogeneous();
}

const size_t NullCameraModel::num_params = 0;
const std::string NullCameraModel::params_info() { return ""; };
const std::vector<size_t> NullCameraModel::focal_idx = {};
const std::vector<size_t> NullCameraModel::principal_point_idx = {};
const std::vector<size_t> NullCameraModel::extra_idx = {};

} // namespace poselib