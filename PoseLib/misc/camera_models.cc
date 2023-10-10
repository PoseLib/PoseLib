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

static const double UNDIST_TOL = 1e-10;
static const size_t UNDIST_MAX_ITER = 100;

///////////////////////////////////////////////////////////////////
// Camera - base class storing ID

Camera::Camera() { Camera(-1, {}, -1, -1); }
Camera::Camera(const std::string &model_name, const std::vector<double> &p, int w, int h) {
    model_id = id_from_string(model_name);
    params = p;
    width = w;
    height = h;
}

Camera::Camera(int id, const std::vector<double> &p, int w, int h) {
    model_id = id;
    params = p;
    width = w;
    height = h;
}
Camera::Camera(const std::string &init_txt) {
    initialize_from_txt(init_txt);
}

int Camera::id_from_string(const std::string &model_name) {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    if (model_name == Model::to_string()) {                                                                            \
        return Model::model_id;                                                                                        \
    }

    SWITCH_CAMERA_MODELS

#undef SWITCH_CAMERA_MODEL_CASE

    return -1;
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
void Camera::project_with_jac(const Eigen::Vector3d &x, Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) const {
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        Model::project_with_jac(params, x, xp, jac);                                                                   \
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

void Camera::project(const std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector2d> *xp) const {
    xp->resize(x.size());
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        for (int i = 0; i < x.size(); ++i) {                                                                           \
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
                              std::vector<Eigen::Matrix<double, 2, 3>> *jac) const {
    xp->resize(x.size());
    jac->resize(x.size());
#define SWITCH_CAMERA_MODEL_CASE(Model)                                                                                \
    case Model::model_id:                                                                                              \
        for (int i = 0; i < x.size(); ++i) {                                                                           \
            Model::project_with_jac(params, x[i], &((*xp)[i]), &((*jac)[i]));                                          \
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
        for (int i = 0; i < xp.size(); ++i) {                                                                          \
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

int Camera::initialize_from_txt(const std::string &line) {
    std::stringstream ss(line);
    int camera_id;
    ss >> camera_id;

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

///////////////////////////////////////////////////////////////////
// Pinhole camera
// params = fx, fy, cx, cy

void PinholeCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    (*xp)(0) = params[0] * x(0) / x(2) + params[2];
    (*xp)(1) = params[1] * x(1) / x(2) + params[3];
}
void PinholeCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                          Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) {
    const double inv_z = 1.0 / x(2);
    const double px = params[0] * x(0) * inv_z;
    const double py = params[1] * x(1) * inv_z;
    (*xp)(0) = px + params[2];
    (*xp)(1) = py + params[3];
    (*jac)(0, 0) = params[0] * inv_z;
    (*jac)(0, 1) = 0.0;
    (*jac)(0, 2) = -px * inv_z;

    (*jac)(1, 0) = 0.0;
    (*jac)(1, 1) = params[1] * inv_z;
    (*jac)(1, 2) = -py * inv_z;
}
void PinholeCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    (*x)(0) = (xp(0) - params[2]) / params[0];
    (*x)(1) = (xp(1) - params[3]) / params[1];
    (*x)(2) = 1.0;
    x->normalize();
}
const std::vector<size_t> PinholeCameraModel::focal_idx = {0, 1};
const std::vector<size_t> PinholeCameraModel::principal_point_idx = {2, 3};

///////////////////////////////////////////////////////////////////
// Simple Pinhole camera
// params = f, cx, cy

void SimplePinholeCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x,
                                       Eigen::Vector2d *xp) {
    (*xp)(0) = params[0] * x(0) / x(2) + params[1];
    (*xp)(1) = params[0] * x(1) / x(2) + params[2];
}
void SimplePinholeCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                                Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) {
    const double inv_z = 1.0 / x(2);
    const double px = params[0] * x(0) * inv_z;
    const double py = params[0] * x(1) * inv_z;
    (*xp)(0) = px + params[1];
    (*xp)(1) = py + params[2];
    (*jac)(0, 0) = params[0] * inv_z;
    (*jac)(0, 1) = 0.0;
    (*jac)(0, 2) = -px * inv_z;

    (*jac)(1, 0) = 0.0;
    (*jac)(1, 1) = params[0] * inv_z;
    (*jac)(1, 2) = -py * inv_z;
}
void SimplePinholeCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp,
                                         Eigen::Vector3d *x) {
    (*x)(0) = (xp(0) - params[1]) / params[0];
    (*x)(1) = (xp(1) - params[2]) / params[0];
    (*x)(2) = 1.0;
    x->normalize();
}
const std::vector<size_t> SimplePinholeCameraModel::focal_idx = {0};
const std::vector<size_t> SimplePinholeCameraModel::principal_point_idx = {1, 2};

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
                                         Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) {
    const double inv_z = 1.0 / x(2);
    const double px = x(0) * inv_z;
    const double py = x(1) * inv_z;
    const double r2 = px * px + py * py;
    const double alpha = (1.0 + params[3] * r2 + params[4] * r2 * r2);
    const double alphap = (2.0 * params[3] + 4.0 * params[4] * r2);
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
const std::vector<size_t> RadialCameraModel::focal_idx = {0};
const std::vector<size_t> RadialCameraModel::principal_point_idx = {1, 2};

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
                                               Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) {
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

    (*jac)(0, 0) = inv_z;
    (*jac)(0, 1) = 0;
    (*jac)(0, 2) = -px * inv_z;
    (*jac)(1, 0) = 0;
    (*jac)(1, 1) = inv_z;
    (*jac)(1, 2) = -py * inv_z;
    *jac = jac_d * (*jac);

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
const std::vector<size_t> SimpleRadialCameraModel::focal_idx = {0};
const std::vector<size_t> SimpleRadialCameraModel::principal_point_idx = {1, 2};


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
                                   Eigen::Vector2d &xp, Eigen::Matrix2d &jac) {
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
}

void OpenCVCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    Eigen::Vector2d x0(x(0)/x(2), x(1)/x(2));
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
                                         Eigen::Vector2d *xp, Eigen::Matrix<double,2,3> *jac) {
    Eigen::Vector2d x0(x(0)/x(2), x(1)/x(2));
    Eigen::Matrix<double, 2, 2> jac0;
    jac0.setZero();
    compute_opencv_distortion_jac(params[4], params[5], params[6], params[7], x0, *xp, jac0);
    *jac << 1.0/x(2), 0.0, -x0(0)/x(2),
            0.0, 1.0/x(2), -x0(1)/x(2);
    *jac = jac0 * (*jac);
    jac->row(0) *= params[0];
    jac->row(1) *= params[1];
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
const std::vector<size_t> OpenCVCameraModel::focal_idx = {0, 1};
const std::vector<size_t> OpenCVCameraModel::principal_point_idx = {2, 3};


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
                                                Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) {

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
        (*jac)(0, 0) = params[0] * (inv_r * rd + x(0) * dinv_r_dx * rd + x(0) * inv_r * drd_dx);
        (*jac)(0, 1) = params[0] * x(0) * (dinv_r_dy * rd + inv_r * drd_dy);
        (*jac)(0, 2) = params[0] * x(0) * inv_r * drd_dz;

        (*xp)(1) = params[1] * x(1) * inv_r * rd + params[3];
        (*jac)(1, 0) = params[1] * x(1) * (dinv_r_dx * rd + inv_r * drd_dx);
        (*jac)(1, 1) = params[1] * (inv_r * rd + x(1) * dinv_r_dy * rd + x(1) * inv_r * drd_dy);
        (*jac)(1, 2) = params[1] * x(1) * inv_r * drd_dz;
    } else {
        // Very close to the principal axis - ignore distortion
        (*xp)(0) = params[0] * x(0) + params[2];
        (*xp)(1) = params[1] * x(1) + params[3];
        (*jac)(0, 0) = params[0];
        (*jac)(0, 1) = 0.0;
        (*jac)(0, 2) = 0.0;
        (*jac)(1, 0) = 0.0;
        (*jac)(1, 1) = params[1];
        (*jac)(1, 2) = 0.0;
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
    double theta = 0;

    if (rd > 1e-8) {
        // try zero-init first
        double res = opencv_fisheye_newton(params, rd, theta);
        if (res > UNDIST_TOL || theta < 0) {
            // If this fails try to initialize with rho (first order approx.)
            theta = rd;
            res = opencv_fisheye_newton(params, rd, theta);

            if (res > UNDIST_TOL || theta < 0) {
                // try once more
                theta = 0.5 * rd;
                res = opencv_fisheye_newton(params, rd, theta);

                if (res > UNDIST_TOL || theta < 0) {
                    // try once more
                    theta = 1.5 * rd;
                    res = opencv_fisheye_newton(params, rd, theta);
                    // if this does not work, just fail silently... yay
                }
            }
        }

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
const std::vector<size_t> OpenCVFisheyeCameraModel::focal_idx = {0, 1};
const std::vector<size_t> OpenCVFisheyeCameraModel::principal_point_idx = {2, 3};



///////////////////////////////////////////////////////////////////
// Full OpenCV camera
//   params = fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

void compute_full_opencv_distortion(double k1, double k2, double p1, double p2, double k3,
                                    double k4, double k5, double k6, const Eigen::Vector2d &x,
                                    Eigen::Vector2d &xp) {
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

void compute_full_opencv_distortion_jac(double k1, double k2, double p1, double p2, double k3,
                                   double k4, double k5, double k6, const Eigen::Vector2d &x,
                                   Eigen::Vector2d &xp, Eigen::Matrix2d &jac) {
    const double u = x(0);
    const double v = x(1);
    const double u2 = u * u;
    const double uv = u * v;
    const double v2 = v * v;
    const double r2 = u * u + v * v;
    const double r4 = r2 * r2;
    const double r6 = r2 * r4;
    
    /* TODO update
    jac(0, 0) = k2 * r2 * r2 + 6 * p2 * u + 2 * p1 * v + u * (2 * k1 * u + 4 * k2 * u * r2) + k1 * r2 + 1.0;
    jac(0, 1) = 2 * p1 * u + 2 * p2 * v + v * (2 * k1 * u + 4 * k2 * u * r2);
    jac(1, 0) = 2 * p1 * u + 2 * p2 * v + u * (2 * k1 * v + 4 * k2 * v * r2);
    jac(1, 1) = k2 * r2 * r2 + 2 * p2 * u + 6 * p1 * v + v * (2 * k1 * v + 4 * k2 * v * r2) + k1 * r2 + 1.0;
    */

    const double alpha = (1.0 + k1 * r2 + k2 * r4 + k3 * r6) / (1.0 + k4 * r2 + k5 * r4 + k6 * r6);
    xp(0) = alpha * u + 2.0 * p1 * uv + p2 * (r2 + 2.0 * u2);
    xp(1) = alpha * v + 2.0 * p2 * uv + p1 * (r2 + 2.0 * v2);
}

void FullOpenCVCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    Eigen::Vector2d x0(x(0)/x(2), x(1)/x(2));
    compute_full_opencv_distortion(params[4], params[5], params[6], params[7], 
                              params[8], params[9], params[10], params[11], x0, *xp);
    (*xp)(0) = params[0] * (*xp)(0) + params[2];
    (*xp)(1) = params[1] * (*xp)(1) + params[3];
}

Eigen::Vector2d undistort_full_opencv(double k1, double k2, double p1, double p2, double k3,
                                   double k4, double k5, double k6, const Eigen::Vector2d &xp) {
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
                                         Eigen::Vector2d *xp, Eigen::Matrix<double,2,3> *jac) {
    Eigen::Vector2d x0(x(0)/x(2), x(1)/x(2));
    Eigen::Matrix<double, 2, 2> jac0;
    jac0.setZero();
    compute_full_opencv_distortion_jac(params[4], params[5], params[6], params[7], 
                                       params[8], params[9], params[10], params[11], x0, *xp, jac0);
    *jac << 1.0/x(2), 0.0, -x0(0)/x(2),
            0.0, 1.0/x(2), -x0(1)/x(2);
    *jac = jac0 * (*jac);
    jac->row(0) *= params[0];
    jac->row(1) *= params[1];
    (*xp)(0) = params[0] * (*xp)(0) + params[2];
    (*xp)(1) = params[1] * (*xp)(1) + params[3];
}
void FullOpenCVCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    Eigen::Vector2d xp0;
    xp0 << (xp(0) - params[2]) / params[0], (xp(1) - params[3]) / params[1];
    Eigen::Vector2d x0;
    x0 = undistort_full_opencv(params[4], params[5], params[6], params[7],
                               params[8], params[9], params[10], params[11], xp0);
    *x << x0(0), x0(1), 1.0;
    x->normalize();
}
const std::vector<size_t> FullOpenCVCameraModel::focal_idx = {0, 1};
const std::vector<size_t> FullOpenCVCameraModel::principal_point_idx = {2, 3};


///////////////////////////////////////////////////////////////////
// 1D Radial camera model
// Note that this does not project onto 2D point, but rather to a direction
// so project(X) will go to a unit 2D vector pointing from the center towards X
// Note also that project and unproject are not consistent!
// params = w, h


void Radial1DCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    // project([X,Y,Z]) = [X,Y] / sqrt(X^2 + Y^2)
    const double nrm = std::max(x.topRows<2>().norm(), 1e-8);
    (*xp)[0] = x(0) / nrm;
    (*xp)[1] = x(1) / nrm;
}

void Radial1DCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x,
                                            Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) {
    const double nrm = std::max(x.topRows<2>().norm(), 1e-8);
    const Eigen::Vector2d v = x.topRows<2>() / nrm;
    (*xp)[0] = v(0);
    (*xp)[1] = v(1);

    // jacobian(x / |x|) = I / |x| - x*x' / |x|^3 = (I - v*v') / |x|
    // v = x / |x|
    jac->block<2,2>(0,0) = (Eigen::Matrix2d::Identity() - (v * v.transpose())) / nrm;
    jac->col(2).setZero();
}

void Radial1DCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    (*x)[0] = xp(0) - params[0];
    (*x)[1] = xp(1) - params[1];
    (*x)[2] = 1.0;
}

const std::vector<size_t> Radial1DCameraModel::focal_idx = {};
const std::vector<size_t> Radial1DCameraModel::principal_point_idx = {0, 1};

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
                                            Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) {
    const double w = params[0], h = params[1];
    const double nrm = x.norm();
    const Eigen::Vector3d v = x / nrm;
    const double v0 = v[0], v1 = v[1], v2 = v[2];

    const double theta = std::atan2(x[0], x[2]);
    const double phi = std::asin(std::clamp(v[1], -1.0, 1.0));
    (*xp)[0] = (theta + M_PI) / (2 * M_PI) * w;
    (*xp)[1] = (phi + M_PI_2) / M_PI * h;

    const Eigen::Matrix3d dnorm = (Eigen::Matrix3d::Identity() - (v * v.transpose())) / nrm;
    const double v02_nrm = v0 * v0 + v2 * v2;
    const double sqrt_1_minus_v1_sq = std::sqrt(std::max(1.0 - v1 * v1, 1e-8));
    Eigen::Matrix<double, 2, 3> jac_norm;
    jac_norm << w / (2 * M_PI) * v2 / v02_nrm, 0.0, -w / (2 * M_PI) * v0 / v02_nrm, 0.0, h / M_PI / sqrt_1_minus_v1_sq,
        0.0;
    (*jac) = jac_norm * dnorm;
}

void SphericalCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    const double theta = xp[0] * (2 * M_PI) / params[0] - M_PI;
    const double phi = xp[1] * M_PI / params[1] - M_PI_2;
    const double cos_phi = std::cos(phi);

    (*x)[0] = std::sin(theta) * cos_phi;
    (*x)[1] = std::sin(phi);
    (*x)[2] = std::cos(theta) * cos_phi;
}

const std::vector<size_t> SphericalCameraModel::focal_idx = {};
const std::vector<size_t> SphericalCameraModel::principal_point_idx = {};

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
                                           Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac) {
    const double k = params[4];
    const double rho = x.topRows<2>().norm();
    const double disc2 = x(2) * x(2) - 4.0 * rho * rho * k;
    if (disc2 < 0) {
        (*xp).setZero();
        return;
    }
    const double sq = std::sqrt(disc2);
    const double den = x(2) + sq;
    const double r = 2.0 / den;

    const double xp0 = r * x(0);
    const double xp1 = r * x(1);
    (*xp)[0] = params[0] * xp0 + params[2];
    (*xp)[1] = params[1] * xp1 + params[3];

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

void DivisionCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    const double x0 = (xp(0) - params[2]) / params[0];
    const double y0 = (xp(1) - params[3]) / params[1];
    const double r2 = x0 * x0 + y0 * y0;

    (*x)[0] = x0;
    (*x)[1] = y0;
    (*x)[2] = 1.0 + params[4] * r2;
    x->normalize();
}

const std::vector<size_t> DivisionCameraModel::focal_idx = {0, 1};
const std::vector<size_t> DivisionCameraModel::principal_point_idx = {2, 3};

///////////////////////////////////////////////////////////////////
// Null camera - this is used as a dummy value in various places
// This is equivalent to a pinhole camera with identity K matrix
// params = {}

void NullCameraModel::project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp) {
    *xp = x.hnormalized();
}
void NullCameraModel::project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp,
                                       Eigen::Matrix<double, 2, 3> *jac) {
    *xp = x.hnormalized();
    const double z_inv = 1.0 / x(2);
    *jac << z_inv, 0.0, -(*xp)(0) * z_inv,
            0.0, z_inv, -(*xp)(1) * z_inv;
}
void NullCameraModel::unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x) {
    *x = xp.homogeneous();
}
const std::vector<size_t> NullCameraModel::focal_idx = {};
const std::vector<size_t> NullCameraModel::principal_point_idx = {};

} // namespace poselib