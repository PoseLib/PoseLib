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

#ifndef POSELIB_CAMERA_MODELS_H_
#define POSELIB_CAMERA_MODELS_H_

#include <Eigen/Dense>
#include <PoseLib/types.h>
#include <vector>

namespace poselib {

// c.f. https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
enum CameraModelId {
    INVALID = -1,
    SIMPLE_PINHOLE = 0,
    PINHOLE = 1,
    SIMPLE_RADIAL = 2,
    RADIAL = 3,
    OPENCV = 4,
    OPENCV_FISHEYE = 5,
    FULL_OPENCV = 6,
    FOV = 7,
    SIMPLE_RADIAL_FISHEYE = 8,
    RADIAL_FISHEYE = 9,
    THIN_PRISM_FISHEYE = 10,
    RADIAL_1D = 11,
    SPHERICAL = 101,
    DIVISION = 102
};

struct Camera {
    int model_id;
    int width;
    int height;
    std::vector<double> params;

    Camera();
    Camera(int model_id);
    Camera(const std::string &model_name, const std::vector<double> &params, int width = 0, int height = 0);
    Camera(int model_id, const std::vector<double> &params, int width = 0, int height = 0);
    Camera(const std::string &init_txt);

    // Ensures the parameters are valid
    // If width/height are non-zero
    //  focal = max(width, height) * 1.2
    //  principal point = (width,height)/2
    //  extra params = 0.0
    // Else
    //  focal = 1.0
    //  principal point = extra params = 0.0
    void init_params();

    // Projection and distortion (2d to 3d)
    void project(const Eigen::Vector3d &x, Eigen::Vector2d *xp) const;
    void project_with_jac(const Eigen::Vector3d &x, Eigen::Vector2d *xp, Eigen::Matrix<double, 2, 3> *jac,
                          Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_params = nullptr) const;
    void unproject(const Eigen::Vector2d &xp, Eigen::Vector3d *x) const;

    // Pinhole wrappers (old colmap-style 2d to 2d)
    void project(const Eigen::Vector2d &x, Eigen::Vector2d *xp) const {
        Eigen::Vector3d x3d(x(0), x(1), 1.0);
        x3d.normalize();
        project(x3d, xp);
    }
    void project_with_jac(const Eigen::Vector2d &x, Eigen::Vector2d *xp, Eigen::Matrix2d *jac) const {
        throw std::runtime_error("old project_with_jac called. No longer supported.");
    }
    void unproject(const Eigen::Vector2d &xp, Eigen::Vector2d *x) const {
        Eigen::Vector3d x3d;
        unproject(xp, &x3d);
        *x = x3d.hnormalized();
    }

    // vector wrappers for the project/unprojection
    void project(const std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector2d> *xp) const;
    void project_with_jac(const std::vector<Eigen::Vector3d> &x, std::vector<Eigen::Vector2d> *xp,
                          std::vector<Eigen::Matrix<double, 2, 3>> *jac_point,
                          std::vector<Eigen::Matrix<double, 2, Eigen::Dynamic>> *jac_param) const;
    void unproject(const std::vector<Eigen::Vector2d> &xp, std::vector<Eigen::Vector3d> *x) const;

    // Update the camera parameters such that the projections are rescaled
    void rescale(double scale);
    // Return camera model as string
    std::string model_name() const;
    // Returns focal length (average in case of non-unit aspect ratio)
    double focal() const;
    double focal_x() const;
    double focal_y() const;
    void set_focal(double f);
    Eigen::Vector2d principal_point() const;
    void set_principal_point(double cx, double cy);

    Eigen::Matrix3d calib_matrix() const;

    double max_dim() const {
        int m_dim = std::max(width, height);
        if (m_dim <= 0) {
            return 1.0;
        } else {
            return static_cast<double>(m_dim);
        }
    }

    std::vector<size_t> focal_idx() const;
    std::vector<size_t> principal_point_idx() const;

    // Parses a camera from a line from cameras.txt, returns the camera_id
    int initialize_from_txt(const std::string &line);
    // Creates line for cameras.txt (inverse of initialize_from_txt)
    // If camera_id == -1 it is ommited
    std::string to_cameras_txt(int camera_id = -1) const;

    // Returns string explaining the params vector
    std::string params_info() const;

    // helpers for camera model ids
    static int id_from_string(const std::string &model_name);
    static std::string name_from_id(int id);

    // helper for refinement
    std::vector<size_t> get_param_refinement_idx(const BundleOptions &opt);
};

#define SETUP_CAMERA_SHARED_DEFS(ClassName, ModelName, ModelId)                                                        \
    class ClassName {                                                                                                  \
      public:                                                                                                          \
        static void project(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp);         \
        static void project_with_jac(const std::vector<double> &params, const Eigen::Vector3d &x, Eigen::Vector2d *xp, \
                                     Eigen::Matrix<double, 2, 3> *jac_point,                                           \
                                     Eigen::Matrix<double, 2, Eigen::Dynamic> *jac_param = nullptr);                   \
        static void unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector3d *x);       \
        static const std::vector<size_t> focal_idx;                                                                    \
        static const std::vector<size_t> principal_point_idx;                                                          \
        static const std::vector<size_t> extra_idx;                                                                    \
        static const size_t num_params;                                                                                \
        static const std::string params_info();                                                                        \
        static const int model_id = ModelId;                                                                           \
        static const std::string to_string() { return ModelName; }                                                     \
    };

SETUP_CAMERA_SHARED_DEFS(NullCameraModel, "NULL", -1);
SETUP_CAMERA_SHARED_DEFS(SimplePinholeCameraModel, "SIMPLE_PINHOLE", 0);
SETUP_CAMERA_SHARED_DEFS(PinholeCameraModel, "PINHOLE", 1);
SETUP_CAMERA_SHARED_DEFS(SimpleRadialCameraModel, "SIMPLE_RADIAL", 2);
SETUP_CAMERA_SHARED_DEFS(RadialCameraModel, "RADIAL", 3);
SETUP_CAMERA_SHARED_DEFS(OpenCVCameraModel, "OPENCV", 4);
SETUP_CAMERA_SHARED_DEFS(OpenCVFisheyeCameraModel, "OPENCV_FISHEYE", 5);
SETUP_CAMERA_SHARED_DEFS(FullOpenCVCameraModel, "FULL_OPENCV", 6);
SETUP_CAMERA_SHARED_DEFS(FOVCameraModel, "FOV", 7);
SETUP_CAMERA_SHARED_DEFS(SimpleRadialFisheyeCameraModel, "SIMPLE_RADIAL_FISHEYE", 8);
SETUP_CAMERA_SHARED_DEFS(RadialFisheyeCameraModel, "RADIAL_FISHEYE", 9);
SETUP_CAMERA_SHARED_DEFS(ThinPrismFisheyeCameraModel, "THIN_PRISM_FISHEYE", 10);
SETUP_CAMERA_SHARED_DEFS(Radial1DCameraModel, "1D_RADIAL", 11);
SETUP_CAMERA_SHARED_DEFS(SphericalCameraModel, "SPHERICAL", 100);
SETUP_CAMERA_SHARED_DEFS(DivisionCameraModel, "DIVISION", 101);

#define SWITCH_CAMERA_MODELS                                                                                           \
    SWITCH_CAMERA_MODEL_CASE(NullCameraModel)                                                                          \
    SWITCH_CAMERA_MODEL_CASE(SimplePinholeCameraModel)                                                                 \
    SWITCH_CAMERA_MODEL_CASE(PinholeCameraModel)                                                                       \
    SWITCH_CAMERA_MODEL_CASE(SimpleRadialCameraModel)                                                                  \
    SWITCH_CAMERA_MODEL_CASE(RadialCameraModel)                                                                        \
    SWITCH_CAMERA_MODEL_CASE(OpenCVCameraModel)                                                                        \
    SWITCH_CAMERA_MODEL_CASE(FullOpenCVCameraModel)                                                                    \
    SWITCH_CAMERA_MODEL_CASE(OpenCVFisheyeCameraModel)                                                                 \
    SWITCH_CAMERA_MODEL_CASE(FOVCameraModel)                                                                           \
    SWITCH_CAMERA_MODEL_CASE(SimpleRadialFisheyeCameraModel)                                                           \
    SWITCH_CAMERA_MODEL_CASE(RadialFisheyeCameraModel)                                                                 \
    SWITCH_CAMERA_MODEL_CASE(ThinPrismFisheyeCameraModel)                                                              \
    SWITCH_CAMERA_MODEL_CASE(Radial1DCameraModel)                                                                      \
    SWITCH_CAMERA_MODEL_CASE(SphericalCameraModel)                                                                     \
    SWITCH_CAMERA_MODEL_CASE(DivisionCameraModel);

#undef SETUP_CAMERA_SHARED_DEFS

} // namespace poselib

#endif