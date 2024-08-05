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

#ifndef POSELIB_COLMAP_MODELS_H_
#define POSELIB_COLMAP_MODELS_H_

#include <Eigen/Dense>
#include <vector>

namespace poselib {
struct Camera {
    int model_id;
    int width;
    int height;
    std::vector<double> params;

    Camera();
    Camera(const std::string &model_name, const std::vector<double> &params, int width, int height);
    Camera(int model_id, const std::vector<double> &params, int width, int height);

    // Projection and distortion
    void project(const Eigen::Vector2d &x, Eigen::Vector2d *xp) const;
    void project_with_jac(const Eigen::Vector2d &x, Eigen::Vector2d *xp, Eigen::Matrix2d *jac) const;
    void unproject(const Eigen::Vector2d &xp, Eigen::Vector2d *x) const;

    // vector wrappers for the project/unprojection
    void project(const std::vector<Eigen::Vector2d> &x, std::vector<Eigen::Vector2d> *xp) const;
    void project_with_jac(const std::vector<Eigen::Vector2d> &x, std::vector<Eigen::Vector2d> *xp,
                          std::vector<Eigen::Matrix<double, 2, 2>> *jac) const;
    void unproject(const std::vector<Eigen::Vector2d> &xp, std::vector<Eigen::Vector2d> *x) const;

    // Update the camera parameters such that the projections are rescaled
    void rescale(double scale);
    // Return camera model as string
    std::string model_name() const;
    // Returns focal length (average in case of non-unit aspect ratio)
    double focal() const;
    double focal_x() const;
    double focal_y() const;
    Eigen::Vector2d principal_point() const;

    // Parses a camera from a line from cameras.txt, returns the camera_id
    int initialize_from_txt(const std::string &line);
    // Creates line for cameras.txt (inverse of initialize_from_txt)
    // If camera_id == -1 it is ommited
    std::string to_cameras_txt(int camera_id = -1) const;

    // helpers for camera model ids
    static int id_from_string(const std::string &model_name);
    static std::string name_from_id(int id);
};

#define SETUP_CAMERA_SHARED_DEFS(ClassName, ModelName, ModelId)                                                        \
    class ClassName {                                                                                                  \
      public:                                                                                                          \
        static void project(const std::vector<double> &params, const Eigen::Vector2d &x, Eigen::Vector2d *xp);         \
        static void project_with_jac(const std::vector<double> &params, const Eigen::Vector2d &x, Eigen::Vector2d *xp, \
                                     Eigen::Matrix2d *jac);                                                            \
        static void unproject(const std::vector<double> &params, const Eigen::Vector2d &xp, Eigen::Vector2d *x);       \
        static const std::vector<size_t> focal_idx;                                                                    \
        static const std::vector<size_t> principal_point_idx;                                                          \
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

#define SWITCH_CAMERA_MODELS                                                                                           \
    SWITCH_CAMERA_MODEL_CASE(NullCameraModel)                                                                          \
    SWITCH_CAMERA_MODEL_CASE(SimplePinholeCameraModel)                                                                 \
    SWITCH_CAMERA_MODEL_CASE(PinholeCameraModel)                                                                       \
    SWITCH_CAMERA_MODEL_CASE(SimpleRadialCameraModel)                                                                  \
    SWITCH_CAMERA_MODEL_CASE(RadialCameraModel)                                                                        \
    SWITCH_CAMERA_MODEL_CASE(OpenCVCameraModel)                                                                        \
    SWITCH_CAMERA_MODEL_CASE(OpenCVFisheyeCameraModel)

// TODO add more models
// SETUP_CAMERA_SHARED_DEFS(OpenCVCameraModel , "OPENCV", 4);
// SETUP_CAMERA_SHARED_DEFS(FullOpenCVCameraModel, "FULL_OPENCV", 6);

#undef SETUP_CAMERA_SHARED_DEFS

} // namespace poselib

#endif