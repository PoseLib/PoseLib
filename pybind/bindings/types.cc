#include "../helpers.h"
#include "../pybind11_extension.h"

#include <PoseLib/poselib.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace poselib {

static py::dict RansacOptions_wrapper(py::dict overwrite) {
    RansacOptions opt;
    update_ransac_options(overwrite, opt);
    py::dict result;
    write_to_dict(opt, result);
    return result;
}

static py::dict BundleOptions_wrapper(py::dict overwrite) {
    BundleOptions opt;
    update_bundle_options(overwrite, opt);
    py::dict result;
    write_to_dict(opt, result);
    return result;
}

void register_types(py::module &m) {
    py::classh<CameraPose>(m, "CameraPose")
        .def(py::init<>())
        .def_readwrite("q", &CameraPose::q)
        .def_readwrite("t", &CameraPose::t)
        .def_property("R", &CameraPose::R,
                      [](CameraPose &self, Eigen::Matrix3d R_new) { self.q = rotmat_to_quat(R_new); })
        .def_property("Rt", &CameraPose::Rt,
                      [](CameraPose &self, Eigen::Matrix<double, 3, 4> Rt_new) {
                          self.q = rotmat_to_quat(Rt_new.leftCols<3>());
                          self.t = Rt_new.col(3);
                      })
        .def("center", &CameraPose::center, "Returns the camera center (c=-R^T*t).")
        .def("__repr__", [](const CameraPose &a) {
            return "[q: " + toString(a.q.transpose()) + ", " + "t: " + toString(a.t.transpose()) + "]";
        });

    py::classh<MonoDepthTwoViewGeometry>(m, "MonoDepthTwoViewGeometry")
        .def(py::init<>())
        .def(py::init<const Eigen::Vector4d &, const Eigen::Vector3d &, double, double, double>())
        .def(py::init<const CameraPose &, double, double, double>())
        .def(py::init<const CameraPose &>())
        .def_readwrite("pose", &MonoDepthTwoViewGeometry::pose)
        .def_readwrite("scale", &MonoDepthTwoViewGeometry::scale)
        .def_readwrite("shift1", &MonoDepthTwoViewGeometry::shift1)
        .def_readwrite("shift2", &MonoDepthTwoViewGeometry::shift2)
        .def("__repr__", [](const MonoDepthTwoViewGeometry &a) {
            return "[q: " + toString(a.pose.q.transpose()) + ", " + "t: " + toString(a.pose.t.transpose()) + ", " +
                   "scale: " + std::to_string(a.scale) + ", " + "shift1: " + std::to_string(a.shift1) + ", " +
                   "shift2: " + std::to_string(a.shift2) + "]";
        });

    py::classh<Camera>(m, "Camera")
        .def(py::init<>())
        .def(py::init<const std::string &, const std::vector<double> &, int, int>())
        .def_readwrite("model_id", &Camera::model_id)
        .def_readwrite("width", &Camera::width)
        .def_readwrite("height", &Camera::height)
        .def_readwrite("params", &Camera::params)
        .def("focal", &Camera::focal, "Returns the camera focal length.")
        .def("focal_x", &Camera::focal_x, "Returns the camera focal_x.")
        .def("focal_y", &Camera::focal_y, "Returns the camera focal_y.")
        .def("model_name", &Camera::model_name, "Returns the camera model name.")
        .def("principal_point", &Camera::principal_point, "Returns the camera principal point.")
        .def("initialize_from_txt", &Camera::initialize_from_txt, "Initialize camera from a cameras.txt line")
        .def("project",
             [](Camera &self, std::vector<Eigen::Vector2d> &xp) {
                 std::vector<Eigen::Vector2d> x;
                 self.project(xp, &x);
                 return x;
             })
        .def("project_with_jac",
             [](Camera &self, std::vector<Eigen::Vector2d> &xp) {
                 std::vector<Eigen::Vector2d> x;
                 std::vector<Eigen::Matrix2d> jac;
                 self.project_with_jac(xp, &x, &jac);
                 return std::make_pair(x, jac);
             })
        .def("unproject",
             [](Camera &self, std::vector<Eigen::Vector2d> &x) {
                 std::vector<Eigen::Vector2d> xp;
                 self.unproject(x, &xp);
                 return xp;
             })
        .def("__repr__", [](const Camera &a) { return a.to_cameras_txt(); });

    py::classh<Image>(m, "Image")
        .def(py::init<>())
        .def_readwrite("camera", &Image::camera)
        .def_readwrite("pose", &Image::pose)
        .def("__repr__", [](const Image &a) {
            return "[pose q: " + toString(a.pose.q.transpose()) + ", t: " + toString(a.pose.t.transpose()) +
                   ", camera: " + a.camera.to_cameras_txt() + "]";
        });

    py::classh<ImagePair>(m, "ImagePair")
        .def(py::init<>())
        .def_readwrite("pose", &ImagePair::pose)
        .def_readwrite("camera1", &ImagePair::camera1)
        .def_readwrite("camera2", &ImagePair::camera2)
        .def("__repr__", [](const ImagePair &a) {
            return "[pose q: " + toString(a.pose.q.transpose()) + ", t: " + toString(a.pose.t.transpose()) +
                   ", camera1: " + a.camera1.to_cameras_txt() + ", camera2: " + a.camera2.to_cameras_txt() + "]";
        });

    py::classh<MonoDepthImagePair>(m, "MonoDepthImagePair")
        .def(py::init<>())
        .def_readwrite("geometry", &MonoDepthImagePair::geometry)
        .def_readwrite("camera1", &MonoDepthImagePair::camera1)
        .def_readwrite("camera2", &MonoDepthImagePair::camera2)
        .def("__repr__", [](const MonoDepthImagePair &a) {
            return "[[pose q: " + toString(a.geometry.pose.q.transpose()) +
                   ", t: " + toString(a.geometry.pose.t.transpose()) + ", scale: " + std::to_string(a.geometry.scale) +
                   ", shift1: " + std::to_string(a.geometry.shift1) + ", shift2: " + std::to_string(a.geometry.shift2) +
                   "], camera1: " + a.camera1.to_cameras_txt() + ", camera2: " + a.camera2.to_cameras_txt() + "]";
        });

    py::classh<PairwiseMatches>(m, "PairwiseMatches")
        .def(py::init<>())
        .def_readwrite("cam_id1", &PairwiseMatches::cam_id1)
        .def_readwrite("cam_id2", &PairwiseMatches::cam_id2)
        .def_readwrite("x1", &PairwiseMatches::x1)
        .def_readwrite("x2", &PairwiseMatches::x2)
        .def("__repr__", [](const PairwiseMatches &a) {
            return "[cam_id1: " + std::to_string(a.cam_id1) + "\n" + "cam_id2: " + std::to_string(a.cam_id2) + "\n" +
                   "x1: [2x" + std::to_string(a.x1.size()) + "]\n" + "x2: [2x" + std::to_string(a.x2.size()) + "]]\n";
        });

    // Options factory functions
    m.def("RansacOptions", &RansacOptions_wrapper, py::arg("opt") = py::dict(), "Options for RANSAC.");
    m.def("BundleOptions", &BundleOptions_wrapper, py::arg("opt") = py::dict(),
          "Options for non-linear refinement.");
}

} // namespace poselib
