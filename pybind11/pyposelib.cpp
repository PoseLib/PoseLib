#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


#include <PoseLib/types.h>
#include <PoseLib/gp4ps.h>


namespace pose_lib {
    
std::vector<CameraPose> gp4ps_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X, bool filter_solutions=true){
    std::vector<CameraPose> output;
    gp4ps(p, x, X, &output, filter_solutions);
    return output;
}
    
}

namespace py = pybind11;

PYBIND11_MODULE(poselib, m)
{
  py::class_<pose_lib::CameraPose>(m, "CameraPose")
            .def(py::init<>())
            .def_readwrite("R", &pose_lib::CameraPose::R)
            .def_readwrite("t", &pose_lib::CameraPose::t)
            .def_readwrite("alpha", &pose_lib::CameraPose::alpha);
    
  m.doc() = "pybind11 poselib";

  m.def("gp4ps", &pose_lib::gp4ps_wrapper);
}
