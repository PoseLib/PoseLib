#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>


#include <PoseLib/types.h>
#include <PoseLib/p3p.h>
#include <PoseLib/gp3p.h>
#include <PoseLib/gp4ps.h>


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
  m.def("p3p", &pose_lib::p3p_wrapper);
  m.def("gp3p", &pose_lib::gp3p_wrapper);
  m.def("gp4ps", &pose_lib::gp4ps_wrapper);
  m.def("gp4ps_kukelova", &pose_lib::gp4ps_kukelova_wrapper);
  m.def("gp4ps_camposeco", &pose_lib::gp4ps_camposeco_wrapper);
}
