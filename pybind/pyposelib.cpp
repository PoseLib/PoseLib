#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include <PoseLib/version.h>
#include <PoseLib/types.h>
#include <PoseLib/gen_relpose_upright_4pt.h>
#include <PoseLib/gp3p.h>
#include <PoseLib/gp4ps.h>
#include <PoseLib/p1p2ll.h>
#include <PoseLib/p2p1ll.h>
#include <PoseLib/p2p2pl.h>
#include <PoseLib/p3ll.h>
#include <PoseLib/p3p.h>
#include <PoseLib/p4pf.h>
#include <PoseLib/p5lp_radial.h>
#include <PoseLib/p6lp.h>
#include <PoseLib/relpose_5pt.h>
#include <PoseLib/relpose_8pt.h>
#include <PoseLib/relpose_upright_3pt.h>
#include <PoseLib/relpose_upright_planar_2pt.h>
#include <PoseLib/relpose_upright_planar_3pt.h>
#include <PoseLib/ugp2p.h>
#include <PoseLib/ugp3ps.h>
#include <PoseLib/ugp4pl.h>
#include <PoseLib/up1p2pl.h>
#include <PoseLib/up2p.h>
#include <PoseLib/up4pl.h>

static std::string toString(const Eigen::MatrixXd& mat){
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

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

}

namespace py = pybind11;

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
                            "alpha: " + std::to_string(a.alpha) + "]";
                }
            );

  m.doc() = "This library provides a collection of minimal solvers for camera pose estimation.";
  m.def("p3p", &pose_lib::p3p_wrapper);
  m.def("gp3p", &pose_lib::gp3p_wrapper);
  m.def("gp4ps", &pose_lib::gp4ps_wrapper);
  m.def("gp4ps_kukelova", &pose_lib::gp4ps_kukelova_wrapper);
  m.def("gp4ps_camposeco", &pose_lib::gp4ps_camposeco_wrapper);
  m.def("p4pf", &pose_lib::p4pf_wrapper);
  m.def("p2p2pl", &pose_lib::p2p2pl_wrapper);
  m.def("p6lp", &pose_lib::p6lp_wrapper);
  m.def("p5lp_radial", &pose_lib::p5lp_radial_wrapper);
  m.def("p2p1ll", &pose_lib::p2p1ll_wrapper);
  m.def("p1p2ll", &pose_lib::p1p2ll_wrapper);
  m.def("p3ll", &pose_lib::p3ll_wrapper);
  m.def("up2p", &pose_lib::up2p_wrapper);
  m.def("ugp2p", &pose_lib::ugp2p_wrapper);
  m.def("ugp3ps", &pose_lib::ugp3ps_wrapper);
  m.def("up1p2pl", &pose_lib::up1p2pl_wrapper);
  m.def("up4pl", &pose_lib::up4pl_wrapper);
  m.def("ugp4pl", &pose_lib::ugp4pl_wrapper);
  m.def("essential_matrix_relpose_5pt", &pose_lib::essential_matrix_relpose_5pt_wrapper);
  m.def("relpose_5pt", &pose_lib::relpose_5pt_wrapper);
  m.def("relpose_8pt", &pose_lib::relpose_8pt_wrapper);
  m.def("essential_matrix_8pt", &pose_lib::essential_matrix_8pt_wrapper);
  m.def("relpose_upright_3pt", &pose_lib::relpose_upright_3pt_wrapper);
  m.def("gen_relpose_upright_4pt", &pose_lib::gen_relpose_upright_4pt_wrapper);
  m.def("relpose_upright_planar_2pt", &pose_lib::relpose_upright_planar_2pt_wrapper);
  m.def("relpose_upright_planar_3pt", &pose_lib::relpose_upright_planar_3pt_wrapper);
  m.attr("__version__") = std::string(POSELIB_VERSION);
}
