#include "../helpers.h"
#include "../pybind11_extension.h"

#include <PoseLib/poselib.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace poselib {
namespace {

std::vector<CameraPose> p3p_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    p3p(x, X, &output);
    return output;
}

std::vector<CameraPose> gp3p_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                                     const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    gp3p(p, x, X, &output);
    return output;
}

std::pair<std::vector<CameraPose>, std::vector<double>> gp4ps_wrapper(const std::vector<Eigen::Vector3d> &p,
                                                                      const std::vector<Eigen::Vector3d> &x,
                                                                      const std::vector<Eigen::Vector3d> &X,
                                                                      bool filter_solutions) {
    std::vector<CameraPose> output;
    std::vector<double> output_scales;
    gp4ps(p, x, X, &output, &output_scales, filter_solutions);
    return std::make_pair(output, output_scales);
}

std::pair<std::vector<CameraPose>, std::vector<double>> gp4ps_kukelova_wrapper(const std::vector<Eigen::Vector3d> &p,
                                                                               const std::vector<Eigen::Vector3d> &x,
                                                                               const std::vector<Eigen::Vector3d> &X,
                                                                               bool filter_solutions) {
    std::vector<CameraPose> output;
    std::vector<double> output_scales;
    gp4ps_kukelova(p, x, X, &output, &output_scales, filter_solutions);
    return std::make_pair(output, output_scales);
}

std::pair<std::vector<CameraPose>, std::vector<double>> gp4ps_camposeco_wrapper(const std::vector<Eigen::Vector3d> &p,
                                                                                const std::vector<Eigen::Vector3d> &x,
                                                                                const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    std::vector<double> output_scales;
    gp4ps_camposeco(p, x, X, &output, &output_scales);
    return std::make_pair(output, output_scales);
}

std::pair<std::vector<CameraPose>, std::vector<double>>
p4pf_wrapper(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, bool filter_solutions) {
    std::vector<CameraPose> output;
    std::vector<double> output_focal;
    p4pf(x, X, &output, &output_focal, filter_solutions);
    return std::make_pair(output, output_focal);
}

std::vector<CameraPose> p2p2pl_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                                       const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
                                       const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    p2p2pl(xp, Xp, x, X, V, &output);
    return output;
}

std::vector<CameraPose> p6lp_wrapper(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    p6lp(l, X, &output);
    return output;
}

std::vector<CameraPose> p5lp_radial_wrapper(const std::vector<Eigen::Vector3d> &l,
                                            const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    p5lp_radial(l, X, &output);
    return output;
}

std::vector<CameraPose> p2p1ll_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                                       const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                                       const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    p2p1ll(xp, Xp, l, X, V, &output);
    return output;
}

std::vector<CameraPose> p1p2ll_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                                       const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                                       const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    p1p2ll(xp, Xp, l, X, V, &output);
    return output;
}

std::vector<CameraPose> p3ll_wrapper(const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                                     const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    p3ll(l, X, V, &output);
    return output;
}

std::pair<std::vector<CameraPose>, std::vector<double>>
p3p1llf_wrapper(const std::vector<Eigen::Vector2d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                const std::vector<Eigen::Vector3d> &V, bool filter_solutions) {
    std::vector<CameraPose> output;
    std::vector<double> output_focal;
    p3p1llf(xp, Xp, l, X, V, &output, &output_focal, filter_solutions);
    return std::make_pair(output, output_focal);
}

std::pair<std::vector<CameraPose>, std::vector<double>>
p2p2llf_wrapper(const std::vector<Eigen::Vector2d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                const std::vector<Eigen::Vector3d> &V, bool filter_solutions) {
    std::vector<CameraPose> output;
    std::vector<double> output_focal;
    p2p2llf(xp, Xp, l, X, V, &output, &output_focal, filter_solutions);
    return std::make_pair(output, output_focal);
}

std::pair<std::vector<CameraPose>, std::vector<double>>
p1p3llf_wrapper(const std::vector<Eigen::Vector2d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                const std::vector<Eigen::Vector3d> &l, const std::vector<Eigen::Vector3d> &X,
                const std::vector<Eigen::Vector3d> &V, bool filter_solutions) {
    std::vector<CameraPose> output;
    std::vector<double> output_focal;
    p1p3llf(xp, Xp, l, X, V, &output, &output_focal, filter_solutions);
    return std::make_pair(output, output_focal);
}

std::pair<std::vector<CameraPose>, std::vector<double>> p4llf_wrapper(const std::vector<Eigen::Vector3d> &l,
                                                                      const std::vector<Eigen::Vector3d> &X,
                                                                      const std::vector<Eigen::Vector3d> &V,
                                                                      bool filter_solutions) {
    std::vector<CameraPose> output;
    std::vector<double> output_focal;
    p4llf(l, X, V, &output, &output_focal, filter_solutions);
    return std::make_pair(output, output_focal);
}

std::vector<CameraPose> up2p_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    up2p(x, X, &output);
    return output;
}

std::vector<CameraPose> ugp2p_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                                      const std::vector<Eigen::Vector3d> &X) {
    std::vector<CameraPose> output;
    ugp2p(p, x, X, &output);
    return output;
}

std::pair<std::vector<CameraPose>, std::vector<double>> ugp3ps_wrapper(const std::vector<Eigen::Vector3d> &p,
                                                                       const std::vector<Eigen::Vector3d> &x,
                                                                       const std::vector<Eigen::Vector3d> &X,
                                                                       bool filter_solutions) {
    std::vector<CameraPose> output;
    std::vector<double> output_scales;
    ugp3ps(p, x, X, &output, &output_scales, filter_solutions);
    return std::make_pair(output, output_scales);
}

std::vector<CameraPose> up1p2pl_wrapper(const std::vector<Eigen::Vector3d> &xp, const std::vector<Eigen::Vector3d> &Xp,
                                        const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
                                        const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    up1p2pl(xp, Xp, x, X, V, &output);
    return output;
}

std::vector<CameraPose> up4pl_wrapper(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
                                      const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    up4pl(x, X, V, &output);
    return output;
}

std::vector<CameraPose> ugp4pl_wrapper(const std::vector<Eigen::Vector3d> &p, const std::vector<Eigen::Vector3d> &x,
                                       const std::vector<Eigen::Vector3d> &X, const std::vector<Eigen::Vector3d> &V) {
    std::vector<CameraPose> output;
    ugp4pl(p, x, X, V, &output);
    return output;
}

std::vector<Eigen::Matrix3d> essential_matrix_relpose_5pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                                  const std::vector<Eigen::Vector3d> &x2) {
    std::vector<Eigen::Matrix3d> essential_matrices;
    relpose_5pt(x1, x2, &essential_matrices);
    return essential_matrices;
}

std::vector<CameraPose> relpose_5pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                            const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_5pt(x1, x2, &output);
    return output;
}

std::vector<MonoDepthTwoViewGeometry> monodepth_relpose_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                                    const std::vector<Eigen::Vector3d> &x2,
                                                                    const std::vector<double> &d1,
                                                                    const std::vector<double> &d2) {
    std::vector<MonoDepthTwoViewGeometry> output;
    relpose_monodepth_3pt(x1, x2, d1, d2, &output);
    return output;
}

ImagePairVector shared_focal_relpose_6pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                 const std::vector<Eigen::Vector3d> &x2) {
    ImagePairVector output;
    relpose_6pt_shared_focal(x1, x2, &output);
    return output;
}

std::vector<MonoDepthImagePair> shared_focal_monodepth_relpose_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                                           const std::vector<Eigen::Vector3d> &x2,
                                                                           const std::vector<double> &d1,
                                                                           const std::vector<double> &d2) {
    std::vector<MonoDepthImagePair> output;
    relpose_monodepth_3pt_shared_focal(x1, x2, d1, d2, &output);
    return output;
}

std::vector<MonoDepthImagePair> varying_focal_monodepth_relpose_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                                            const std::vector<Eigen::Vector3d> &x2,
                                                                            const std::vector<double> &d1,
                                                                            const std::vector<double> &d2) {
    std::vector<MonoDepthImagePair> output;
    relpose_monodepth_3pt_varying_focal(x1, x2, d1, d2, &output);
    return output;
}

std::vector<CameraPose> relpose_8pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                            const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_8pt(x1, x2, &output);
    return output;
}

Eigen::Matrix3d essential_matrix_8pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                             const std::vector<Eigen::Vector3d> &x2) {
    Eigen::Matrix3d essential_matrix;
    essential_matrix_8pt(x1, x2, &essential_matrix);
    return essential_matrix;
}

std::vector<CameraPose> relpose_upright_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                    const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_upright_3pt(x1, x2, &output);
    return output;
}

std::vector<CameraPose> gen_relpose_upright_4pt_wrapper(const std::vector<Eigen::Vector3d> &p1,
                                                        const std::vector<Eigen::Vector3d> &x1,
                                                        const std::vector<Eigen::Vector3d> &p2,
                                                        const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    gen_relpose_upright_4pt(p1, x1, p2, x2, &output);
    return output;
}

std::vector<CameraPose> gen_relpose_6pt_wrapper(const std::vector<Eigen::Vector3d> &p1,
                                                const std::vector<Eigen::Vector3d> &x1,
                                                const std::vector<Eigen::Vector3d> &p2,
                                                const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    gen_relpose_6pt(p1, x1, p2, x2, &output);
    return output;
}

std::vector<CameraPose> relpose_upright_planar_2pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                           const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_upright_planar_2pt(x1, x2, &output);
    return output;
}

std::vector<CameraPose> relpose_upright_planar_3pt_wrapper(const std::vector<Eigen::Vector3d> &x1,
                                                           const std::vector<Eigen::Vector3d> &x2) {
    std::vector<CameraPose> output;
    relpose_upright_planar_3pt(x1, x2, &output);
    return output;
}

} // namespace

void register_solvers(py::module &m) {
    // Minimal solvers
    m.def("p3p", &p3p_wrapper, py::arg("x"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("gp3p", &gp3p_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("gp4ps", &gp4ps_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("filter_solutions"),
          py::call_guard<py::gil_scoped_release>());
    m.def("gp4ps_kukelova", &gp4ps_kukelova_wrapper, py::arg("p"), py::arg("x"), py::arg("X"),
          py::arg("filter_solutions"), py::call_guard<py::gil_scoped_release>());
    m.def("gp4ps_camposeco", &gp4ps_camposeco_wrapper, py::arg("p"), py::arg("x"), py::arg("X"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p4pf", &p4pf_wrapper, py::arg("x"), py::arg("X"), py::arg("filter_solutions"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p2p2pl", &p2p2pl_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("x"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p6lp", &p6lp_wrapper, py::arg("l"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("p5lp_radial", &p5lp_radial_wrapper, py::arg("l"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("p2p1ll", &p2p1ll_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p1p2ll", &p1p2ll_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("p3ll", &p3ll_wrapper, py::arg("l"), py::arg("X"), py::arg("V"), py::call_guard<py::gil_scoped_release>());
    m.def("p3p1llf", &p3p1llf_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"),
          py::arg("filter_solutions"), py::call_guard<py::gil_scoped_release>());
    m.def("p2p2llf", &p2p2llf_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"),
          py::arg("filter_solutions"), py::call_guard<py::gil_scoped_release>());
    m.def("p1p3llf", &p1p3llf_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("l"), py::arg("X"), py::arg("V"),
          py::arg("filter_solutions"), py::call_guard<py::gil_scoped_release>());
    m.def("p4llf", &p4llf_wrapper, py::arg("l"), py::arg("X"), py::arg("V"), py::arg("filter_solutions"),
          py::call_guard<py::gil_scoped_release>());
    m.def("up2p", &up2p_wrapper, py::arg("x"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("ugp2p", &ugp2p_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::call_guard<py::gil_scoped_release>());
    m.def("ugp3ps", &ugp3ps_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("filter_solutions"),
          py::call_guard<py::gil_scoped_release>());
    m.def("up1p2pl", &up1p2pl_wrapper, py::arg("xp"), py::arg("Xp"), py::arg("x"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("up4pl", &up4pl_wrapper, py::arg("x"), py::arg("X"), py::arg("V"), py::call_guard<py::gil_scoped_release>());
    m.def("ugp4pl", &ugp4pl_wrapper, py::arg("p"), py::arg("x"), py::arg("X"), py::arg("V"),
          py::call_guard<py::gil_scoped_release>());
    m.def("essential_matrix_5pt", &essential_matrix_relpose_5pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("shared_focal_relpose_6pt", &shared_focal_relpose_6pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("shared_focal_monodepth_pose_3pt", &shared_focal_monodepth_relpose_3pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::arg("d1"), py::arg("d2"), py::call_guard<py::gil_scoped_release>());
    m.def("varying_focal_monodepth_pose_4pt", &varying_focal_monodepth_relpose_3pt_wrapper, py::arg("x1"),
          py::arg("x2"), py::arg("d1"), py::arg("d2"), py::call_guard<py::gil_scoped_release>());
    m.def("relpose_5pt", &relpose_5pt_wrapper, py::arg("x1"), py::arg("x2"), py::call_guard<py::gil_scoped_release>());
    m.def("monodepth_pose_3pt", &monodepth_relpose_3pt_wrapper, py::arg("x1"), py::arg("x2"), py::arg("d1"),
          py::arg("d2"), py::call_guard<py::gil_scoped_release>());
    m.def("relpose_8pt", &relpose_8pt_wrapper, py::arg("x1"), py::arg("x2"), py::call_guard<py::gil_scoped_release>());
    m.def("essential_matrix_8pt", &essential_matrix_8pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("relpose_upright_3pt", &relpose_upright_3pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("gen_relpose_upright_4pt", &gen_relpose_upright_4pt_wrapper, py::arg("p1"), py::arg("x1"), py::arg("p2"),
          py::arg("x2"), py::call_guard<py::gil_scoped_release>());
    m.def("gen_relpose_6pt", &gen_relpose_6pt_wrapper, py::arg("p1"), py::arg("x1"), py::arg("p2"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("relpose_upright_planar_2pt", &relpose_upright_planar_2pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
    m.def("relpose_upright_planar_3pt", &relpose_upright_planar_3pt_wrapper, py::arg("x1"), py::arg("x2"),
          py::call_guard<py::gil_scoped_release>());
}

} // namespace poselib
