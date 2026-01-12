#include "helpers.h"
#include "pybind11_extension.h"

#include <PoseLib/poselib.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace poselib {

// Forward declarations of registration functions
void register_types(py::module &m);
void register_solvers(py::module &m);
void register_absolute_pose(py::module &m);
void register_relative_pose(py::module &m);
void register_homography(py::module &m);
void register_hybrid_pose(py::module &m);

} // namespace poselib

PYBIND11_MODULE(_core, m) {
    m.doc() = "This library provides a collection of minimal solvers for camera pose estimation.";

    // Register all bindings
    poselib::register_types(m);
    poselib::register_solvers(m);
    poselib::register_absolute_pose(m);
    poselib::register_relative_pose(m);
    poselib::register_homography(m);
    poselib::register_hybrid_pose(m);

    m.attr("__version__") = std::string(POSELIB_VERSION);
}
