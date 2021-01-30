cmake_minimum_required(VERSION 3.13)
project(pyposelib)

find_package(Python3 REQUIRED)
find_package(PoseLib REQUIRED)
find_package(Eigen3 REQUIRED)

include(CMakePrintHelpers)
cmake_print_variables(PYTHON3_INCLUDE_DIR)
cmake_print_variables(EIGEN3_INCLUDE_DIR)
cmake_print_variables(POSELIB_INCLUDE_DIR)

configure_file(
    "setup.py.in"
    "setup.py"
     INSTALL_DESTINATION "."
)
