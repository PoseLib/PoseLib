#!/bin/bash

# Determine script absolute path
SCRIPT_ABS_PATH=$(readlink -f ${BASH_SOURCE[0]})
SCRIPT_ABS_PATH=$(dirname ${SCRIPT_ABS_PATH})

# root folder where top-level CMakeLists.txt lives
ROOT="$(readlink -f ${SCRIPT_ABS_PATH}/..)"


find ${ROOT}/PoseLib -iname "*.h" -o -iname "*.cc" | xargs clang-format -i --verbose
find ${ROOT}/benchmark -iname "*.h" -o -iname "*.cc" | xargs clang-format -i --verbose
clang-format -i --verbose ${ROOT}/pybind/pyposelib.cc
clang-format -i --verbose ${ROOT}/pybind/pybind11_extension.h
clang-format -i --verbose ${ROOT}/pybind/helpers.h

echo $(clang-format --version)
