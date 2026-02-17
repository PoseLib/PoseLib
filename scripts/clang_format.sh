#!/usr/bin/env bash
set -euo pipefail

# Check version
version_string=$(clang-format --version | sed -E 's/^.*([0-9]+\.[0-9]+\.[0-9]+-.*).*$/\1/')
expected_version_string='21.1.8'
if [[ "$version_string" =~ "$expected_version_string" ]]; then
    echo "clang-format version '$version_string' matches '$expected_version_string'"
else
    echo "clang-format version '$version_string' doesn't match '$expected_version_string'"
    echo "Install the correct version with: pip install clang-format==$expected_version_string"
    exit 1
fi

# Determine script absolute path
SCRIPT_ABS_PATH=$(readlink -f ${BASH_SOURCE[0]})
SCRIPT_ABS_PATH=$(dirname ${SCRIPT_ABS_PATH})

# root folder where top-level CMakeLists.txt lives
ROOT="$(readlink -f ${SCRIPT_ABS_PATH}/..)"

find ${ROOT}/PoseLib \( -iname "*.h" -o -iname "*.cc" \) | xargs -r clang-format -i --verbose
find ${ROOT}/benchmark \( -iname "*.h" -o -iname "*.cc" \) | xargs -r clang-format -i --verbose
clang-format -i --verbose ${ROOT}/pybind/pyposelib.cc
clang-format -i --verbose ${ROOT}/pybind/pybind11_extension.h
clang-format -i --verbose ${ROOT}/pybind/helpers.h
