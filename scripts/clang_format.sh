#!/bin/bash
set -euo pipefail

REQUIRED_VERSION="21.1.8"

# Check clang-format version
VERSION=$(clang-format --version)
if ! echo "$VERSION" | grep -q "version $REQUIRED_VERSION"; then
    echo "Error: clang-format version $REQUIRED_VERSION required, found: $VERSION"
    echo "Install the correct version with: pip install clang-format==$REQUIRED_VERSION"
    exit 1
fi

echo "Using $VERSION"

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
