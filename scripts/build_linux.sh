#!/bin/bash

# This script can be called from anywhere and allows to build out of source.


# Determine script absolute path
SCRIPT_ABS_PATH=$(readlink -f ${BASH_SOURCE[0]})
SCRIPT_ABS_PATH=$(dirname ${SCRIPT_ABS_PATH})

# root folder where top-level CMakeLists.txt lives
ROOT="${SCRIPT_ABS_PATH}/../"

# Build type
BUILD_TYPE=Release
# BUILD_TYPE=Debug

# Build folder
BUILD_DIR=_build_linux

# Installation folder
INSTALL_DIR=_install_linux

# Library type
BUILD_SHARED_LIBS=OFF    # Static
# BUILD_SHARED_LIBS=ON   # Shared

# Number of cores
NUM_CORES=$(nproc)

# Options summary
echo ""
echo "BUILD_TYPE        =" ${BUILD_TYPE}
echo "BUILD_DIR         =" ${SCRIPT_ABS_PATH}/${BUILD_DIR}/
echo "INSTALL_DIR       =" ${SCRIPT_ABS_PATH}/${INSTALL_DIR}/
echo "BUILD_SHARED_LIBS =" ${BUILD_SHARED_LIBS}
echo ""


# switch to ROOT path
cd ${ROOT}

# clean
# rm -fr ${BUILD_DIR} ${INSTALL_DIR}

# cmake
cmake \
    -S . \
    -B ${BUILD_DIR} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS} \
    -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}"

# compile & install
cmake \
    --build ${BUILD_DIR} \
    --target install \
    -j $NUM_CORES
