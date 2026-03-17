#!/usr/bin/env bash
set -euo pipefail

# Check clang-format is available
if ! uv pip show clang-format &> /dev/null; then
    echo "clang-format is not installed. Run: uv sync --locked --dev"
    exit 1
fi

if ! command -v clang-format &> /dev/null; then
    echo "clang-format not found. Run this script with: uv run ./scripts/clang_format.sh"
    exit 1
fi

echo "Using $(clang-format --version)"

# Determine script absolute path
SCRIPT_ABS_PATH=$(readlink -f ${BASH_SOURCE[0]})
SCRIPT_ABS_PATH=$(dirname ${SCRIPT_ABS_PATH})

# root folder where top-level CMakeLists.txt lives
ROOT="$(readlink -f ${SCRIPT_ABS_PATH}/..)"

find ${ROOT}/PoseLib \( -iname "*.h" -o -iname "*.cc" \) | xargs -r clang-format -i --verbose
find ${ROOT}/benchmark \( -iname "*.h" -o -iname "*.cc" \) | xargs -r clang-format -i --verbose
find ${ROOT}/tests \( -iname "*.h" -o -iname "*.cc" \) | xargs -r clang-format -i --verbose
clang-format -i --verbose ${ROOT}/pybind/pyposelib.cc
clang-format -i --verbose ${ROOT}/pybind/pybind11_extension.h
clang-format -i --verbose ${ROOT}/pybind/helpers.h
