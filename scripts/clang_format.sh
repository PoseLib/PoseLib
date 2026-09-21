#!/usr/bin/env bash
set -euo pipefail

if ! command -v clang-format &> /dev/null; then
    if command -v uv &> /dev/null; then
        echo "clang-format not found in PATH. Install clang-format or run this script with: uv run ./scripts/clang_format.sh"
    else
        echo "clang-format not found in PATH. Install clang-format and rerun this script."
    fi
    exit 1
fi

echo "Using $(clang-format --version)"

# Determine script absolute path
SCRIPT_ABS_PATH=$(readlink -f "${BASH_SOURCE[0]}")
SCRIPT_ABS_PATH=$(dirname "${SCRIPT_ABS_PATH}")

# root folder where top-level CMakeLists.txt lives
ROOT="$(readlink -f "${SCRIPT_ABS_PATH}/..")"

find "${ROOT}/PoseLib" \( -iname "*.h" -o -iname "*.cc" \) -print0 | xargs -0 -r clang-format -i --verbose
find "${ROOT}/benchmark" \( -iname "*.h" -o -iname "*.cc" \) -print0 | xargs -0 -r clang-format -i --verbose
find "${ROOT}/tests" \( -iname "*.h" -o -iname "*.cc" \) -print0 | xargs -0 -r clang-format -i --verbose
clang-format -i --verbose "${ROOT}/pybind/pyposelib.cc"
clang-format -i --verbose "${ROOT}/pybind/pybind11_extension.h"
clang-format -i --verbose "${ROOT}/pybind/helpers.h"
