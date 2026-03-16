#!/usr/bin/env bash
set -euo pipefail

# Determine script absolute path
SCRIPT_ABS_PATH=$(readlink -f ${BASH_SOURCE[0]})
SCRIPT_ABS_PATH=$(dirname ${SCRIPT_ABS_PATH})

# root folder
ROOT="$(readlink -f ${SCRIPT_ABS_PATH}/..)"

echo "Using $(ruff --version)"

ruff check --fix ${ROOT}
ruff format ${ROOT}
