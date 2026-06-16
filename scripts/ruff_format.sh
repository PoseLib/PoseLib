#!/usr/bin/env bash
set -euo pipefail

# Check ruff is available
if ! command -v ruff &> /dev/null; then
    echo "ruff not found. Please install it: pip install ruff"
    exit 1
fi

echo "Using $(ruff --version)"

# Determine script absolute path
SCRIPT_ABS_PATH=$(readlink -f ${BASH_SOURCE[0]})
SCRIPT_ABS_PATH=$(dirname ${SCRIPT_ABS_PATH})

# root folder
ROOT="$(readlink -f "${SCRIPT_ABS_PATH}"/..)"

ruff check --fix "${ROOT}"
ruff format "${ROOT}"
