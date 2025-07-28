#!/bin/bash
# Adapted from https://github.com/colmap/colmap/blob/1ec758f2eb028049129b852d4d020d7542c866c9/python/generate_stubs.sh
set -e
PYTHON_EXEC=$1
OUTPUT=$2
PACKAGE_NAME="_core"
echo "Building stubs with $PYTHON_EXEC to $OUTPUT"

# Run pybind11_stubgen and check if it succeeds
$PYTHON_EXEC -m pybind11_stubgen $PACKAGE_NAME -o $OUTPUT \
        --numpy-array-use-type-var \
        --enum-class-locations=.+:$PACKAGE_NAME \
        --ignore-invalid-expressions "poselib::*" \
        --print-invalid-expressions-as-is \
        --print-safe-value-reprs "[a-zA-Z]+Options\(\)"

# Check for the stub file - pybind11_stubgen now creates files directly in the output directory
if [ -f "$OUTPUT/$PACKAGE_NAME.pyi" ]; then
    FILES="$OUTPUT/$PACKAGE_NAME.pyi"
elif [ -d "$OUTPUT/$PACKAGE_NAME" ]; then
    FILES=$(find $OUTPUT/$PACKAGE_NAME/ -name '*.pyi' -type f)
else
    echo "Error: Neither stub file $OUTPUT/$PACKAGE_NAME.pyi nor directory $OUTPUT/$PACKAGE_NAME exists"
    echo "Available files in $OUTPUT:"
    ls -la $OUTPUT/
    exit 1
fi

# Replace the long pybind11 module names with poselib._core
perl -i -pe's/pybind11_detail_function_record_v1_system_libcpp_abi1/poselib._core/g' $FILES
perl -i -pe's/\b_core\b/poselib._core/g' $FILES

# Clean up any poselib:: references
perl -i -pe's/: poselib::([a-zA-Z]|::)+//g' $FILES
perl -i -pe's/ -> poselib::([a-zA-Z]|::)+:$/:/g' $FILES

# pybind issue, will not be fixed: https://github.com/pybind/pybind11/pull/2277
perl -i -pe's/(?<=\b__(eq|ne)__\(self, )arg0: [a-zA-Z0-9_]+\)/other: object)/g' $FILES

# mypy bug: https://github.com/python/mypy/issues/4266
perl -i -pe's/(__hash__:? .*= None)$/\1  # type: ignore/g' $FILES

# Format the files if ruff is available
if command -v ruff &> /dev/null; then
    echo "Formatting stubs with ruff..."
    ruff format $FILES || echo "Warning: ruff formatting failed, continuing anyway"
else
    echo "ruff not found, skipping formatting"
fi 