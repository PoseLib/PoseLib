#!/bin/bash

# Format all .cc and .h files tracked by git
git ls-files | grep -E '\.(cc|h)$' | xargs clang-format-18 -i --verbose

# Display clang-format version
clang-format-18 --version
