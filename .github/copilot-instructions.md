# PoseLib Development Instructions

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

PoseLib is a C++ library for camera pose estimation with Python bindings, providing minimal solvers for calibrated absolute pose estimation problems from different types of correspondences.

## Working Effectively

### Environment Setup
- Install required system dependencies:
  - `sudo apt-get update && sudo apt-get install -y libeigen3-dev clang-format build-essential`
  - `sudo apt-get install -y python3-numpy python3-dev python3-pybind11 python3-setuptools python3-wheel`
- Initialize git submodules (required for Python bindings):
  - `git submodule update --init --recursive`

### C++ Library Build
- Configure and build the C++ library:
  - `mkdir -p _build && cd _build`
  - `cmake -DCMAKE_BUILD_TYPE=Release -DWITH_BENCHMARK=OFF -DCMAKE_INSTALL_PREFIX=../_install ..`
  - `cmake --build . --target install -j $(nproc)` -- NEVER CANCEL: takes 125 seconds. Set timeout to 180+ seconds.
- Build with benchmark (adds benchmark binary):
  - `cmake -DCMAKE_BUILD_TYPE=Release -DWITH_BENCHMARK=ON -DCMAKE_INSTALL_PREFIX=../_install ..`
  - `cmake --build . --target install -j $(nproc)` -- NEVER CANCEL: takes 125 seconds. Set timeout to 180+ seconds.
- Use the provided build script (alternative approach):
  - `chmod +x scripts/build_linux.sh && cd scripts && ./build_linux.sh` -- NEVER CANCEL: takes 120 seconds. Set timeout to 180+ seconds.

### Python Package Build
- Build Python bindings manually (recommended for development):
  - Ensure submodules are initialized: `git submodule update --init --recursive`
  - `mkdir -p build && cd build`
  - `cmake -DPYTHON_PACKAGE=ON -DGENERATE_STUBS=OFF -DCMAKE_BUILD_TYPE=Release ..`
  - `cmake --build . --target pyposelib -j $(nproc)` -- NEVER CANCEL: takes 140 seconds. Set timeout to 200+ seconds.
  - Copy built module: `cp build/pybind/_core.cpython-*-linux-gnu.so pyposelib/`
- Install Python package with pip (requires internet access, may fail due to network issues):
  - `pip install .` -- Can fail due to network timeouts. Use manual build method above if this fails.

### Running Applications
- Execute the C++ benchmark (if built with `-DWITH_BENCHMARK=ON`):
  - `./_install/bin/benchmark` -- NEVER CANCEL: takes 35 seconds. Set timeout to 60+ seconds.
- Test Python functionality:
  - `PYTHONPATH=. python3 -c "import pyposelib as poselib; print('PoseLib version:', poselib.__version__)"`

## Validation

### Code Formatting
- Always run clang-format before committing or the CI will fail:
  - `git ls-files | grep -E '\.(cc|h)$' | xargs clang-format -i --verbose`
  - `git diff --exit-code` -- Should show no changes if formatting is correct.

### Testing
- Test C++ library by running benchmark:
  - Build with `-DWITH_BENCHMARK=ON` and run `./_install/bin/benchmark`
  - Should complete in ~35 seconds with solver runtime results.
- Test Python bindings:
  - Ensure module imports correctly: `PYTHONPATH=. python3 -c "import pyposelib; print('Import successful')"`
  - Test basic functionality: `PYTHONPATH=. python3 -c "import pyposelib as poselib; import numpy as np; camera = {'model': 'SIMPLE_PINHOLE', 'width': 1200, 'height': 800, 'params': [600, 600, 400]}; p2d = np.random.rand(10, 2) * 400; p3d = np.random.rand(10, 3) * 10; pose, info = poselib.estimate_absolute_pose(p2d, p3d, camera, {'max_reproj_error': 16.0}, {}); print('estimate_absolute_pose works')"`

### CI Validation
- The CI pipeline (.github/workflows/main.yml) requires:
  - Code must pass clang-format check
  - C++ library must build successfully
  - Benchmark must execute without errors
- Always run formatting check before committing: `git ls-files | grep -E '\.(cc|h)$' | xargs clang-format -i --verbose`

## Common Issues and Solutions

### Python Package Build Issues
- If `pip install .` fails with network timeouts, use the manual CMake build approach described above.
- If stub generation fails during build, add `-DGENERATE_STUBS=OFF` to cmake configuration.
- Ensure submodules are initialized before building Python bindings: `git submodule update --init --recursive`

### Build Performance
- NEVER CANCEL long-running builds - they are expected to take 2+ minutes.
- Use parallel builds with `-j $(nproc)` to utilize all CPU cores.
- Clean builds by removing `_build*`, `_install*`, and `build` directories if needed.

## Repository Structure

### Key Directories
- `PoseLib/` - Core C++ library source code
- `pybind/` - Python bindings using pybind11
- `pyposelib/` - Python package structure and utilities
- `benchmark/` - C++ benchmark application
- `tests/` - Python test files
- `scripts/` - Build scripts for different platforms
- `.github/workflows/` - CI/CD pipeline configuration

### Build Artifacts (excluded from git)
- `_build*/`, `build*/` - CMake build directories
- `_install*/`, `install*/` - Installation directories  
- `*.so`, `*.a`, `*.o` - Compiled libraries and objects
- `__pycache__/`, `*.egg-info` - Python build artifacts

### Important Files
- `CMakeLists.txt` - Main C++ build configuration
- `setup.py` - Python package build script
- `pyproject.toml` - Python package metadata
- `.clang-format` - Code formatting configuration
- `.gitmodules` - Git submodule configuration (pybind11)

## Build Timing Reference
- CMake configuration: ~2 seconds
- C++ library build: ~125 seconds (NEVER CANCEL, set 180+ second timeout)
- Python package build: ~140 seconds (NEVER CANCEL, set 200+ second timeout)
- Benchmark execution: ~35 seconds (NEVER CANCEL, set 60+ second timeout)
- Build script execution: ~120 seconds (NEVER CANCEL, set 180+ second timeout)
- Code formatting check: ~1 second

## Typical Development Workflow
1. Make code changes to C++ or Python files
2. Run formatting: `git ls-files | grep -E '\.(cc|h)$' | xargs clang-format -i --verbose`
3. Build and test C++ changes: `mkdir -p _build && cd _build && cmake -DCMAKE_BUILD_TYPE=Release -DWITH_BENCHMARK=ON .. && cmake --build . --target install -j $(nproc) && ./_install/bin/benchmark`
4. Build and test Python changes: `rm -rf build && mkdir build && cd build && cmake -DPYTHON_PACKAGE=ON -DGENERATE_STUBS=OFF .. && cmake --build . --target pyposelib -j $(nproc) && cp pybind/_core.cpython-*-linux-gnu.so ../pyposelib/ && cd .. && PYTHONPATH=. python3 -c "import pyposelib; print('Test passed')"`
5. Verify no formatting issues: `git diff --exit-code`
6. Commit changes