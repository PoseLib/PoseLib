## Compilation

Add `-DPYTHON_PACKAGE=ON` option to the compilation:

    cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install
    cmake --build _build/ --target install -j 8
    cmake --build _build/ --target pip-package
    cmake --build _build/ --target install-pip-package

## Usage

Import with `import poselib`

Usage: `poselib.p3p(...)`

Arguments are typically lists of numpy vectors of size (3,), and follow the c++ function
signatures. The output pose is returned by the function and does not need to be
passed as argument.

See `help(poselib)` for a list of the available solvers and `help(poselib.p3p)` for more details on the arguments of a particular solver.
