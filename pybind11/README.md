## Compilation

Add `-DPYTHON_PACKAGE=ON` option to the compilation:

    cmake -S . -B _build/ -DPYTHON_PACKAGE=ON -DCMAKE_INSTALL_PREFIX=_install
    cmake --build _build/ --target install -j 8
    cmake --build _build/ --target pip-package
    cmake --build _build/ --target install-pip-package

## Usage

Import with `import libpose`

Usage: `libpose.p3p(...)`

Arguments are lists of numpy vectors of size (3,), following the c++ function
signatures. The output pose is returned by the function and does not need to be
passed as argument.
