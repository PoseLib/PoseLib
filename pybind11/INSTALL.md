Adjusts library and include paths in setup.py.
Build with `python setup.py build`.
Install with `python setup.py install`.
Import with `import libpose`
Usage: `libpose.p3p(...)`
Arguments are lists of numpy vectors of size (3,), following the c++ function signatures. The output pose is returned by the function and does not need to be passed as argument.
