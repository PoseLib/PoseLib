configure_file(
    "setup.py.in"
    "setup.py"
)

set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/pybind11")

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
    COMMAND ${CMAKE_COMMAND}
        -DPYTHON_PACKAGE=ON
        ${CMAKE_BINARY_DIR}
    COMMAND
        ${Python_EXECUTABLE} ${PYTHON_PACKAGE_DST_DIR}/setup.py bdist_wheel --dist-dir pip_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
)

# Use `make install-pip-package` to install pip wheel package to the current
# python environment.
add_custom_target(install-pip-package
    COMMAND ${CMAKE_COMMAND}
            -DPYTHON_PACKAGE=ON
            -DPYTHON_PACKAGE_DST_DIR=${PYTHON_PACKAGE_DST_DIR}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/make_install_pip_package.cmake
    DEPENDS pip-package
)
