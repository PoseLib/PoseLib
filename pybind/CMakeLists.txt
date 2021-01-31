# setup PYTHON_EXECUTABLE
find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

# generate `setup.py` file
configure_file(
    "setup.py.in"
    "setup.py"
)

# PYTHON_PACKAGE_DST_DIR
set(PYTHON_PACKAGE_DST_DIR "${CMAKE_BINARY_DIR}/pybind")

# Use `make pip-package` to create the pip package in the build directory
add_custom_target(pip-package
    COMMAND ${CMAKE_COMMAND}
        -DPYTHON_PACKAGE=ON
        -DBUILD_SHARED_LIBS=OFF  # force static library
        ${CMAKE_BINARY_DIR}
    COMMAND
        ${Python_EXECUTABLE} ${PYTHON_PACKAGE_DST_DIR}/setup.py bdist_wheel --dist-dir pip_package
    COMMAND echo "pip wheel created at ${PYTHON_PACKAGE_DST_DIR}/pip_package"
    WORKING_DIRECTORY ${PYTHON_PACKAGE_DST_DIR}
    DEPENDS ${LIBRARY_NAME}
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
