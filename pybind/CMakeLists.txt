# setup PYTHON_EXECUTABLE
if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()
find_package(Python REQUIRED COMPONENTS Interpreter ${DEV_MODULE})
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

option(GENERATE_STUBS "Whether to generate stubs" ON)

# Find pybind11 from pip installation
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
find_package(pybind11 3.0.1 REQUIRED)

pybind11_add_module(pyposelib MODULE
    pyposelib.cc
    bindings/types.cc
    bindings/solvers.cc
    bindings/estimators/absolute_pose.cc
    bindings/estimators/relative_pose.cc
    bindings/estimators/homography.cc
    bindings/estimators/hybrid_pose.cc
)
target_link_libraries(pyposelib PUBLIC PoseLib::PoseLib Eigen3::Eigen)
set_target_properties(pyposelib PROPERTIES OUTPUT_NAME _core)
set_compile_flags(pyposelib)

# Install the compiled module
install(TARGETS pyposelib LIBRARY DESTINATION . COMPONENT python)

if(GENERATE_STUBS)
    message(STATUS "Enabling stubs generation")
    set(STUBGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_core")
    add_custom_command(
        TARGET pyposelib POST_BUILD
        COMMAND
          "${CMAKE_COMMAND}" -E env
          "PYTHONPATH=$<TARGET_FILE_DIR:pyposelib>$<SEMICOLON>$ENV{PYTHONPATH}"
          "${Python_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/pybind/generate_stubs.py" "${Python_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating pybind11 stubs"
        VERBATIM
    )

    # Install the generated stub file (handle both single file and directory cases)
    # Most pybind11_stubgen versions create a single .pyi file
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_core.pyi"
            DESTINATION .
            COMPONENT python
            OPTIONAL)  # OPTIONAL in case it doesn't exist

    # Some versions might create a directory instead
    install(DIRECTORY "${STUBGEN_OUTPUT_DIR}/"
            DESTINATION .
            COMPONENT python
            OPTIONAL)  # OPTIONAL in case it doesn't exist
endif()
