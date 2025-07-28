# setup PYTHON_EXECUTABLE

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

option(GENERATE_STUBS "Whether to generate stubs" ON)

add_subdirectory(pybind11)

pybind11_add_module(pyposelib MODULE pyposelib.cc)
target_link_libraries(pyposelib PUBLIC PoseLib::PoseLib Eigen3::Eigen)
target_compile_features(pyposelib PRIVATE cxx_std_17)
set_target_properties(pyposelib PROPERTIES OUTPUT_NAME _core)

# Install the compiled module
install(TARGETS pyposelib LIBRARY DESTINATION . COMPONENT python)

if(MSVC)
	target_compile_options(pyposelib PRIVATE /bigobj /fp:fast)
else()
    if (MARCH_NATIVE)
        target_compile_options(pyposelib PRIVATE -march=native -Wall -Werror -fPIC -Wno-ignored-optimization-argument)
    else()
        target_compile_options(pyposelib PRIVATE -Wall -Werror -fPIC)
    endif()
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_compile_options(pyposelib PRIVATE
            -Wno-maybe-uninitialized)
    endif()
endif()

if(GENERATE_STUBS AND UNIX)
    message(STATUS "Enabling stubs generation")
    set(STUBGEN_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/_core")
    add_custom_command(
        TARGET pyposelib POST_BUILD
        COMMAND
          "${CMAKE_COMMAND}" -E env
          "PYTHONPATH=$<TARGET_FILE_DIR:pyposelib>:$ENV{PYTHONPATH}"
          bash ${PROJECT_SOURCE_DIR}/pybind/generate_stubs.sh "${Python_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}"
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
