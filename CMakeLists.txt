cmake_minimum_required(VERSION 3.10)

project(PoseLib VERSION 2.0.0)

# Set variables
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

function(set_compile_flags build_target)
if(MSVC)
	target_compile_options(${build_target} PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(${build_target} PRIVATE
		-march=native -Wall -Werror -fPIC -Wno-sign-compare)
endif()
target_compile_features(${build_target} PUBLIC cxx_std_17)
endfunction()


# Eigen
find_package(Eigen3)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})
# Compilation options
set_compile_flags(${LIBRARY_NAME})


# Benchmark
option(WITH_BENCHMARK "Build benchmark example." OFF)
if(WITH_BENCHMARK)
	add_subdirectory(benchmark)
endif()

# python bindings
option(PYTHON_PACKAGE "Build python package." ON)
if(PYTHON_PACKAGE)
	add_subdirectory(pybind)
endif()

option(BUILD_TESTS "Build tests." ON)
if(BUILD_TESTS)
	add_subdirectory(tests)
endif()
