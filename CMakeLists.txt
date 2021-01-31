cmake_minimum_required(VERSION 3.13)

project(PoseLib VERSION 1.0.0)

# Set variables
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

# Eigen
find_package(Eigen3 REQUIRED)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})

# Benchmark
option(WITH_BENCHMARK "Build benchmark example." OFF)
if(WITH_BENCHMARK)
	add_subdirectory(benchmark)
endif()

# python bindings
option(PYTHON_PACKAGE "Build python package." OFF)
if(PYTHON_PACKAGE)
	add_subdirectory(pybind11)
endif()

# Compilation options
if(MSVC)
	target_compile_options(${LIBRARY_NAME} PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(${LIBRARY_NAME} PRIVATE
		-march=native -Wall -Werror -Wno-sign-compare
	 	-Wno-unused-variable -ffast-math -fPIC)
endif()
