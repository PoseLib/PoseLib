cmake_minimum_required(VERSION 3.10)

project(PoseLib VERSION 2.0.3)

# Set variables
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

# Eigen
find_package(Eigen3)

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
	add_subdirectory(pybind)
endif()

# Compilation options
if(MSVC)
	target_compile_options(${LIBRARY_NAME} PRIVATE /bigobj /fp:fast)
else()
	# If you change this, make sure to update the corresponding line in the pybind CMakeLists
	target_compile_options(${LIBRARY_NAME} PRIVATE
		-march=native -Wall -Werror -fPIC -Wno-ignored-optimization-argument)
	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		target_compile_options(${LIBRARY_NAME} PRIVATE
				-Wno-maybe-uninitialized)
	endif()
endif()
