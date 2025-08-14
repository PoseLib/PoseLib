cmake_minimum_required(VERSION 3.10)

project(PoseLib VERSION 2.0.5)

# Compilation flags function
function(set_compile_flags build_target)
if(MSVC)
	target_compile_options(${build_target} PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(${build_target} PRIVATE
		-O3 -Wall -Werror -fPIC -Wno-sign-compare -Wfatal-errors)
	if(MARCH_NATIVE)
		target_compile_options(${build_target} PRIVATE -march=native)
	endif()
	if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
		target_compile_options(${build_target} PRIVATE
				-Wno-maybe-uninitialized)
	endif()
endif()
target_compile_features(${build_target} PUBLIC cxx_std_17)
endfunction()

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


# Compilation options
option(MARCH_NATIVE "Enable CPU specific optimizations" OFF)
set_compile_flags(${LIBRARY_NAME})

# python bindings
option(PYTHON_PACKAGE "Build python package." OFF)
if(PYTHON_PACKAGE)
	add_subdirectory(pybind)
endif()
