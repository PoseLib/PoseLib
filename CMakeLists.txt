cmake_minimum_required(VERSION 3.10)

project(PoseLib VERSION 2.1.0)

# Set variables
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

option(ENABLE_ASAN "Builds with address sanitizer" OFF)
if(ENABLE_ASAN)
	add_compile_options(-fno-omit-frame-pointer -fsanitize=address)
	add_link_options(-fno-omit-frame-pointer -fsanitize=address)
endif()

option(MARCH_NATIVE "Builds with -march=native" OFF)
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

# Eigen
find_package(Eigen3 REQUIRED)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})
# Compilation options
set_compile_flags(${LIBRARY_NAME})


# Sturm sequence solver max recursion depth
# You should lower this value only if you need to run PoseLib in environments with low stack limit (<1MB).
# Changing this may affect solver stability.
set(MAX_STURM_RECURSION_DEPTH_LIMIT 300 CACHE STRING
	"Maximum recursion depth for Sturm sequence solver of univariate polynomials.")

target_compile_definitions(PoseLib PUBLIC
	MAX_STURM_RECURSION_DEPTH_LIMIT=${MAX_STURM_RECURSION_DEPTH_LIMIT})

# Benchmark
option(WITH_BENCHMARK "Build benchmark example." OFF)
if(WITH_BENCHMARK)
	add_subdirectory(benchmark)
endif()


# Unit tests
option(BUILD_TESTS "Build unit tests." OFF)
if(BUILD_TESTS)
	add_subdirectory(tests)
endif()


# python bindings
option(PYTHON_PACKAGE "Build python package." OFF)
if(PYTHON_PACKAGE)
	add_subdirectory(pybind)
endif()