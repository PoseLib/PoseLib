cmake_minimum_required(VERSION 3.13)

# Hunter (package manager)
include("cmake/Hunter.cmake")

# Project name and version
project(PoseLib VERSION 1.0.0)

# Set variables
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)

# Eigen
hunter_add_package(Eigen)
find_package(Eigen3 CONFIG REQUIRED)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})

# Benchmark
add_subdirectory(benchmark)

# Compilation options
if(MSVC)
	target_compile_options(${LIBRARY_NAME} PRIVATE /bigobj /fp:fast)
	target_compile_options(benchmark PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(${LIBRARY_NAME} PRIVATE
		-march=native -Wall -Werror -Wno-sign-compare
	 	-Wno-unused-variable -ffast-math)
	target_compile_options(benchmark PRIVATE
		-march=native -Wall -Werror -Wno-sign-compare
		-Wno-unused-variable -ffast-math)
endif()
