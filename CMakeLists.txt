cmake_minimum_required(VERSION 3.0)

project(PoseLib)

# Set variables
include(${PROJECT_SOURCE_DIR}/cmake/SetEnv.cmake)
set(LIBRARY_NAME   ${PROJECT_NAME})
set(LIBRARY_FOLDER ${PROJECT_NAME})

# Eigen
find_package(Eigen3 REQUIRED)

# Library sources
add_subdirectory(${LIBRARY_FOLDER})

# Benchmark
add_subdirectory(benchmark)

# Install targets
include(${PROJECT_SOURCE_DIR}/cmake/InstallConfig.cmake)

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
