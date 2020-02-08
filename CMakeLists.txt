cmake_minimum_required (VERSION 3.1)
project(PoseLib)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(poselib STATIC PoseLib/univariate.cc PoseLib/p3p.cc PoseLib/p2p2l.cc PoseLib/up2p.cc PoseLib/up1p2l.cc PoseLib/up4l.cc PoseLib/re3q3.cc PoseLib/gp3p.cc PoseLib/gp4ps.cc PoseLib/p4pf.cc)
target_link_libraries(poselib Eigen3::Eigen)

add_executable(benchmark benchmark/benchmark.cc benchmark/problem_generator.cc)
target_link_libraries(benchmark Eigen3::Eigen poselib)
target_include_directories(benchmark PUBLIC ".")

if(MSVC)	
	target_compile_options(poselib PRIVATE /bigobj /fp:fast)
	target_compile_options(benchmark PRIVATE /bigobj /fp:fast)
else()
	add_compile_options(-march=native)
	target_compile_options(poselib PRIVATE -Wall -Werror -Wno-sign-compare -Wno-unused-variable -ffast-math)
	target_compile_options(benchmark PRIVATE -Wall -Werror -Wno-sign-compare -Wno-unused-variable -ffast-math)
endif()