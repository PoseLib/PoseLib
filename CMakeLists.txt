cmake_minimum_required (VERSION 3.1)
project(PoseLib)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()


add_compile_options(-Wall -Werror -Wno-sign-compare -Wno-unused-variable)

find_package(Eigen3 REQUIRED NO_MODULE)

add_library(poselib STATIC PoseLib/univariate.cc PoseLib/p3p.cc PoseLib/p2p2l.cc PoseLib/up2p.cc PoseLib/up1p2l.cc PoseLib/up4l.cc PoseLib/re3q3.cc PoseLib/gp3p.cc)
target_link_libraries(poselib Eigen3::Eigen)


add_executable(benchmark benchmark/benchmark.cc benchmark/problem_generator.cc)
target_link_libraries(benchmark Eigen3::Eigen poselib)
target_include_directories(benchmark PUBLIC ".")