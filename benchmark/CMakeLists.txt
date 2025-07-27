add_executable(benchmark
    benchmark.cc
    problem_generator.cc
)

target_link_libraries(benchmark PRIVATE PoseLib::PoseLib Eigen3::Eigen)

set_target_properties(benchmark PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

install(
  TARGETS benchmark
  RUNTIME DESTINATION bin
  COMPONENT "${INSTALL_BIN_DIR}"
)

# Compilation options
if(MSVC)
  target_compile_options(benchmark PRIVATE /bigobj /fp:fast)
else()
  if (MARCH_NATIVE)
    target_compile_options(benchmark PRIVATE
      -march=native -Wall -Werror -fPIC -Wno-sign-compare
      -Wno-unused-variable -ffast-math -Wno-ignored-optimization-argument)
  else()
    target_compile_options(benchmark PRIVATE
      -Wall -Werror -fPIC -Wno-sign-compare
      -Wno-unused-variable -ffast-math)
  endif()
  if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(benchmark PRIVATE
      -Wno-maybe-uninitialized)
  endif()
endif()
