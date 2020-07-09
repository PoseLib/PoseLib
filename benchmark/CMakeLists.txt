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
  target_compile_options(benchmark PRIVATE
    -march=native -Wall -Werror -Wno-sign-compare
    -Wno-unused-variable -ffast-math)
endif()
