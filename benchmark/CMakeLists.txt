add_executable(benchmark
    benchmark.cc
    problem_generator.cc
)

target_link_libraries(benchmark PRIVATE PoseLib::PoseLib Eigen3::Eigen)
set_compile_flags(benchmark)

install(
  TARGETS benchmark
  RUNTIME DESTINATION bin
  COMPONENT "${INSTALL_BIN_DIR}"
)
