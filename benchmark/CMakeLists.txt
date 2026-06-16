add_executable(solver_benchmark
    solver_benchmark.cc
    problem_generator.cc
)

target_link_libraries(solver_benchmark PRIVATE PoseLib::PoseLib Eigen3::Eigen)

install(
  TARGETS solver_benchmark
  RUNTIME DESTINATION bin
  COMPONENT "${INSTALL_BIN_DIR}"
)

# Compilation options
set_compile_flags(solver_benchmark)
