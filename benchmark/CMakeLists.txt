include_directories(${PROJECT_SOURCE_DIR}/re3q3)

add_executable(benchmark
    benchmark.cc
    problem_generator.cc
)

target_link_libraries(benchmark PRIVATE ${LIBRARY_NAME} Eigen3::Eigen)

set_target_properties(benchmark PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

install(
  TARGETS benchmark
  # EXPORT ${PROJECT_EXPORT} # uncomment in order to export target
  RUNTIME DESTINATION bin
  COMPONENT "${INSTALL_BIN_DIR}"
)
