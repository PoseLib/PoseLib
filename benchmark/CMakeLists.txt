include_directories(${CMAKE_SOURCE_DIR}/re3q3)

add_executable(benchmark
    benchmark.cc
    problem_generator.cc
    ${CMAKE_SOURCE_DIR}/re3q3/re3q3/re3q3.cc
    ${CMAKE_SOURCE_DIR}/re3q3/re3q3/sturm8.cc
)

target_link_libraries(benchmark ${LIBRARY_NAME} Eigen3::Eigen)

install(
  TARGETS benchmark
  # EXPORT ${PROJECT_EXPORT} # uncomment in order to export target
  RUNTIME DESTINATION bin
  COMPONENT "${INSTALL_BIN_DIR}"
)
