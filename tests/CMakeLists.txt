add_executable(run_tests
    run_tests.cc
    camera_models_test.cc
)
target_link_libraries(run_tests PRIVATE PoseLib::PoseLib Eigen3::Eigen)
set_compile_flags(run_tests)