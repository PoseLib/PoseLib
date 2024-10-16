add_executable(run_tests
    run_tests.cc
    camera_models_test.cc
    optim_absolute_test.cc
    optim_gen_absolute_test.cc
    optim_relative_test.cc
    optim_fundamental_test.cc
    optim_gen_relative_test.cc
    optim_homography_test.cc
    recalibrator_test.cc
)
target_link_libraries(run_tests PRIVATE PoseLib::PoseLib Eigen3::Eigen)
set_compile_flags(run_tests)