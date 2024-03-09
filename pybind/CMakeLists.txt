# setup PYTHON_EXECUTABLE

find_package (Python COMPONENTS Interpreter Development)
message(STATUS "Python_EXECUTABLE: " ${Python_EXECUTABLE})

add_subdirectory(pybind11)

pybind11_add_module(pyposelib MODULE pyposelib.cc)
target_link_libraries(pyposelib PUBLIC PoseLib::PoseLib Eigen3::Eigen)
target_compile_features(pyposelib PRIVATE cxx_std_17)
set_target_properties(pyposelib PROPERTIES OUTPUT_NAME poselib)
if(MSVC)
	target_compile_options(pyposelib PRIVATE /bigobj /fp:fast)
else()
	target_compile_options(pyposelib PRIVATE
		-Wall -Werror -fPIC -Wno-ignored-optimization-argument)
endif()