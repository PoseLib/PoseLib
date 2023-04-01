
const std::string example_camera1 = "0 SIMPLE_RADIAL 1936 1296 2425.85 932.383 628.265 -0.0397695";
const std::string example_camera2 = "1 PINHOLE 6214 4138 3425.62 3426.29 3118.41 2069.07";
const std::string example_camera3 = "2 SIMPLE_PINHOLE 6214 4138 3425.62 3118.41 2069.07";
const std::string example_camera4 = "3 RADIAL 1936 1296 2425.85 932.38 629.325 -0.04012 0.00123";
const std::string example_camera5 = "4 OPENCV_FISHEYE 4288 2848 1921.45 1922.76 2156.14 1446.19 -0.0447036 -0.00511989 -0.00034067 -6.773700000000001e-05";
const std::string example_camera6 = "5 OPENCV 3200 2400 2575.94 2608.29 1599.26 1257.13 0.141865 -0.465301 0 0"; // 1.23e-05 -5.55e-06
const std::string example_camera7 = "6 OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571";


//const std::string example_camera6 = "5 OPENCV_FISHEYE 4288 2848 1921.45 1922.76 2156.14 1446.19 0.0 0.0 0.0 0.0";
//const std::string example_camera6 = "5 OPENCV_FISHEYE 4288 2848 1921.0 1921.0 2156.0 1446.0 0.0 0.0 0.0 0.0";
//const std::string example_camera6 = "5 OPENCV_FISHEYE 4288 2848 1910.824198099596 1909.238000041893 2156.1360104186715 1446.19059357357971446.0 -0.03634870146026381 -0.007469599586535449 -0.0006216688655440421 3.847917469074874e-05";

const std::vector<std::string> example_cameras = {
    example_camera1, example_camera2, example_camera3,
    example_camera4, example_camera5};