#include <string>
#include <vector>

/*
const std::string example_camera111 = "0 SIMPLE_RADIAL 1936 1296 2425.85 932.383 628.265 -0.0397695";
const std::string example_camera2 = "1 PINHOLE 6214 4138 3425.62 3426.29 3118.41 2069.07";
const std::string example_camera3 = "2 SIMPLE_PINHOLE 6214 4138 3425.62 3118.41 2069.07";
const std::string example_camera4 = "3 RADIAL 1936 1296 2425.85 932.38 629.325 -0.04012 0.00123";
const std::string example_camera5 = "4 OPENCV_FISHEYE 4288 2848 1921.45 1922.76 2156.14 1446.19 -0.0447036 -0.00511989
-0.00034067 -6.773700000000001e-05"; const std::string example_camera6 = "5 OPENCV 3200 2400 2575.94 2608.29 1599.26
1257.13 0.141865 -0.465301 0 0"; // 1.23e-05 -5.55e-06 const std::string example_camera7 = "6 OPENCV 1024 768 868.993378
866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571";
//const std::string example_camera8 = "7 FULL_OPENCV 2048 1536 975.0780885651661 975.0780885651661 948.9138024627247
713.4183625458141 -0.3108865430058566 0.1015445969542265 0.000123 -0.00321 -0.01305628265334396 0.0012 -0.000034
0.000056"; const std::string example_camera8 = "7 FULL_OPENCV 2048 1536 975.0780885651661 975.0780885651661
948.9138024627247 713.4183625458141 -0.3108865430058566 0.1015445969542265 0.0012 0.000013 -0.01305628265334396 0.0014
0.000012 0.0000013"; const std::string example_camera9 = "8 FOV 1024 1024 384.1 385.2 512 512 -0.012"; const std::string
example_camera10 = "9 FOV 1024 1024 384.1 385.2 512 512 0.0";
*/

// const std::string example_camera6 = "5 OPENCV_FISHEYE 4288 2848 1921.45 1922.76 2156.14 1446.19 0.0 0.0 0.0 0.0";
// const std::string example_camera6 = "5 OPENCV_FISHEYE 4288 2848 1921.0 1921.0 2156.0 1446.0 0.0 0.0 0.0 0.0";
// const std::string example_camera6 = "5 OPENCV_FISHEYE 4288 2848 1910.824198099596 1909.238000041893
// 2156.1360104186715 1446.19059357357971446.0 -0.03634870146026381 -0.007469599586535449
// -0.0006216688655440421 3.847917469074874e-05";

const std::vector<std::string> example_cameras = {
    "0 SIMPLE_RADIAL 1936 1296 2425.85 932.383 628.265 -0.0397695",
    "1 PINHOLE 6214 4138 3425.62 3426.29 3118.41 2069.07",
    "2 SIMPLE_PINHOLE 6214 4138 3425.62 3118.41 2069.07",
    "3 RADIAL 1936 1296 2425.85 932.38 629.325 -0.04012 0.00123",
    "4 OPENCV_FISHEYE 4288 2848 1921.45 1922.76 2156.14 1446.19 -0.0447036 -0.00511989 -0.00034067 "
    "-6.773700000000001e-05",
    "5 OPENCV 3200 2400 2575.94 2608.29 1599.26 1257.13 0.141865 -0.465301 0 0",
    "6 OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571",
    "7 FULL_OPENCV 2048 1536 975.0780885651661 975.0780885651661 948.9138024627247 713.4183625458141 "
    "-0.3108865430058566 0.1015445969542265 0.0012 0.000013 -0.01305628265334396 0.0014 0.000012 0.0000013",
    "8 FOV 1024 1024 384.1 385.2 512 512 -0.012",
    "9 FOV 1024 1024 384.1 385.2 512 512 0.0",
    "10 SIMPLE_RADIAL_FISHEYE  4288 2848 1921.45 2156.14 1446.19 -0.0447036",
    "11 RADIAL_FISHEYE  4288 2848 1921.45 2156.14 1446.19 -0.0447036 -0.00511989",
    "12 THIN_PRISM_FISHEYE 6048 4032 3408.35 3408.8 3033.92 2019.32 0.21167 0.20864 0.00053 -0.00015 -0.16568 0.4075 "
    "0.00048 0.00028",
    "13 THIN_PRISM_FISHEYE 752 480 713.177 713.461 357.084 245.409 0.507481 0.391235 -0.0020782 0.00449456 -5.4533 "
    "11.1658 -0.00513268 0.00208362",
    "14 THIN_PRISM_FISHEYE 752 480 721.094 721.073 401.115 259.583 0.53325 -0.137724 -0.00199129 -0.00380773 -2.43356 "
    "5.10319 0.00535658 0.00141871",
    "15 THIN_PRISM_FISHEYE 752 480 540.715 541.069 369.138 235.317 -0.0829528 -0.00550879 0.000636439 0.00273958 "
    "-0.0174673 0.0109266 -0.0132617 -0.00932591",
    "16 THIN_PRISM_FISHEYE 752 480 545.334 545.679 389.666 223.311 -0.0811108 -0.00816016 0.000636755 0.0020227 "
    "-0.0137968 0.00782607 -0.00357899 -0.0043318",
    "17 DIVISION 2560 1152 1265.772 1262.546 1257.333 571.722 0.005963",
    "18 DIVISION 4032 3024 3032.384 3032.384 2026.282 508.856 -0.75688",
    "19 SIMPLE_DIVISION 2560 1152 1265.772 1257.333 571.722 0.005963",
    "20 SIMPLE_DIVISION 4032 3024 3032.384 2026.282 508.856 -0.75688"};

// Cameras that are radially symmetric
const std::vector<std::string> radially_symmetric_example_cameras = {example_cameras[0], example_cameras[2],
                                                                     example_cameras[10]};