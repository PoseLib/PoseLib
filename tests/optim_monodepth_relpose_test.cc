#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/monodepth_relpose.h>

using namespace poselib;

namespace test::monodepth_relpose {

CameraPose random_camera() {
    Eigen::Vector3d cc;
    cc.setRandom();
    cc.normalize();
    cc *= 2.0;

    Eigen::Vector3d p;
    p.setRandom();
    p *= 0.1;

    Eigen::Vector3d r3 = p - cc;
    r3.normalize();

    Eigen::Vector3d r2 = r3.cross(Eigen::Vector3d::UnitX());
    r2.normalize();
    Eigen::Vector3d r1 = r2.cross(r3);

    Eigen::Matrix3d R;
    R.row(0) = r1;
    R.row(1) = r2;
    R.row(2) = r3;

    return CameraPose(R, -R * cc);
}

// Setup a scene with monodepth data (depths computed from geometry).
// Returns relative pose and corresponding 2D points + depths in both cameras.
// Points are in normalized coordinates for the calibrated case.
void setup_monodepth_scene(int N, CameraPose &pose, std::vector<Point2D> &x1, std::vector<Point2D> &x2,
                           std::vector<double> &d1, std::vector<double> &d2) {
    CameraPose p1 = random_camera();
    CameraPose p2 = random_camera();

    for (int i = 0; i < N;) {
        Eigen::Vector3d Xi;
        Xi.setRandom();

        Eigen::Vector3d Z1 = p1.apply(Xi);
        Eigen::Vector3d Z2 = p2.apply(Xi);

        // Only keep points with positive depth in both cameras
        if (Z1(2) <= 0.1 || Z2(2) <= 0.1)
            continue;

        x1.push_back(Eigen::Vector2d(Z1(0) / Z1(2), Z1(1) / Z1(2)));
        x2.push_back(Eigen::Vector2d(Z2(0) / Z2(2), Z2(1) / Z2(2)));
        d1.push_back(Z1(2));
        d2.push_back(Z2(2));
        ++i;
    }

    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - p2.R() * p1.R().transpose() * p1.t;
    pose = CameraPose(R, t);
}

// Same but with focal-length pixel coordinates.
void setup_monodepth_focal_scene(int N, CameraPose &pose, std::vector<Point2D> &x1, std::vector<Point2D> &x2,
                                 std::vector<double> &d1, std::vector<double> &d2, double f1, double f2) {
    CameraPose p1 = random_camera();
    CameraPose p2 = random_camera();

    for (int i = 0; i < N;) {
        Eigen::Vector3d Xi;
        Xi.setRandom();

        Eigen::Vector3d Z1 = p1.apply(Xi);
        Eigen::Vector3d Z2 = p2.apply(Xi);

        if (Z1(2) <= 0.1 || Z2(2) <= 0.1)
            continue;

        // Pixel coordinates (SIMPLE_PINHOLE with principal point at 0)
        x1.push_back(Eigen::Vector2d(f1 * Z1(0) / Z1(2), f1 * Z1(1) / Z1(2)));
        x2.push_back(Eigen::Vector2d(f2 * Z2(0) / Z2(2), f2 * Z2(1) / Z2(2)));
        d1.push_back(Z1(2));
        d2.push_back(Z2(2));
        ++i;
    }

    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - p2.R() * p1.R().transpose() * p1.t;
    pose = CameraPose(R, t);
}

bool test_monodepth_relpose_jacobian() {
    const size_t N = 10;
    CameraPose pose;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    setup_monodepth_scene(N, pose, x1, x2, d1, d2);

    MonoDepthTwoViewGeometry geometry(pose, 1.0);

    MonoDepthRelPoseRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2, d1, d2, 1.0, 1.0, false);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), MonoDepthTwoViewGeometry>(refiner, geometry, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, geometry);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, geometry);
    double r2 = 0.0;
    for (size_t i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_monodepth_relpose_shift_jacobian() {
    const size_t N = 10;
    CameraPose pose;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    setup_monodepth_scene(N, pose, x1, x2, d1, d2);

    // Use non-zero shifts: adjust stored depths so (d + shift) = actual_depth
    double shift1 = 0.5, shift2 = -0.3;
    for (size_t i = 0; i < N; ++i) {
        d1[i] -= shift1;
        d2[i] -= shift2;
    }

    MonoDepthTwoViewGeometry geometry(pose, 1.0, shift1, shift2);

    MonoDepthRelPoseRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2, d1, d2, 1.0, 1.0, true);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), MonoDepthTwoViewGeometry>(refiner, geometry, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, geometry);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, geometry);
    double r2 = 0.0;
    for (size_t i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_monodepth_shared_focal_relpose_jacobian() {
    const size_t N = 10;
    double f = 1.2;
    CameraPose pose;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    setup_monodepth_focal_scene(N, pose, x1, x2, d1, d2, f, f);

    Camera camera("SIMPLE_PINHOLE", {f, 0, 0}, -1, -1);
    MonoDepthTwoViewGeometry geometry(pose, 1.0);
    MonoDepthImagePair image_pair(geometry, camera, camera);

    MonoDepthSharedFocalRelPoseRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2, d1, d2, 1.0, 1.0);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), MonoDepthImagePair>(refiner, image_pair, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, image_pair);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, image_pair);
    double r2 = 0.0;
    for (size_t i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_monodepth_varying_focal_relpose_jacobian() {
    const size_t N = 10;
    double f1 = 1.2, f2 = 0.9;
    CameraPose pose;
    std::vector<Point2D> x1, x2;
    std::vector<double> d1, d2;
    setup_monodepth_focal_scene(N, pose, x1, x2, d1, d2, f1, f2);

    Camera camera1("SIMPLE_PINHOLE", {f1, 0, 0}, -1, -1);
    Camera camera2("SIMPLE_PINHOLE", {f2, 0, 0}, -1, -1);
    MonoDepthTwoViewGeometry geometry(pose, 1.0);
    MonoDepthImagePair image_pair(geometry, camera1, camera2);

    MonoDepthVaryingFocalRelPoseRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2, d1, d2, 1.0, 1.0);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), MonoDepthImagePair>(refiner, image_pair, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, image_pair);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, image_pair);
    double r2 = 0.0;
    for (size_t i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

} // namespace test::monodepth_relpose

using namespace test::monodepth_relpose;
std::vector<Test> register_optim_monodepth_relpose_test() {
    return {TEST(test_monodepth_relpose_jacobian),
            TEST(test_monodepth_relpose_shift_jacobian),
            TEST(test_monodepth_shared_focal_relpose_jacobian),
            TEST(test_monodepth_varying_focal_relpose_jacobian)};
}
