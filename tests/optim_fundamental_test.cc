#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/misc/essential.h>
#include <PoseLib/robust/optim/fundamental.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <PoseLib/types.h>

using namespace poselib;

//////////////////////////////
// Fundamental matrix

namespace test::fundamental {

CameraPose random_camera(test_rng::Rng &rng) {
    Eigen::Vector3d cc = test_rng::symmetric_vec3(rng);
    cc.normalize();
    cc *= 2.0;

    // Lookat point
    Eigen::Vector3d p = test_rng::symmetric_vec3(rng, 0.1);

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

void setup_scene(int N, CameraPose &pose, Eigen::Matrix3d &F, std::vector<Point2D> &x1, std::vector<Point2D> &x2,
                 Camera &cam1, Camera &cam2, const std::string &case_name = "fundamental_scene",
                 size_t case_index = 0) {

    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    CameraPose p1 = random_camera(rng);
    CameraPose p2 = random_camera(rng);

    while (x1.size() < N) {
        Eigen::Vector3d Xi = test_rng::symmetric_vec3(rng);

        Eigen::Vector2d xi1;
        cam1.project(p1.apply(Xi), &xi1);
        if (!check_valid_camera_projection(cam1, xi1, p1.apply(Xi))) {
            continue;
        }

        Eigen::Vector2d xi2;
        cam2.project(p2.apply(Xi), &xi2);
        if (!check_valid_camera_projection(cam2, xi2, p2.apply(Xi))) {
            continue;
        }

        x1.push_back(xi1);
        x2.push_back(xi2);
    }

    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - p2.R() * p1.R().transpose() * p1.t;

    pose = CameraPose(R, t);

    Eigen::Matrix3d K1, K2;
    K1 << cam1.focal_x(), 0.0, cam1.principal_point()(0), 0.0, cam1.focal_y(), cam1.principal_point()(1), 0.0, 0.0, 1.0;
    K2 << cam2.focal_x(), 0.0, cam2.principal_point()(0), 0.0, cam2.focal_y(), cam2.principal_point()(1), 0.0, 0.0, 1.0;

    Eigen::Matrix3d E;
    essential_from_motion(pose, &E);
    F = K2.inverse().transpose() * E * K1.inverse();
    F = F / F.norm();
}

bool test_fundamental_pose_normal_acc() {

    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 2.0 2.0 0.5 0.5";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    Eigen::Matrix3d F;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, F, x1, x2, camera, camera);
    FactorizedFundamentalMatrix FF(F);

    NormalAccumulator acc;
    PinholeFundamentalRefiner refiner(x1, x2);
    acc.initialize(refiner.num_params);

    // Check that residual is zero
    acc.reset_residual();
    refiner.compute_residual(acc, FF);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    // Check the gradient is zero
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, FF);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}

bool test_fundamental_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 2.0 2.0 0.5 0.5";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    Eigen::Matrix3d F;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, F, x1, x2, camera, camera);
    FactorizedFundamentalMatrix FF(F);

    PinholeFundamentalRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), FactorizedFundamentalMatrix>(refiner, FF, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, FF);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, FF);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_fundamental_pose_refinement() {
    const size_t N = 100;
    std::string camera_str = "0 PINHOLE 1 1 2.0 2.0 1.0 1.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    Eigen::Matrix3d F;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, F, x1, x2, camera, camera);
    FactorizedFundamentalMatrix FF(F);

    // Add some noise
    test_rng::Rng noise_rng = test_rng::make_rng("fundamental_pose_refinement_noise");
    for (size_t i = 0; i < N; ++i) {
        x1[i] += test_rng::symmetric_vec2(noise_rng, 0.001);
        x2[i] += test_rng::symmetric_vec2(noise_rng, 0.001);
    }

    PinholeFundamentalRefiner refiner(x1, x2);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &FF, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_fundamental_pose_refinement");

    REQUIRE_SMALL(stats.grad_norm, 1e-6);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

bool test_rd_fundamental_pose_normal_acc() {
    const size_t N = 15;
    std::string camera_str1 = "0 DIVISION 1 1 1.0 1.0 0.0 0.0 -0.25";
    std::string camera_str2 = "0 DIVISION 1 1 1.0 1.0 0.0 0.0 -0.15";
    Camera camera1, camera2;
    camera1.initialize_from_txt(camera_str1);
    camera2.initialize_from_txt(camera_str2);

    CameraPose pose;
    Eigen::Matrix3d F;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, F, x1, x2, camera1, camera2);
    F.normalize();
    FactorizedProjectiveImagePair proj_image_pair(F, camera1, camera2);

    NormalAccumulator acc;
    RDFundamentalRefiner refiner(x1, x2);
    acc.initialize(refiner.num_params);

    // Check that residual is zero
    acc.reset_residual();
    refiner.compute_residual(acc, proj_image_pair);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    // Check the gradient is zero
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, proj_image_pair);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}

bool test_rd_fundamental_pose_jacobian() {
    const size_t N = 15;
    std::string camera_str1 = "0 DIVISION 1 1 1.0 1.0 0.0 0.0 -0.25";
    std::string camera_str2 = "0 DIVISION 1 1 1.0 1.0 0.0 0.0 -0.15";
    Camera camera1, camera2;
    camera1.initialize_from_txt(camera_str1);
    camera2.initialize_from_txt(camera_str2);

    CameraPose pose;
    Eigen::Matrix3d F;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, F, x1, x2, camera1, camera2);

    FactorizedProjectiveImagePair proj_image_pair(F, camera1, camera2);

    //    std::cout << "x1[0]: " << x1[0].transpose() << " x2[0]: " << x2[0].transpose() << std::endl;

    RDFundamentalRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2);

    const double delta = 1e-8;
    double jac_err = verify_jacobian<decltype(refiner), FactorizedProjectiveImagePair>(refiner, proj_image_pair, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, proj_image_pair);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, proj_image_pair);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_rd_fundamental_pose_refinement() {
    const size_t N = 100;
    std::string camera_str1 = "0 DIVISION 1 1 1.0 1.0 0.0 0.0 -0.25";
    std::string camera_str2 = "0 DIVISION 1 1 1.0 1.0 0.0 0.0 -0.15";
    Camera camera1, camera2;
    camera1.initialize_from_txt(camera_str1);
    camera2.initialize_from_txt(camera_str2);

    CameraPose pose;
    Eigen::Matrix3d F;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, F, x1, x2, camera1, camera2);
    FactorizedProjectiveImagePair proj_image_pair(F, camera1, camera2);

    // Add some noise
    test_rng::Rng noise_rng = test_rng::make_rng("rd_fundamental_pose_refinement_noise");
    for (size_t i = 0; i < N; ++i) {
        x1[i] += test_rng::symmetric_vec2(noise_rng, 0.001);
        x2[i] += test_rng::symmetric_vec2(noise_rng, 0.001);
    }

    RDFundamentalRefiner refiner(x1, x2);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &proj_image_pair, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_rd_fundamental_pose_refinement");

    REQUIRE_SMALL(stats.grad_norm, 1e-5);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

bool test_shared_rd_fundamental_pose_jacobian() {
    const size_t N = 15;
    std::string camera_str1 = "0 DIVISION 1 1 1.0 1.0 0.0 0.0 -0.25";
    Camera camera1, camera2;
    camera1.initialize_from_txt(camera_str1);
    camera2.initialize_from_txt(camera_str1);

    CameraPose pose;
    Eigen::Matrix3d F;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, F, x1, x2, camera1, camera2);

    FactorizedProjectiveImagePair proj_image_pair(F, camera1, camera2);

    //    std::cout << "x1[0]: " << x1[0].transpose() << " x2[0]: " << x2[0].transpose() << std::endl;

    SharedRDFundamentalRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2);

    const double delta = 1e-8;
    double jac_err = verify_jacobian<decltype(refiner), FactorizedProjectiveImagePair>(refiner, proj_image_pair, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, proj_image_pair);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, proj_image_pair);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_shared_rd_fundamental_pose_refinement() {
    const size_t N = 100;
    std::string camera_str1 = "0 DIVISION 1 1 1.0 1.0 0.0 0.0 -0.25";
    Camera camera1, camera2;
    camera1.initialize_from_txt(camera_str1);
    camera2.initialize_from_txt(camera_str1);

    CameraPose pose;
    Eigen::Matrix3d F;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, F, x1, x2, camera1, camera2);
    FactorizedProjectiveImagePair proj_image_pair(F, camera1, camera2);

    // Add some noise
    test_rng::Rng noise_rng = test_rng::make_rng("shared_rd_fundamental_pose_refinement_noise");
    for (size_t i = 0; i < N; ++i) {
        x1[i] += test_rng::symmetric_vec2(noise_rng, 0.001);
        x2[i] += test_rng::symmetric_vec2(noise_rng, 0.001);
    }

    SharedRDFundamentalRefiner refiner(x1, x2);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &proj_image_pair, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_shared_rd_fundamental_pose_refinement");

    REQUIRE_SMALL(stats.grad_norm, 1e-5); // TODO: Look into this threshold. Perhaps some scaling is wonky.
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

} // namespace test::fundamental

using namespace test::fundamental;
std::vector<Test> register_optim_fundamental_test() {
    return {TEST(test_fundamental_pose_normal_acc),         TEST(test_fundamental_pose_jacobian),
            TEST(test_fundamental_pose_refinement),         TEST(test_rd_fundamental_pose_normal_acc),
            TEST(test_rd_fundamental_pose_jacobian),        TEST(test_rd_fundamental_pose_refinement),
            TEST(test_shared_rd_fundamental_pose_jacobian), TEST(test_shared_rd_fundamental_pose_refinement)};
}
