#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/optim/relative.h>
#include <PoseLib/robust/robust_loss.h>

using namespace poselib;

//////////////////////////////
// Relative pose

namespace test::relative {

CameraPose random_camera() {
    Eigen::Vector3d cc;
    cc.setRandom();
    cc.normalize();
    cc *= 2.0;

    // Lookat point
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

void setup_scene(int N, CameraPose &pose, std::vector<Point2D> &x1, std::vector<Point2D> &x2, Camera &cam1,
                 Camera &cam2) {

    CameraPose p1 = random_camera();
    CameraPose p2 = random_camera();

    for (size_t i = 0; i < N; ++i) {
        Eigen::Vector3d Xi;
        Xi.setRandom();

        Eigen::Vector2d xi;
        cam1.project(p1.apply(Xi), &xi);
        x1.push_back(xi);

        cam2.project(p2.apply(Xi), &xi);
        x2.push_back(xi);
    }

    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - p2.R() * p1.R().transpose() * p1.t;

    pose = CameraPose(R, t);
}

bool test_relative_pose_normal_acc() {

    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    NormalAccumulator acc;
    PinholeRelativePoseRefiner refiner(x1, x2);
    acc.initialize(refiner.num_params);

    // Check that residual is zero
    acc.reset_residual();
    refiner.compute_residual(acc, pose);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    // Check the gradient is zero
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}

bool test_relative_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    PinholeRelativePoseRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), CameraPose>(refiner, pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, pose);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_relative_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    // Add some noise
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d n;
        n.setRandom();
        x1[i] += 0.001 * n;
        n.setRandom();
        x2[i] += 0.001 * n;
    }

    PinholeRelativePoseRefiner refiner(x1, x2);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);

    // std::cout << "iter = " << stats.iterations << "\n";
    // std::cout << "initial_cost = " << stats.initial_cost << "\n";
    // std::cout << "cost = " << stats.cost << "\n";
    // std::cout << "lambda = " << stats.lambda << "\n";
    // std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    // std::cout << "step_norm = " << stats.step_norm << "\n";
    // std::cout << "grad_norm = " << stats.grad_norm << "\n";

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

bool test_shared_focal_relative_pose_normal_acc() {
    const size_t N = 10;
    std::string camera_str = "0 SIMPLE_PINHOLE 1 1 1.2 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);
    ImagePair image_pair = ImagePair(pose, camera, camera);

    NormalAccumulator acc;
    SharedFocalRelativePoseRefiner refiner(x1, x2);
    acc.initialize(refiner.num_params);

    // Check that residual is zero
    acc.reset_residual();
    refiner.compute_residual(acc, image_pair);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    // Check the gradient is zero
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, image_pair);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}

bool test_shared_focal_relative_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 SIMPLE_PINHOLE 1 1 1.2 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);
    ImagePair image_pair = ImagePair(pose, camera, camera);

    SharedFocalRelativePoseRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), ImagePair>(refiner, image_pair, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, image_pair);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, image_pair);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}
//
bool test_shared_focal_relative_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 SIMPLE_PINHOLE 1 1 1.2 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);
    ImagePair image_pair = ImagePair(pose, camera, camera);

    // Add some noise
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d n;
        n.setRandom();
        x1[i] += 0.001 * n;
        n.setRandom();
        x2[i] += 0.001 * n;
    }

    SharedFocalRelativePoseRefiner refiner(x1, x2);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &image_pair, bundle_opt, print_iteration);

    // std::cout << "iter = " << stats.iterations << "\n";
    // std::cout << "initial_cost = " << stats.initial_cost << "\n";
    // std::cout << "cost = " << stats.cost << "\n";
    // std::cout << "lambda = " << stats.lambda << "\n";
    // std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    // std::cout << "step_norm = " << stats.step_norm << "\n";
    // std::cout << "grad_norm = " << stats.grad_norm << "\n";

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

bool test_tangent_sampson_fix_camera_relative_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "3 RADIAL 1936 1296 2425.85 932.38 629.325 -0.04012 0.00123";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    std::vector<Eigen::Vector3d> d1, d2;
    std::vector<Eigen::Matrix<double, 3, 2>> J1inv, J2inv;
    camera.unproject_with_jac(x1, &d1, &J1inv);
    camera.unproject_with_jac(x2, &d2, &J2inv);

    FixCameraRelativePoseRefiner<UniformWeightVector, TestAccumulator> refiner(d1, d2, J1inv, J2inv);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), CameraPose>(refiner, pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, pose);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_tangent_sampson_fix_camera_relative_pose_refinement() {
    const size_t N = 25;
    std::string camera_str = "3 RADIAL 1936 1296 2425.85 932.38 629.325 -0.04012 0.00123";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    // Add some noise
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d n;
        n.setRandom();
        x1[i] += 2.0 * n;
        n.setRandom();
        x2[i] += 2.0 * n;
    }

    double f = camera.focal();
    for (int i = 0; i < N; ++i) {
        x1[i] /= f;
        x2[i] /= f;
    }
    camera.rescale(1.0 / f);

    std::vector<Eigen::Vector3d> d1, d2;
    std::vector<Eigen::Matrix<double, 3, 2>> J1inv, J2inv;
    camera.unproject_with_jac(x1, &d1, &J1inv);
    camera.unproject_with_jac(x2, &d2, &J2inv);
    FixCameraRelativePoseRefiner refiner(d1, d2, J1inv, J2inv);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);

    // std::cout << "iter = " << stats.iterations << "\n";
    // std::cout << "initial_cost = " << stats.initial_cost << "\n";
    // std::cout << "cost = " << stats.cost << "\n";
    // std::cout << "lambda = " << stats.lambda << "\n";
    // std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    // std::cout << "step_norm = " << stats.step_norm << "\n";
    // std::cout << "grad_norm = " << stats.grad_norm << "\n";

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

bool test_tangent_sampson_camera_relative_pose_jacobian() {
    const size_t N = 50;
    std::string camera1_str = "3 RADIAL 1936 1296 2425.85 932.38 629.325 -0.04012 0.00123";
    std::string camera2_str = "0 SIMPLE_RADIAL 1936 1296 2425.85 932.383 628.265 -0.0397695";
    Camera camera1, camera2;
    camera1.initialize_from_txt(camera1_str);
    camera2.initialize_from_txt(camera2_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera1, camera2);

    BundleOptions opt;
    opt.refine_principal_point = false;
    opt.refine_focal_length = true;
    opt.refine_extra_params = true;
    std::vector<size_t> cam1_ref_idx = camera1.get_param_refinement_idx(opt);
    std::vector<size_t> cam2_ref_idx = camera2.get_param_refinement_idx(opt);

    // Add some noise
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d n;
        n.setRandom();
        x1[i] += 2.0 * n;
        n.setRandom();
        x2[i] += 2.0 * n;
    }

    CameraRelativePoseRefiner<UniformWeightVector, TestAccumulator> refiner(x1, x2, cam1_ref_idx, cam2_ref_idx, false);

    ImagePair pair(pose, camera1, camera2);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), ImagePair>(refiner, pair, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, pair);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pair);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_tangent_sampson_camera_relative_pose_refinement() {
    const size_t N = 50;
    std::string camera1_str = "3 RADIAL 1936 1296 2425.85 932.38 629.325 -0.04012 0.00123";
    std::string camera2_str = "0 SIMPLE_RADIAL 1936 1296 2425.85 932.383 628.265 -0.0397695";
    camera2_str = camera1_str;
    Camera camera1, camera2;
    camera1.initialize_from_txt(camera1_str);
    camera2.initialize_from_txt(camera2_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera1, camera2);

    BundleOptions opt;
    opt.refine_principal_point = false;
    opt.refine_focal_length = true;
    opt.refine_extra_params = true;
    std::vector<size_t> cam1_ref_idx = camera1.get_param_refinement_idx(opt);
    std::vector<size_t> cam2_ref_idx = camera2.get_param_refinement_idx(opt);

    // Add some noise
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d n;
        n.setRandom();
        x1[i] += 0.1 * n;
        n.setRandom();
        x2[i] += 0.1 * n;
    }

    double f1 = camera1.focal();
    double f2 = camera2.focal();
    for (int i = 0; i < N; ++i) {
        x1[i] /= f1;
        x2[i] /= f2;
    }
    camera1.rescale(1.0 / f1);
    camera2.rescale(1.0 / f2);

    ImagePair pair(pose, camera1, camera2);

    CameraRelativePoseRefiner refiner(x1, x2, cam1_ref_idx, cam2_ref_idx, false);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pair, bundle_opt, print_iteration);

    // std::cout << "iter = " << stats.iterations << "\n";
    // std::cout << "initial_cost = " << stats.initial_cost << "\n";
    // std::cout << "cost = " << stats.cost << "\n";
    // std::cout << "lambda = " << stats.lambda << "\n";
    // std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    // std::cout << "step_norm = " << stats.step_norm << "\n";
    // std::cout << "grad_norm = " << stats.grad_norm << "\n";

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

bool test_tangent_sampson_shared_camera_relative_pose_refinement() {
    const size_t N = 50;
    std::string camera_str = "0 SIMPLE_RADIAL 1936 1296 2425.85 932.383 628.265 -0.0397695";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    BundleOptions opt;
    opt.refine_principal_point = false;
    opt.refine_focal_length = true;
    opt.refine_extra_params = true;
    std::vector<size_t> cam_ref_idx = camera.get_param_refinement_idx(opt);

    // Add some noise
    for (int i = 0; i < N; ++i) {
        Eigen::Vector2d n;
        n.setRandom();
        x1[i] += 0.1 * n;
        n.setRandom();
        x2[i] += 0.1 * n;
    }

    double f = camera.focal();
    for (int i = 0; i < N; ++i) {
        x1[i] /= f;
        x2[i] /= f;
    }
    camera.rescale(1.0 / f);

    ImagePair pair(pose, camera, camera);

    CameraRelativePoseRefiner refiner(x1, x2, cam_ref_idx, cam_ref_idx, true);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pair, bundle_opt, print_iteration);

    // std::cout << "iter = " << stats.iterations << "\n";
    // std::cout << "initial_cost = " << stats.initial_cost << "\n";
    // std::cout << "cost = " << stats.cost << "\n";
    // std::cout << "lambda = " << stats.lambda << "\n";
    // std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    // std::cout << "step_norm = " << stats.step_norm << "\n";
    // std::cout << "grad_norm = " << stats.grad_norm << "\n";

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

} // namespace test::relative

using namespace test::relative;
std::vector<Test> register_optim_relative_test() {
    return {TEST(test_relative_pose_normal_acc),
            TEST(test_relative_pose_jacobian),
            TEST(test_relative_pose_refinement),
            TEST(test_shared_focal_relative_pose_normal_acc),
            TEST(test_shared_focal_relative_pose_jacobian),
            TEST(test_shared_focal_relative_pose_refinement),
            TEST(test_tangent_sampson_fix_camera_relative_pose_jacobian),
            TEST(test_tangent_sampson_fix_camera_relative_pose_refinement),
            TEST(test_tangent_sampson_camera_relative_pose_jacobian),
            TEST(test_tangent_sampson_camera_relative_pose_refinement),
            TEST(test_tangent_sampson_shared_camera_relative_pose_refinement)};
}