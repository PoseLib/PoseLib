#include "example_cameras.h"
#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/optim/generalized_absolute.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/robust_loss.h>

using namespace poselib;

//////////////////////////////
// Absolute pose

namespace test::gen_absolute {

void setup_scene(int Ncam, int N, CameraPose &pose, std::vector<std::vector<Point2D>> &x,
                 std::vector<std::vector<Point3D>> &X, std::vector<CameraPose> &cam_ext, Camera &cam0,
                 std::vector<Camera> &cam_int, std::vector<std::vector<double>> &weights) {

    pose.q.setRandom();
    pose.q.normalize();
    pose.t.setRandom();

    x.resize(Ncam);
    X.resize(Ncam);
    weights.resize(Ncam);

    for (int k = 0; k < Ncam; ++k) {
        Eigen::VectorXd depth_factor(N);
        depth_factor.setRandom();
        cam_int.push_back(cam0);
        cam_ext.push_back(CameraPose());
        cam_ext[k].q.setRandom();
        cam_ext[k].q.normalize();
        cam_ext[k].t.setRandom();
        CameraPose full_pose;
        full_pose.q = quat_multiply(cam_ext[k].q, pose.q);
        full_pose.t = cam_ext[k].rotate(pose.t) + cam_ext[k].t;
        Camera cam = cam_int[k];
        for (size_t i = 0; i < N; ++i) {
            Eigen::Vector2d xi;
            // we sample points in [0.2, 0.8] of the image
            xi.setRandom();
            xi *= 0.3;
            xi += Eigen::Vector2d(0.5, 0.5);
            // xi = [-1, 1] -> xi = [0.2, 0.8]
            xi << xi(0) * cam.width, xi(1) * cam.height;

            Eigen::Vector3d Xi;
            cam.unproject(xi, &Xi);
            Xi *= (2.0 + 10.0 * depth_factor(i)); // backproject
            x[k].push_back(xi);
            X[k].push_back(full_pose.apply_inverse(Xi));
            weights[k].push_back(1.0 * (i + 1.0));
        }
    }
}

bool test_gen_absolute_pose_normal_acc() {
    const size_t N = 10;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<CameraPose> cam_ext;
    std::vector<Camera> cam_int;
    std::vector<std::vector<Eigen::Vector2d>> x;
    std::vector<std::vector<Eigen::Vector3d>> X;
    std::vector<std::vector<double>> weights;
    setup_scene(Ncam, N, pose, x, X, cam_ext, camera, cam_int, weights);

    NormalAccumulator acc;
    GeneralizedAbsolutePoseRefiner<std::vector<std::vector<double>>> refiner(x, X, cam_ext, cam_int, weights);
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

bool test_gen_absolute_pose_jacobian() {
    const size_t N = 10;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<CameraPose> cam_ext;
    std::vector<Camera> cam_int;
    std::vector<std::vector<Eigen::Vector2d>> x;
    std::vector<std::vector<Eigen::Vector3d>> X;
    std::vector<std::vector<double>> weights;
    setup_scene(Ncam, N, pose, x, X, cam_ext, camera, cam_int, weights);

    // add noise
    for (int k = 0; k < Ncam; ++k) {
        for (size_t i = 0; i < N; ++i) {
            Eigen::Vector2d noise;
            noise.setRandom();
            x[k][i] += 0.001 * noise;
        }
    }

    GeneralizedAbsolutePoseRefiner<std::vector<std::vector<double>>, TestAccumulator> refiner(x, X, cam_ext, cam_int,
                                                                                              weights);

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

bool test_gen_absolute_pose_jacobian_cameras() {
    const size_t N = 10;
    const size_t Ncam = 4;

    for (std::string camera_str : example_cameras) {
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;
        std::vector<CameraPose> cam_ext;
        std::vector<Camera> cam_int;
        std::vector<std::vector<Eigen::Vector2d>> x;
        std::vector<std::vector<Eigen::Vector3d>> X;
        std::vector<std::vector<double>> weights;
        setup_scene(Ncam, N, pose, x, X, cam_ext, camera, cam_int, weights);

        // add noise
        double max_dim = camera.max_dim();
        for (int k = 0; k < Ncam; ++k) {
            for (size_t i = 0; i < N; ++i) {
                Eigen::Vector2d noise;
                noise.setRandom();
                x[k][i] += 0.001 * noise * max_dim;
            }
        }

        GeneralizedAbsolutePoseRefiner<std::vector<std::vector<double>>, TestAccumulator> refiner(x, X, cam_ext,
                                                                                                  cam_int, weights);

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
        REQUIRE_SMALL(std::abs(r1 - r2), 1e-8);
    }
    return true;
}

bool test_gen_absolute_pose_refinement() {
    const size_t N = 10;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<CameraPose> cam_ext;
    std::vector<Camera> cam_int;
    std::vector<std::vector<Eigen::Vector2d>> x;
    std::vector<std::vector<Eigen::Vector3d>> X;
    std::vector<std::vector<double>> weights;
    setup_scene(Ncam, N, pose, x, X, cam_ext, camera, cam_int, weights);

    // add noise
    for (int k = 0; k < Ncam; ++k) {
        for (size_t i = 0; i < N; ++i) {
            Eigen::Vector2d noise;
            noise.setRandom();
            x[k][i] += 0.01 * noise;
        }
    }

    GeneralizedAbsolutePoseRefiner refiner(x, X, cam_ext, cam_int);
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);

    std::cout << "iter = " << stats.iterations << "\n";
    std::cout << "initial_cost = " << stats.initial_cost << "\n";
    std::cout << "cost = " << stats.cost << "\n";
    std::cout << "lambda = " << stats.lambda << "\n";
    std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    std::cout << "step_norm = " << stats.step_norm << "\n";
    std::cout << "grad_norm = " << stats.grad_norm << "\n";

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

bool test_gen_absolute_pose_weighted_refinement() {
    const size_t N = 10;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<CameraPose> cam_ext;
    std::vector<Camera> cam_int;
    std::vector<std::vector<Eigen::Vector2d>> x;
    std::vector<std::vector<Eigen::Vector3d>> X;
    std::vector<std::vector<double>> weights;
    setup_scene(Ncam, N, pose, x, X, cam_ext, camera, cam_int, weights);

    // add noise
    for (int k = 0; k < Ncam; ++k) {
        for (size_t i = 0; i < N; ++i) {
            Eigen::Vector2d noise;
            noise.setRandom();
            x[k][i] += 0.01 * noise;
        }
    }

    GeneralizedAbsolutePoseRefiner<decltype(weights)> refiner(x, X, cam_ext, cam_int, weights);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);

    std::cout << "iter = " << stats.iterations << "\n";
    std::cout << "initial_cost = " << stats.initial_cost << "\n";
    std::cout << "cost = " << stats.cost << "\n";
    std::cout << "lambda = " << stats.lambda << "\n";
    std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    std::cout << "step_norm = " << stats.step_norm << "\n";
    std::cout << "grad_norm = " << stats.grad_norm << "\n";

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

bool test_gen_absolute_pose_cameras_refinement() {
    const size_t N = 10;
    const size_t Ncam = 4;

    for (std::string camera_str : example_cameras) {
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;
        std::vector<CameraPose> cam_ext;
        std::vector<Camera> cam_int;
        std::vector<std::vector<Eigen::Vector2d>> x;
        std::vector<std::vector<Eigen::Vector3d>> X;
        std::vector<std::vector<double>> weights;
        setup_scene(Ncam, N, pose, x, X, cam_ext, camera, cam_int, weights);

        const double max_dim = camera.max_dim();

        // add noise
        for (int k = 0; k < Ncam; ++k) {
            for (size_t i = 0; i < N; ++i) {
                Eigen::Vector2d noise;
                noise.setRandom();
                x[k][i] += 0.01 * noise * max_dim;
            }
        }

        // Rescale all points by maxdim to improve numerics
        double scale = 0.5 * max_dim;
        for (int k = 0; k < Ncam; ++k) {
            for (size_t i = 0; i < N; ++i) {
                x[k][i] /= scale;
            }
            cam_int[k].rescale(1 / scale);
        }

        GeneralizedAbsolutePoseRefiner<decltype(weights)> refiner(x, X, cam_ext, cam_int, weights);

        BundleOptions bundle_opt;
        bundle_opt.step_tol = 1e-16;
        BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);

        std::cout << "iter = " << stats.iterations << "\n";
        std::cout << "initial_cost = " << stats.initial_cost << "\n";
        std::cout << "cost = " << stats.cost << "\n";
        std::cout << "lambda = " << stats.lambda << "\n";
        std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
        std::cout << "step_norm = " << stats.step_norm << "\n";
        std::cout << "grad_norm = " << stats.grad_norm << "\n";
        std::cout << "camera = " << camera_str << "\n";
        REQUIRE_SMALL(stats.step_norm, 1e-6);
        REQUIRE_SMALL(stats.grad_norm, 1e-3);
        REQUIRE(stats.cost < stats.initial_cost);
    }

    return true;
}
} // namespace test::gen_absolute

using namespace test::gen_absolute;
std::vector<Test> register_optim_gen_absolute_test() {
    return {TEST(test_gen_absolute_pose_normal_acc),          TEST(test_gen_absolute_pose_jacobian),
            TEST(test_gen_absolute_pose_jacobian_cameras),    TEST(test_gen_absolute_pose_refinement),
            TEST(test_gen_absolute_pose_weighted_refinement), TEST(test_gen_absolute_pose_cameras_refinement)};
}
