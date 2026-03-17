#include "example_cameras.h"
#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/optim/generalized_absolute.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <algorithm>

using namespace poselib;

//////////////////////////////
// Absolute pose

namespace test::gen_absolute {

namespace {

// Fixed rig pose so the generalized fixture is reproducible across runs.
CameraPose rig_pose() {
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(-0.28, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(0.22, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(0.1, Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();
    return CameraPose(R, Eigen::Vector3d(0.28, -0.12, 0.5));
}

// Small deterministic per-sensor offsets to create a non-degenerate camera rig.
CameraPose rig_sensor_pose(size_t index, size_t num_cams) {
    const double centered = static_cast<double>(index) - 0.5 * static_cast<double>(num_cams - 1);
    const Eigen::Matrix3d R = Eigen::AngleAxisd(0.05 * centered, Eigen::Vector3d::UnitY()).toRotationMatrix();
    return CameraPose(R, Eigen::Vector3d(0.12 * centered, 0.04 * (static_cast<double>(index % 2) - 0.5), 0.0));
}


// Vary depth by point and sensor while keeping all points in front of the cameras.
double depth_sample(size_t idx, size_t cam_idx, test_rng::Rng &rng) {
    return 4.5 + 0.35 * static_cast<double>(idx % 6) + 0.55 * static_cast<double>(cam_idx) + rng.uniform(0.0, 0.3);
}

// Apply deterministic perturbations to all observations in the multi-camera fixture.
void add_multi_point_noise(std::vector<std::vector<Point2D>> &x, double scale, const std::string &case_name,
                           size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    for (std::vector<Point2D> &points : x) {
        for (Point2D &point : points) {
            point += test_rng::symmetric_vec2(rng, scale);
        }
    }
}

} // namespace

void setup_scene(int Ncam, int N, CameraPose &pose, std::vector<std::vector<Point2D>> &x,
                 std::vector<std::vector<Point3D>> &X, std::vector<CameraPose> &cam_ext, Camera &cam0,
                 std::vector<Camera> &cam_int, std::vector<std::vector<double>> &weights,
                 const std::string &case_name = "gen_absolute_scene", size_t case_index = 0) {

    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    pose = rig_pose();

    // Build a deterministic generalized scene by composing a fixed rig pose with
    // per-sensor extrinsics and backprojected image samples.
    x.assign(Ncam, {});
    X.assign(Ncam, {});
    weights.assign(Ncam, {});
    cam_ext.clear();
    cam_int.clear();

    for (int k = 0; k < Ncam; ++k) {
        cam_int.push_back(cam0);
        cam_ext.push_back(rig_sensor_pose(static_cast<size_t>(k), static_cast<size_t>(Ncam)));
        CameraPose full_pose;
        full_pose.q = quat_multiply(cam_ext[k].q, pose.q);
        full_pose.t = cam_ext[k].rotate(pose.t) + cam_ext[k].t;
        const Camera &cam = cam_int[k];
        for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
            const Eigen::Vector2d xi = image_sample(cam, i, static_cast<size_t>(N), rng);

            Eigen::Vector3d Xi;
            cam.unproject(xi, &Xi);
            Xi *= depth_sample(i, static_cast<size_t>(k), rng);
            x[k].push_back(xi);
            X[k].push_back(full_pose.apply_inverse(Xi));
            weights[k].push_back(1.0 + 0.05 * static_cast<double>(i + k));
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

    add_multi_point_noise(x, 5e-4, "gen_absolute_pose_jacobian_noise");

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

    for (size_t camera_idx = 0; camera_idx < example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;
        std::vector<CameraPose> cam_ext;
        std::vector<Camera> cam_int;
        std::vector<std::vector<Eigen::Vector2d>> x;
        std::vector<std::vector<Eigen::Vector3d>> X;
        std::vector<std::vector<double>> weights;
        setup_scene(Ncam, N, pose, x, X, cam_ext, camera, cam_int, weights, "gen_absolute_pose_jacobian_cameras_scene",
                    camera_idx);
        add_multi_point_noise(x, 2e-4 * camera.max_dim(), "gen_absolute_pose_jacobian_cameras_noise", camera_idx);
        normalize_camera_points(x, &cam_int);

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
        REQUIRE_SMALL_M(std::abs(r1 - r2), 1e-8, test_rng::case_id(camera_str, camera_idx));
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

    add_multi_point_noise(x, 2e-3, "gen_absolute_pose_refinement_noise");

    GeneralizedAbsolutePoseRefiner refiner(x, X, cam_ext, cam_int);
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_gen_absolute_pose_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "test_gen_absolute_pose_refinement"));

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

    add_multi_point_noise(x, 2e-3, "gen_absolute_pose_weighted_refinement_noise");

    GeneralizedAbsolutePoseRefiner<decltype(weights)> refiner(x, X, cam_ext, cam_int, weights);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_gen_absolute_pose_weighted_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "test_gen_absolute_pose_weighted_refinement"));

    return true;
}

bool test_gen_absolute_pose_cameras_refinement() {
    const size_t N = 32;
    const size_t Ncam = 4;

    for (size_t camera_idx = 0; camera_idx < example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;
        std::vector<CameraPose> cam_ext;
        std::vector<Camera> cam_int;
        std::vector<std::vector<Eigen::Vector2d>> x;
        std::vector<std::vector<Eigen::Vector3d>> X;
        std::vector<std::vector<double>> weights;
        setup_scene(Ncam, N, pose, x, X, cam_ext, camera, cam_int, weights,
                    "gen_absolute_pose_cameras_refinement_scene", camera_idx);
        add_multi_point_noise(x, 2e-4 * camera.max_dim(), "gen_absolute_pose_cameras_refinement_noise", camera_idx);
        normalize_camera_points(x, &cam_int);

        GeneralizedAbsolutePoseRefiner<decltype(weights)> refiner(x, X, cam_ext, cam_int, weights);

        BundleOptions bundle_opt;
        bundle_opt.step_tol = 1e-12;
        bundle_opt.relative_cost_tol = 0.0;
        BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);
        log_bundle_stats(stats, test_rng::case_id(camera_str, camera_idx));
        REQUIRE(check_bundle_cost_gradient_and_step(stats, 1e-3, 1e-6, test_rng::case_id(camera_str, camera_idx)));
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
