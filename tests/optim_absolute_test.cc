#include "example_cameras.h"
#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/optim/absolute.h>
#include <PoseLib/robust/optim/hybrid.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <algorithm>

using namespace poselib;

//////////////////////////////
// Absolute pose

namespace test::absolute {

namespace {

// Fixed reference pose so every test run sees the same optimization target.
CameraPose reference_pose() {
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(0.35, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-0.18, Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();
    return CameraPose(R, Eigen::Vector3d(0.25, -0.18, 0.45));
}

// Keep depths positive and moderately separated to avoid near-degenerate scenes.
double depth_sample(size_t idx, test_rng::Rng &rng) {
    return 4.0 + 0.45 * static_cast<double>(idx % 7) + 0.9 * static_cast<double>(idx / 7) + rng.uniform(0.0, 0.35);
}

// Apply deterministic point noise keyed by test/case id.
void add_point_noise(std::vector<Point2D> &x, double scale, const std::string &case_name, size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    for (size_t i = 0; i < x.size(); ++i) {
        x[i] += test_rng::symmetric_vec2(rng, scale);
    }
}

// Apply deterministic endpoint noise to every line observation.
void add_line_noise(std::vector<Line2D> &lines, double scale, const std::string &case_name, size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    for (Line2D &line : lines) {
        line.x1 += test_rng::symmetric_vec2(rng, scale);
        line.x2 += test_rng::symmetric_vec2(rng, scale);
    }
}

} // namespace

void setup_scene(int N, CameraPose &pose, std::vector<Point2D> &x, std::vector<Point3D> &X, Camera &cam,
                 std::vector<double> &weights, const std::string &case_name = "absolute_scene", size_t case_index = 0) {

    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    pose = reference_pose();
    // Build a deterministic central-camera scene directly from image samples so the
    // geometry stays well-conditioned across camera models and test order.
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
        const Eigen::Vector2d xi = image_sample(cam, i, static_cast<size_t>(N), rng);

        Eigen::Vector3d Xi;
        cam.unproject(xi, &Xi);
        Xi *= depth_sample(i, rng);
        x.push_back(xi);
        X.push_back(pose.apply_inverse(Xi));
        weights.push_back(1.0 + 0.1 * static_cast<double>(i));
    }
}

void setup_scene_w_lines(int N_pts, int N_lines, CameraPose &pose, std::vector<Point2D> &x, std::vector<Point3D> &X,
                         std::vector<Line2D> &lin2D, std::vector<Line3D> &lin3D, Camera &cam,
                         std::vector<double> &weights_pts, std::vector<double> &weights_lin,
                         const std::string &case_name = "absolute_line_scene", size_t case_index = 0) {

    std::vector<Point2D> x_all;
    std::vector<Point3D> X_all;
    std::vector<double> w_all;
    setup_scene(N_pts + 2 * N_lines, pose, x_all, X_all, cam, w_all, case_name, case_index);

    // Reuse the same deterministic point fixture and reinterpret consecutive pairs as lines.
    for (int i = 0; i < N_pts; ++i) {
        x.push_back(x_all[i]);
        X.push_back(X_all[i]);
        weights_pts.push_back(w_all[i]);
    }

    for (int i = N_pts; i < N_pts + 2 * N_lines; i += 2) {
        Line2D l2D;
        Line3D l3D;
        l2D.x1 = x_all[i];
        l2D.x2 = x_all[i + 1];
        l3D.X1 = X_all[i];
        l3D.X2 = X_all[i + 1];
        lin2D.push_back(l2D);
        lin3D.push_back(l3D);
        weights_lin.push_back(w_all[i]);
    }
}

////////////////////////////
// Point (2D-3D correspondence)

bool test_absolute_pose_normal_acc() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<double> weights;
    setup_scene(N, pose, x, X, camera, weights);

    Image image;
    image.pose = pose;
    image.camera = camera;

    NormalAccumulator acc;
    AbsolutePoseRefiner refiner(x, X);
    acc.initialize(refiner.num_params);

    // Check that residual is zero
    acc.reset_residual();
    refiner.compute_residual(acc, image);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    // Check the gradient is zero
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, image);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}

bool test_absolute_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<double> weights;
    setup_scene(N, pose, x, X, camera, weights);

    add_point_noise(x, 2e-3, "absolute_pose_jacobian_noise");

    Image image(pose, camera);

    AbsolutePoseRefiner<std::vector<double>, TestAccumulator> refiner(x, X, {}, weights);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), Image>(refiner, image, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, image);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, image);
    double r2 = 0.0;
    for (int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}

bool test_absolute_pose_jacobian_cameras() {
    const size_t N = 10;
    for (size_t camera_idx = 0; camera_idx < example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;
        std::vector<Eigen::Vector2d> x;
        std::vector<Eigen::Vector3d> X;
        std::vector<double> weights;
        setup_scene(N, pose, x, X, camera, weights, "absolute_pose_jacobian_cameras_scene", camera_idx);
        normalize_camera_points(x, &camera);

        BundleOptions opt;
        opt.refine_principal_point = true;
        opt.refine_focal_length = true;
        opt.refine_extra_params = true;
        std::vector<size_t> ref_idx = camera.get_param_refinement_idx(opt);

        Image image(pose, camera);
        AbsolutePoseRefiner<std::vector<double>, TestAccumulator> refiner(x, X, ref_idx, weights);

        const double delta = 1e-6;
        double jac_err = verify_jacobian<decltype(refiner), Image>(refiner, image, delta);
        REQUIRE_SMALL(jac_err, 1e-6)

        // Test that compute_residual and compute_jacobian are compatible
        TestAccumulator acc;
        acc.reset_residual();
        double r1 = refiner.compute_residual(acc, image);
        acc.reset_jacobian();
        refiner.compute_jacobian(acc, image);
        double r2 = 0.0;
        for (int i = 0; i < acc.rs.size(); ++i) {
            r2 += acc.weights[i] * acc.rs[i].squaredNorm();
        }
        REQUIRE_SMALL_M(std::abs(r1 - r2), 1e-10, test_rng::case_id(camera_str, camera_idx));
    }
    return true;
}

bool test_absolute_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<double> weights;
    setup_scene(N, pose, x, X, camera, weights);

    add_point_noise(x, 2e-3, "absolute_pose_refinement_noise");

    Image image(pose, camera);
    AbsolutePoseRefiner<> refiner(x, X);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, &image, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_absolute_pose_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "test_absolute_pose_refinement"));

    return true;
}

bool test_absolute_pose_weighted_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<double> weights;
    setup_scene(N, pose, x, X, camera, weights);

    add_point_noise(x, 5e-4, "absolute_pose_weighted_refinement_noise");

    Image image(pose, camera);
    NormalAccumulator acc;
    AbsolutePoseRefiner<std::vector<double>> refiner(x, X, {}, weights);
    acc.initialize(refiner.num_params);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl<decltype(refiner)>(refiner, &image, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_absolute_pose_weighted_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "test_absolute_pose_weighted_refinement"));

    return true;
}

bool test_absolute_pose_cameras_refinement() {
    const size_t N = 64;
    for (size_t camera_idx = 0; camera_idx < example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;
        std::vector<Eigen::Vector2d> x;
        std::vector<Eigen::Vector3d> X;
        std::vector<double> weights;
        setup_scene(N, pose, x, X, camera, weights, "absolute_pose_cameras_refinement_scene", camera_idx);
        add_point_noise(x, 2e-4 * camera.max_dim(), "absolute_pose_cameras_refinement_noise", camera_idx);
        normalize_camera_points(x, &camera);

        Image image(pose, camera);
        AbsolutePoseRefiner<> refiner(x, X);

        BundleOptions bundle_opt;
        bundle_opt.step_tol = 1e-12;
        bundle_opt.relative_cost_tol = 0.0;
        BundleStats stats = lm_impl(refiner, &image, bundle_opt, print_iteration);
        log_bundle_stats(stats, test_rng::case_id(camera_str, camera_idx));
        REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, test_rng::case_id(camera_str, camera_idx)));
    }

    return true;
}

/////////////////////////////
// Line (2D-3D)

bool test_line_absolute_pose_normal_acc() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D;
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    NormalAccumulator acc;
    PinholeLineAbsolutePoseRefiner refiner(lin2D, lin3D);
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

bool test_line_absolute_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D;
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    add_line_noise(lin2D, 2e-3, "line_absolute_pose_jacobian_noise");

    PinholeLineAbsolutePoseRefiner<std::vector<double>, TestAccumulator> refiner(lin2D, lin3D, w_lin);

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

bool test_line_absolute_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D;
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    add_line_noise(lin2D, 2e-3, "line_absolute_pose_refinement_noise");

    PinholeLineAbsolutePoseRefiner<decltype(w_lin)> refiner(lin2D, lin3D, w_lin);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_line_absolute_pose_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "test_line_absolute_pose_refinement"));

    return true;
}

////////////////////////////////////////////////
// Point + Line

bool test_point_line_absolute_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.5 2.5 0.1 -0.1";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D;
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    add_point_noise(x, 2e-3, "point_line_absolute_pose_jacobian_points");
    add_line_noise(lin2D, 2e-3, "point_line_absolute_pose_jacobian_lines");

    PinholeAbsolutePoseRefiner<decltype(w_pts), TestAccumulator> pts_refiner(x, X, w_pts);
    PinholeLineAbsolutePoseRefiner<decltype(w_lin), TestAccumulator> lin_refiner(lin2D, lin3D, w_lin);
    HybridRefiner<CameraPose, TestAccumulator> refiner;
    refiner.register_refiner(&pts_refiner);
    refiner.register_refiner(&lin_refiner);

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

bool test_point_line_absolute_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.1 0.9 0.1 0.2";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D;
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    add_point_noise(x, 2e-3, "point_line_absolute_pose_refinement_points");
    add_line_noise(lin2D, 2e-3, "point_line_absolute_pose_refinement_lines");

    PinholeAbsolutePoseRefiner<decltype(w_lin)> pts_refiner(x, X, w_pts);
    PinholeLineAbsolutePoseRefiner<decltype(w_lin)> lin_refiner(lin2D, lin3D, w_lin);
    HybridRefiner<CameraPose> refiner;
    refiner.register_refiner(&pts_refiner);
    refiner.register_refiner(&lin_refiner);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_point_line_absolute_pose_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "test_point_line_absolute_pose_refinement"));

    return true;
}

////////////////////////////////////////////////
// 1D Radial Camera model

bool test_1d_radial_absolute_pose_jacobian_cameras() {
    const size_t N = 25;
    for (size_t camera_idx = 0; camera_idx < radially_symmetric_example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = radially_symmetric_example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);

        CameraPose pose;
        std::vector<Eigen::Vector2d> x;
        std::vector<Eigen::Vector3d> X;
        std::vector<double> weights;
        setup_scene(N, pose, x, X, camera, weights, "radial_1d_absolute_pose_jacobian_scene", camera_idx);

        NormalAccumulator normal_acc;
        Radial1DAbsolutePoseRefiner<std::vector<double>> norm_refiner(x, X, camera, weights);
        normal_acc.initialize(norm_refiner.num_params);
        // Check that residual is zero
        normal_acc.reset_residual();
        norm_refiner.compute_residual(normal_acc, pose);
        double residual = normal_acc.get_residual();
        REQUIRE_SMALL(residual, 1e-6);

        add_point_noise(x, 2e-3, "radial_1d_absolute_pose_jacobian_noise", camera_idx);

        Radial1DAbsolutePoseRefiner<std::vector<double>, TestAccumulator> refiner(x, X, camera, weights);
        const double delta = 1e-8;
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
        REQUIRE_SMALL_M(std::abs(r1 - r2), 1e-10, test_rng::case_id(camera_str, camera_idx));
    }
    return true;
}

bool test_1d_radial_absolute_pose_cameras_refinement() {
    const size_t N = 48;
    for (size_t camera_idx = 0; camera_idx < radially_symmetric_example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = radially_symmetric_example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;
        std::vector<Eigen::Vector2d> x;
        std::vector<Eigen::Vector3d> X;
        std::vector<double> weights;
        setup_scene(N, pose, x, X, camera, weights, "radial_1d_absolute_pose_cameras_refinement_scene", camera_idx);
        add_point_noise(x, 1e-4 * camera.max_dim(), "radial_1d_absolute_pose_cameras_refinement_noise", camera_idx);
        normalize_camera_points(x, &camera);

        NormalAccumulator acc;
        Radial1DAbsolutePoseRefiner refiner(x, X, camera);

        BundleOptions bundle_opt;
        bundle_opt.step_tol = 1e-12;
        bundle_opt.relative_cost_tol = 0.0;
        BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);
        log_bundle_stats(stats, test_rng::case_id(camera_str, camera_idx));
        REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, test_rng::case_id(camera_str, camera_idx)));
    }

    return true;
}

} // namespace test::absolute

using namespace test::absolute;
std::vector<Test> register_optim_absolute_test() {
    return {// Points
            TEST(test_absolute_pose_normal_acc), TEST(test_absolute_pose_jacobian),
            TEST(test_absolute_pose_jacobian_cameras), TEST(test_absolute_pose_refinement),
            TEST(test_absolute_pose_weighted_refinement), TEST(test_absolute_pose_cameras_refinement),
            // Lines
            TEST(test_line_absolute_pose_normal_acc), TEST(test_line_absolute_pose_jacobian),
            TEST(test_line_absolute_pose_refinement),
            // Poiints + Lines
            TEST(test_point_line_absolute_pose_jacobian), TEST(test_point_line_absolute_pose_refinement),
            // 1D radial camera
            TEST(test_1d_radial_absolute_pose_jacobian_cameras), TEST(test_1d_radial_absolute_pose_cameras_refinement)};
}
