#include "example_cameras.h"
#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust.h>
#include <PoseLib/robust/optim/generalized_absolute.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/sampling.h>
#include <algorithm>

using namespace poselib;

//////////////////////////////
// Generalized absolute pose and scale

namespace test::gen_absolute_scale {

namespace {

// Fixed rig pose so the fixture is reproducible across runs.
CameraPose rig_pose() {
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(-0.24, Eigen::Vector3d::UnitX()) * Eigen::AngleAxisd(0.31, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(0.12, Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();
    return CameraPose(R, Eigen::Vector3d(0.32, -0.15, 0.44));
}

// Rig with distinct camera centers. The scale is only observable through the parallax
// between them, so the centers are spread out w.r.t. the depth of the points.
std::vector<CameraPose> rig_extrinsics(size_t num_cams) {
    std::vector<CameraPose> camera_ext;
    for (size_t k = 0; k < num_cams; ++k) {
        const double centered = static_cast<double>(k) - 0.5 * static_cast<double>(num_cams - 1);
        const Eigen::Matrix3d R = Eigen::AngleAxisd(0.12 * centered, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const Eigen::Vector3d center(0.45 * centered, 0.2 * (static_cast<double>(k % 2) - 0.5), 0.18 * centered);
        camera_ext.emplace_back(R, -R * center);
    }
    return camera_ext;
}

// Rig which only rotates around a single center, i.e. which cannot constrain the scale. With a
// center away from the origin the camera centers -Rk' * tk recovered from the extrinsics agree
// only up to rounding.
std::vector<CameraPose> rotating_rig_extrinsics(size_t num_cams,
                                                const Eigen::Vector3d &center = Eigen::Vector3d::Zero()) {
    std::vector<CameraPose> camera_ext;
    for (size_t k = 0; k < num_cams; ++k) {
        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(0.9 * static_cast<double>(k), Eigen::Vector3d::UnitY()).toRotationMatrix();
        camera_ext.emplace_back(R, -R * center);
    }
    return camera_ext;
}

struct Scene {
    CameraPose pose;
    double scale;
    std::vector<CameraPose> camera_ext;
    std::vector<Camera> cameras;
    std::vector<std::vector<Point2D>> x;
    std::vector<std::vector<Point3D>> X;
};

// Builds a rig observing 3D points which live at another scale than the rig itself: the pose
// of the k:th rig camera is (Rk*R, Rk*t + scale*tk).
Scene setup_scene(const std::vector<CameraPose> &camera_ext, size_t num_pts, double scale, const Camera &camera,
                  const std::string &case_name, size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    const size_t num_cams = camera_ext.size();

    Scene scene;
    scene.pose = rig_pose();
    scene.scale = scale;
    scene.camera_ext = camera_ext;
    scene.cameras.assign(num_cams, camera);
    scene.x.assign(num_cams, {});
    scene.X.assign(num_cams, {});

    const ScaledCameraPose scaled_pose(scene.pose, scale);
    for (size_t k = 0; k < num_cams; ++k) {
        const CameraPose full_pose = scaled_pose.camera_pose(camera_ext[k]);
        for (size_t i = 0; i < num_pts; ++i) {
            const Eigen::Vector2d xi = image_sample(camera, i, num_pts, rng);
            Eigen::Vector3d Xi;
            camera.unproject(xi, &Xi);
            Xi *= 3.5 + 0.45 * static_cast<double>(i % 7) + rng.uniform(0.0, 0.4);
            scene.x[k].push_back(xi);
            scene.X[k].push_back(full_pose.apply_inverse(Xi));
        }
    }
    return scene;
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

// Perturb the observations and replace a fraction of them with random image points.
size_t add_noise_and_outliers(Scene *scene, double noise, double outlier_ratio, const std::string &case_name) {
    test_rng::Rng rng = test_rng::make_rng(case_name);
    size_t num_outliers = 0;
    for (size_t k = 0; k < scene->x.size(); ++k) {
        const Camera &camera = scene->cameras[k];
        for (size_t i = 0; i < scene->x[k].size(); ++i) {
            if (rng.uniform(0.0, 1.0) < outlier_ratio) {
                scene->x[k][i] = Eigen::Vector2d(rng.uniform(0.0, camera.width), rng.uniform(0.0, camera.height));
                num_outliers++;
            } else {
                scene->x[k][i] += test_rng::symmetric_vec2(rng, noise);
            }
        }
    }
    return num_outliers;
}

// Unit bearings of the observations, in the coordinate system of the observing rig camera.
std::vector<std::vector<Point3D>> scene_bearings(const Scene &scene) {
    std::vector<std::vector<Point3D>> bearings(scene.x.size());
    for (size_t k = 0; k < scene.x.size(); ++k) {
        for (const Point2D &xi : scene.x[k]) {
            Eigen::Vector3d bi;
            scene.cameras[k].unproject(xi, &bi);
            bearings[k].push_back(bi.normalized());
        }
    }
    return bearings;
}

// Per-residual weights which differ across cameras and across points, so that a weight
// dropped or mixed up in the scale column shows up in the jacobian checks.
std::vector<std::vector<double>> scene_weights(const Scene &scene) {
    std::vector<std::vector<double>> weights(scene.x.size());
    for (size_t k = 0; k < scene.x.size(); ++k) {
        for (size_t i = 0; i < scene.x[k].size(); ++i) {
            weights[k].push_back(1.0 + 0.05 * static_cast<double>(i + k));
        }
    }
    return weights;
}

// Verify the gradient of the robust cost actually minimized by lm_impl, i.e. the jacobians
// after the loss weights of the accumulator have been applied, against central differences of
// the accumulated cost. With cost = sum_i w_i * loss(|r_i|^2) the gradient is 2 * Jtr.
// Returns the maximum relative error over the parameters.
template <typename Refiner, typename Model>
double verify_gradient(Refiner &refiner, const Model &m, const BundleOptions &opt, double delta) {
    const int num_params = static_cast<int>(refiner.num_params);
    NormalAccumulator acc;
    acc.initialize(num_params, RobustLoss::factory(opt));

    acc.reset_jacobian();
    refiner.compute_jacobian(acc, m);
    const Eigen::VectorXd grad = 2.0 * acc.Jtr;

    Eigen::VectorXd grad_est(num_params);
    for (int i = 0; i < num_params; ++i) {
        Eigen::VectorXd dp = Eigen::VectorXd::Zero(num_params);
        dp(i) = delta;

        acc.reset_residual();
        refiner.compute_residual(acc, refiner.step(dp, m));
        const double cost_forward = acc.residual_acc;

        acc.reset_residual();
        refiner.compute_residual(acc, refiner.step(-dp, m));
        const double cost_backward = acc.residual_acc;

        grad_est(i) = (cost_forward - cost_backward) / (2.0 * delta);
    }

    const double err = (grad - grad_est).norm() / std::max(1.0, grad_est.norm());
    if (err > 1e-6) {
        std::cout << "Gradient failure!\n grad=" << grad.transpose() << "\n grad (finite) = " << grad_est.transpose()
                  << "\n";
    }
    return err;
}

double rotation_error(const CameraPose &pose, const CameraPose &pose_gt) { return (pose.R() - pose_gt.R()).norm(); }

double translation_error(const CameraPose &pose, const CameraPose &pose_gt) { return (pose.t - pose_gt.t).norm(); }

} // namespace

bool test_gen_absolute_scale_jacobian() {
    const size_t N = 10;
    const size_t Ncam = 4;

    for (size_t camera_idx = 0; camera_idx < example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);

        Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_jacobian", camera_idx);
        add_multi_point_noise(scene.x, 2e-4 * camera.max_dim(), "gen_absolute_scale_jacobian_noise", camera_idx);
        normalize_camera_points(scene.x, &scene.cameras);

        GeneralizedAbsolutePoseScaleRefiner<UniformWeightVectors, TestAccumulator> refiner(
            scene.x, scene.X, scene.camera_ext, scene.cameras);

        const ScaledCameraPose scaled_pose(scene.pose, scene.scale);
        const double delta = 1e-6;
        double jac_err = verify_jacobian<decltype(refiner), ScaledCameraPose>(refiner, scaled_pose, delta);
        REQUIRE_SMALL_M(jac_err, 1e-6, test_rng::case_id(camera_str, camera_idx));

        // Test that compute_residual and compute_jacobian are compatible
        TestAccumulator acc;
        acc.reset_residual();
        double r1 = refiner.compute_residual(acc, scaled_pose);
        acc.reset_jacobian();
        refiner.compute_jacobian(acc, scaled_pose);
        double r2 = 0.0;
        for (size_t i = 0; i < acc.rs.size(); ++i) {
            r2 += acc.weights[i] * acc.rs[i].squaredNorm();
        }
        REQUIRE_SMALL_M(std::abs(r1 - r2), 1e-8, test_rng::case_id(camera_str, camera_idx));
    }
    return true;
}

bool test_gen_absolute_scale_bearing_jacobian() {
    const size_t N = 10;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_bearing_jacobian");
    add_multi_point_noise(scene.x, 5e-4, "gen_absolute_scale_bearing_jacobian_noise");
    const std::vector<std::vector<Point3D>> bearings = scene_bearings(scene);

    BearingGeneralizedAbsolutePoseScaleRefiner<UniformWeightVectors, TestAccumulator> refiner(bearings, scene.X,
                                                                                              scene.camera_ext);

    const ScaledCameraPose scaled_pose(scene.pose, scene.scale);
    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), ScaledCameraPose>(refiner, scaled_pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6);

    return true;
}

// Same jacobian checks with per-residual weights, which the RANSAC path does not use but the
// public refinement entry points do.
bool test_gen_absolute_scale_weighted_jacobian() {
    const size_t N = 10;
    const size_t Ncam = 4;

    for (size_t camera_idx = 0; camera_idx < example_cameras.size(); ++camera_idx) {
        const std::string &camera_str = example_cameras[camera_idx];
        log_test_case("camera", test_rng::case_id(camera_str, camera_idx));
        Camera camera;
        camera.initialize_from_txt(camera_str);

        Scene scene =
            setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_weighted_jacobian", camera_idx);
        add_multi_point_noise(scene.x, 2e-4 * camera.max_dim(), "gen_absolute_scale_weighted_jacobian_noise",
                              camera_idx);
        normalize_camera_points(scene.x, &scene.cameras);
        const std::vector<std::vector<double>> weights = scene_weights(scene);

        GeneralizedAbsolutePoseScaleRefiner<std::vector<std::vector<double>>, TestAccumulator> refiner(
            scene.x, scene.X, scene.camera_ext, scene.cameras, weights);

        const ScaledCameraPose scaled_pose(scene.pose, scene.scale);
        const double delta = 1e-6;
        double jac_err = verify_jacobian<decltype(refiner), ScaledCameraPose>(refiner, scaled_pose, delta);
        REQUIRE_SMALL_M(jac_err, 1e-6, test_rng::case_id(camera_str, camera_idx));
    }

    // On an ideal pinhole rig every observation reaches the accumulator, so we can also check
    // that the scale column forwards the weights in order and untouched
    Camera camera;
    camera.initialize_from_txt("0 PINHOLE 1 1 1.0 1.0 0.0 0.0");
    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_weight_forwarding");
    const std::vector<std::vector<double>> weights = scene_weights(scene);

    GeneralizedAbsolutePoseScaleRefiner<std::vector<std::vector<double>>, TestAccumulator> refiner(
        scene.x, scene.X, scene.camera_ext, scene.cameras, weights);
    TestAccumulator acc;
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, ScaledCameraPose(scene.pose, scene.scale));

    REQUIRE_EQ(acc.weights.size(), Ncam * N);
    double weight_err = 0.0;
    size_t idx = 0;
    for (size_t k = 0; k < Ncam; ++k) {
        for (size_t i = 0; i < N; ++i, ++idx) {
            weight_err = std::max(weight_err, std::abs(acc.weights[idx] - weights[k][i]));
            REQUIRE_EQ(acc.Js[idx].cols(), Eigen::Index(7));
        }
    }
    REQUIRE_SMALL(weight_err, 1e-12);

    return true;
}

bool test_gen_absolute_scale_bearing_weighted_jacobian() {
    const size_t N = 10;
    const size_t Ncam = 4;

    // Bearings are camera model agnostic, so we take them through a strongly distorted model
    std::string camera_str = "6 OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 "
                             "0.000153 0.000571";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_bearing_weighted_jacobian");
    add_multi_point_noise(scene.x, 0.2, "gen_absolute_scale_bearing_weighted_jacobian_noise");
    const std::vector<std::vector<Point3D>> bearings = scene_bearings(scene);
    const std::vector<std::vector<double>> weights = scene_weights(scene);

    BearingGeneralizedAbsolutePoseScaleRefiner<std::vector<std::vector<double>>, TestAccumulator> refiner(
        bearings, scene.X, scene.camera_ext, weights);

    const ScaledCameraPose scaled_pose(scene.pose, scene.scale);
    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner), ScaledCameraPose>(refiner, scaled_pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6);

    return true;
}

// The jacobians reach lm_impl through the loss weights of the accumulator, so check the
// gradient of the robust cost itself, with the truncated loss used by the RANSAC refinement
// and with the trivial loss used by the public refinement entry points.
bool test_gen_absolute_scale_loss_gradient() {
    const size_t N = 12;
    const size_t Ncam = 4;

    std::string camera_str = "5 OPENCV 3200 2400 2575.94 2608.29 1599.26 1257.13 0.141865 -0.465301 0 0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_loss_gradient");
    // Displace a third of the observations well past the truncation threshold, so that the
    // truncated loss actually zeroes part of the gradient
    for (size_t k = 0; k < Ncam; ++k) {
        for (size_t i = 0; i < N; i += 3) {
            scene.x[k][i] += Eigen::Vector2d(37.0, -29.0);
        }
    }
    add_multi_point_noise(scene.x, 0.4, "gen_absolute_scale_loss_gradient_noise");
    const std::vector<std::vector<Point3D>> bearings = scene_bearings(scene);
    const std::vector<std::vector<double>> weights = scene_weights(scene);

    const ScaledCameraPose scaled_pose(scene.pose, scene.scale);
    const double delta = 1e-6;

    for (size_t loss_idx = 0; loss_idx < 2; ++loss_idx) {
        BundleOptions bundle_opt;
        bundle_opt.loss_type = loss_idx == 0 ? BundleOptions::LossType::TRIVIAL : BundleOptions::LossType::TRUNCATED;
        // Well clear of both the inlier and the outlier residuals
        bundle_opt.loss_scale = 4.0;
        const std::string label = loss_idx == 0 ? "trivial" : "truncated";
        log_test_case("loss", label);

        GeneralizedAbsolutePoseScaleRefiner<std::vector<std::vector<double>>> refiner(
            scene.x, scene.X, scene.camera_ext, scene.cameras, weights);
        REQUIRE_SMALL_M(verify_gradient(refiner, scaled_pose, bundle_opt, delta), 1e-6, label);

        BundleOptions bearing_opt = bundle_opt;
        bearing_opt.loss_scale = 4.0 / camera.focal();
        BearingGeneralizedAbsolutePoseScaleRefiner<std::vector<std::vector<double>>> bearing_refiner(
            bearings, scene.X, scene.camera_ext, weights);
        REQUIRE_SMALL_M(verify_gradient(bearing_refiner, scaled_pose, bearing_opt, delta), 1e-6, label);
    }

    return true;
}

bool test_gen_absolute_scale_refinement() {
    const size_t N = 16;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_refinement");
    add_multi_point_noise(scene.x, 1e-3, "gen_absolute_scale_refinement_noise");

    // Start away from the solution in all seven parameters
    ScaledCameraPose scaled_pose(scene.pose, scene.scale);
    scaled_pose.pose.q = quat_step_post(scaled_pose.pose.q, Eigen::Vector3d(0.02, -0.015, 0.01));
    scaled_pose.pose.t += Eigen::Vector3d(0.03, 0.02, -0.025);
    scaled_pose.scale *= 1.1;

    GeneralizedAbsolutePoseScaleRefiner refiner(scene.x, scene.X, scene.camera_ext, scene.cameras);
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &scaled_pose, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_gen_absolute_scale_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "test_gen_absolute_scale_refinement"));

    REQUIRE_SMALL(rotation_error(scaled_pose.pose, scene.pose), 1e-2);
    REQUIRE_SMALL(translation_error(scaled_pose.pose, scene.pose), 1e-2);
    REQUIRE_SMALL(scaled_pose.scale - scene.scale, 1e-2);

    return true;
}

// On noise-free data the seven parameters are recoverable exactly, so the refinement has to
// land on the ground truth and not merely reduce the cost.
bool test_gen_absolute_scale_refinement_exact() {
    const size_t N = 16;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_refinement_exact");

    ScaledCameraPose scaled_pose(scene.pose, scene.scale);
    scaled_pose.pose.q = quat_step_post(scaled_pose.pose.q, Eigen::Vector3d(0.05, -0.04, 0.03));
    scaled_pose.pose.t += Eigen::Vector3d(0.08, -0.06, 0.07);
    scaled_pose.scale *= 1.25;

    GeneralizedAbsolutePoseScaleRefiner refiner(scene.x, scene.X, scene.camera_ext, scene.cameras);
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-14;
    bundle_opt.relative_cost_tol = 0.0;
    bundle_opt.max_iterations = 200;
    BundleStats stats = lm_impl(refiner, &scaled_pose, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_gen_absolute_scale_refinement_exact");

    REQUIRE_SMALL(stats.cost, 1e-18);
    REQUIRE_SMALL(rotation_error(scaled_pose.pose, scene.pose), 1e-8);
    REQUIRE_SMALL(translation_error(scaled_pose.pose, scene.pose), 1e-8);
    REQUIRE_SMALL(scaled_pose.scale - scene.scale, 1e-8);

    return true;
}

// On noisy data there is no ground truth to land on, so instead check that adding the scale
// column keeps the LM iteration well behaved: the cost never goes up and ends below the cost
// at the perturbed starting point.
bool test_gen_absolute_scale_refinement_monotonic() {
    const size_t N = 24;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_scale_refinement_monotonic");
    add_multi_point_noise(scene.x, 4e-3, "gen_absolute_scale_refinement_monotonic_noise");

    ScaledCameraPose scaled_pose(scene.pose, scene.scale);
    scaled_pose.pose.q = quat_step_post(scaled_pose.pose.q, Eigen::Vector3d(0.04, 0.03, -0.02));
    scaled_pose.pose.t += Eigen::Vector3d(-0.05, 0.06, 0.04);
    scaled_pose.scale *= 0.85;

    std::vector<double> costs;
    IterationCallback record = [&costs](const BundleStats &iter_stats, RobustLoss *) {
        costs.push_back(iter_stats.cost);
    };

    GeneralizedAbsolutePoseScaleRefiner refiner(scene.x, scene.X, scene.camera_ext, scene.cameras);
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    bundle_opt.relative_cost_tol = 0.0;
    BundleStats stats = lm_impl(refiner, &scaled_pose, bundle_opt, record);
    log_bundle_stats(stats, "test_gen_absolute_scale_refinement_monotonic");

    REQUIRE(costs.size() > 1);
    for (size_t i = 1; i < costs.size(); ++i) {
        REQUIRE_M(costs[i] <= costs[i - 1], test_rng::case_id("iteration", i));
    }
    REQUIRE(costs.back() < stats.initial_cost);
    REQUIRE(stats.cost < stats.initial_cost);

    return true;
}

// At the true scale the scale column contributes nothing that the rigid refiner does not
// already have, so both have to converge to the same pose from the same starting point.
bool test_gen_absolute_scale_matches_rigid() {
    const size_t N = 16;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 1.0, camera, "gen_absolute_scale_matches_rigid");

    CameraPose start = scene.pose;
    start.q = quat_step_post(start.q, Eigen::Vector3d(0.03, -0.02, 0.025));
    start.t += Eigen::Vector3d(0.05, 0.04, -0.03);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-14;
    bundle_opt.relative_cost_tol = 0.0;
    bundle_opt.max_iterations = 200;

    CameraPose rigid_pose = start;
    GeneralizedAbsolutePoseRefiner rigid_refiner(scene.x, scene.X, scene.camera_ext, scene.cameras);
    BundleStats rigid_stats = lm_impl(rigid_refiner, &rigid_pose, bundle_opt, print_iteration);
    log_bundle_stats(rigid_stats, "rigid");

    ScaledCameraPose scaled_pose(start, 1.0);
    GeneralizedAbsolutePoseScaleRefiner scaled_refiner(scene.x, scene.X, scene.camera_ext, scene.cameras);
    BundleStats scaled_stats = lm_impl(scaled_refiner, &scaled_pose, bundle_opt, print_iteration);
    log_bundle_stats(scaled_stats, "scaled");

    REQUIRE_SMALL(rotation_error(scaled_pose.pose, rigid_pose), 1e-8);
    REQUIRE_SMALL(translation_error(scaled_pose.pose, rigid_pose), 1e-8);
    REQUIRE_SMALL(scaled_pose.scale - 1.0, 1e-8);

    return true;
}

// A rig which only rotates around one center has an identically zero scale column, so the
// scale cannot move while the six pose parameters still have to converge.
bool test_gen_absolute_scale_single_center_refinement() {
    const size_t N = 24;
    const size_t Ncam = 3;

    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rotating_rig_extrinsics(Ncam), N, 1.0, camera, "gen_absolute_scale_single_center");

    const ScaledCameraPose gt(scene.pose, 1.7);
    GeneralizedAbsolutePoseScaleRefiner<UniformWeightVectors, TestAccumulator> test_refiner(
        scene.x, scene.X, scene.camera_ext, scene.cameras);
    TestAccumulator acc;
    acc.reset_jacobian();
    test_refiner.compute_jacobian(acc, gt);
    double scale_column = 0.0;
    for (const Eigen::MatrixXd &J : acc.Js) {
        scale_column = std::max(scale_column, J.col(6).cwiseAbs().maxCoeff());
    }
    REQUIRE_EQ(scale_column, 0.0);

    // The scale is free to start anywhere; the residuals do not depend on it
    const double start_scale = 1.7;
    ScaledCameraPose scaled_pose(scene.pose, start_scale);
    scaled_pose.pose.q = quat_step_post(scaled_pose.pose.q, Eigen::Vector3d(0.04, -0.03, 0.02));
    scaled_pose.pose.t += Eigen::Vector3d(0.06, 0.05, -0.04);

    GeneralizedAbsolutePoseScaleRefiner refiner(scene.x, scene.X, scene.camera_ext, scene.cameras);
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-14;
    bundle_opt.relative_cost_tol = 0.0;
    bundle_opt.max_iterations = 200;
    BundleStats stats = lm_impl(refiner, &scaled_pose, bundle_opt, print_iteration);
    log_bundle_stats(stats, "test_gen_absolute_scale_single_center_refinement");

    REQUIRE_SMALL(scaled_pose.scale - start_scale, 1e-12);
    REQUIRE_SMALL(rotation_error(scaled_pose.pose, scene.pose), 1e-8);
    REQUIRE_SMALL(translation_error(scaled_pose.pose, scene.pose), 1e-8);

    return true;
}

bool test_gen_absolute_pose_scale_ransac() {
    const size_t N = 30;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1200 800 800.0 800.0 600.0 400.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_pose_scale_ransac");
    const size_t num_outliers = add_noise_and_outliers(&scene, 0.5, 0.25, "gen_absolute_pose_scale_ransac_noise");

    AbsolutePoseOptions opt;
    opt.max_error = 2.0;
    opt.ransac.min_iterations = 100;
    opt.ransac.max_iterations = 1000;

    CameraPose pose;
    double scale = 1.0;
    std::vector<std::vector<char>> inliers;
    RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras, opt,
                                                                 &pose, &scale, &inliers);

    log_test_message("num_inliers=" + std::to_string(stats.num_inliers) +
                     ", num_outliers=" + std::to_string(num_outliers) + ", scale=" + std::to_string(scale));
    REQUIRE(stats.num_inliers + num_outliers >= Ncam * N - 2);
    REQUIRE_SMALL(rotation_error(pose, scene.pose), 1e-2);
    REQUIRE_SMALL(translation_error(pose, scene.pose), 2e-2);
    REQUIRE_SMALL(scale - scene.scale, 2e-2);

    return true;
}

bool test_gen_absolute_pose_scale_bearings_ransac() {
    const size_t N = 30;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1200 800 800.0 800.0 600.0 400.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_pose_scale_bearings_ransac");
    const size_t num_outliers =
        add_noise_and_outliers(&scene, 0.5, 0.25, "gen_absolute_pose_scale_bearings_ransac_noise");
    const std::vector<std::vector<Point3D>> bearings = scene_bearings(scene);

    AbsolutePoseOptions opt;
    // Angular threshold matching the 2 pixel threshold of the pixel-space estimator
    opt.max_error = 2.0 / camera.focal();
    opt.ransac.min_iterations = 100;
    opt.ransac.max_iterations = 1000;

    CameraPose pose;
    double scale = 1.0;
    std::vector<std::vector<char>> inliers;
    RansacStats stats = estimate_generalized_absolute_pose_scale_bearings(bearings, scene.X, scene.camera_ext, opt,
                                                                          &pose, &scale, &inliers);

    log_test_message("num_inliers=" + std::to_string(stats.num_inliers) +
                     ", num_outliers=" + std::to_string(num_outliers) + ", scale=" + std::to_string(scale));
    REQUIRE(stats.num_inliers + num_outliers >= Ncam * N - 2);
    REQUIRE_SMALL(rotation_error(pose, scene.pose), 1e-2);
    REQUIRE_SMALL(translation_error(pose, scene.pose), 2e-2);
    REQUIRE_SMALL(scale - scene.scale, 2e-2);

    return true;
}

// A rig ten times larger and ten times smaller than the points it is registered against.
// Tolerances are on the relative scale error. The small rig spans about a hundredth of the
// depth of the points, so its centers see barely half a degree of parallax and the scale is
// by far the weakest of the seven parameters there; the large rig is the easy direction.
bool test_gen_absolute_pose_scale_large_ratio() {
    const size_t N = 40;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1200 800 800.0 800.0 600.0 400.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    const std::vector<double> scales = {10.0, 0.1};
    const std::vector<double> scale_tol = {2e-3, 3e-2};

    for (size_t case_idx = 0; case_idx < scales.size(); ++case_idx) {
        const double scale_gt = scales[case_idx];
        const std::string label = test_rng::case_id("scale", case_idx);
        log_test_case("scale", std::to_string(scale_gt));

        Scene scene =
            setup_scene(rig_extrinsics(Ncam), N, scale_gt, camera, "gen_absolute_pose_scale_large_ratio", case_idx);
        const size_t num_outliers =
            add_noise_and_outliers(&scene, 0.5, 0.2, "gen_absolute_pose_scale_large_ratio_noise");

        AbsolutePoseOptions opt;
        opt.max_error = 2.0;
        opt.ransac.min_iterations = 100;
        opt.ransac.max_iterations = 2000;

        CameraPose pose;
        double scale = 1.0;
        std::vector<std::vector<char>> inliers;
        RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras,
                                                                     opt, &pose, &scale, &inliers);

        log_test_message("num_inliers=" + std::to_string(stats.num_inliers) +
                         ", num_outliers=" + std::to_string(num_outliers) + ", scale=" + std::to_string(scale));
        REQUIRE_M(stats.num_inliers + num_outliers >= Ncam * N - 4, label);
        REQUIRE_SMALL_M(rotation_error(pose, scene.pose), 1e-2, label);
        REQUIRE_SMALL_M(translation_error(pose, scene.pose), 2e-2, label);
        REQUIRE_SMALL_M((scale - scale_gt) / scale_gt, scale_tol[case_idx], label);
    }

    return true;
}

// Half the observations replaced by random image points. gp4ps needs four correspondences
// spanning two centers, so the all inlier sample probability is well below a percent here.
bool test_gen_absolute_pose_scale_outliers() {
    const size_t N = 50;
    const size_t Ncam = 4;

    std::string camera_str = "0 PINHOLE 1200 800 800.0 800.0 600.0 400.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Scene scene = setup_scene(rig_extrinsics(Ncam), N, 2.4, camera, "gen_absolute_pose_scale_outliers");
    const size_t num_outliers = add_noise_and_outliers(&scene, 0.5, 0.5, "gen_absolute_pose_scale_outliers_noise");

    AbsolutePoseOptions opt;
    opt.max_error = 2.0;
    opt.ransac.min_iterations = 100;
    opt.ransac.max_iterations = 5000;

    CameraPose pose;
    double scale = 1.0;
    std::vector<std::vector<char>> inliers;
    RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras, opt,
                                                                 &pose, &scale, &inliers);

    log_test_message("num_inliers=" + std::to_string(stats.num_inliers) +
                     ", num_outliers=" + std::to_string(num_outliers) + ", scale=" + std::to_string(scale));
    REQUIRE(num_outliers > 2 * Ncam * N / 5);
    REQUIRE(stats.num_inliers + num_outliers >= Ncam * N - 4);
    REQUIRE_SMALL(rotation_error(pose, scene.pose), 1e-2);
    REQUIRE_SMALL(translation_error(pose, scene.pose), 2e-2);
    REQUIRE_SMALL(scale - scene.scale, 2e-2);

    return true;
}

// Runs both entry points on a rig which cannot constrain the scale and checks that no model
// is returned: zero inliers, and the pose and scale the caller passed in are left untouched.
bool degenerate_rig_returns_no_model(const Scene &scene, const AbsolutePoseOptions &opt, const std::string &name) {
    const CameraPose untouched_pose = rig_pose();
    const double untouched_scale = 0.37;

    CameraPose pose = untouched_pose;
    double scale = untouched_scale;
    std::vector<std::vector<char>> inliers;
    RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras, opt,
                                                                 &pose, &scale, &inliers);
    REQUIRE_EQ_M(stats.num_inliers, size_t(0), name);
    REQUIRE_EQ_M(scale, untouched_scale, name);
    REQUIRE_EQ_M((pose.q - untouched_pose.q).norm() + (pose.t - untouched_pose.t).norm(), 0.0, name);

    // The bearing entry point takes an angular threshold
    AbsolutePoseOptions opt_bearings = opt;
    opt_bearings.max_error = opt.max_error / scene.cameras[0].focal();
    pose = untouched_pose;
    scale = untouched_scale;
    stats = estimate_generalized_absolute_pose_scale_bearings(scene_bearings(scene), scene.X, scene.camera_ext,
                                                              opt_bearings, &pose, &scale, &inliers);
    REQUIRE_EQ_M(stats.num_inliers, size_t(0), name + " bearings");
    REQUIRE_EQ_M(scale, untouched_scale, name + " bearings");
    REQUIRE_EQ_M((pose.q - untouched_pose.q).norm() + (pose.t - untouched_pose.t).norm(), 0.0, name + " bearings");

    return true;
}

bool test_gen_absolute_pose_scale_degenerate_rig() {
    const size_t N = 30;

    std::string camera_str = "0 PINHOLE 1200 800 800.0 800.0 600.0 400.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    AbsolutePoseOptions opt;
    opt.max_error = 2.0;
    opt.ransac.min_iterations = 100;
    opt.ransac.max_iterations = 1000;

    // A single camera cannot constrain the scale: scale * p is absorbed by the translation
    REQUIRE(degenerate_rig_returns_no_model(
        setup_scene(rig_extrinsics(1), N, 2.4, camera, "gen_absolute_pose_scale_single_camera"), opt, "single camera"));

    // Neither can a rig which only rotates around a single center
    REQUIRE(degenerate_rig_returns_no_model(
        setup_scene(rotating_rig_extrinsics(3), N, 2.4, camera, "gen_absolute_pose_scale_rotating_rig"), opt,
        "rotating rig"));

    // Nor when that center lies away from the origin, so that the centers recovered from the
    // extrinsics agree only up to rounding
    REQUIRE(degenerate_rig_returns_no_model(setup_scene(rotating_rig_extrinsics(3, Eigen::Vector3d(0.3, -0.2, 0.5)), N,
                                                        2.4, camera, "gen_absolute_pose_scale_offset_rotating_rig"),
                                            opt, "offset rotating rig"));

    // Two distinct centers are enough, even if one of them holds a single observation
    {
        Scene scene = setup_scene(rig_extrinsics(2), N, 2.4, camera, "gen_absolute_pose_scale_two_cameras");
        scene.x[1].resize(1);
        scene.X[1].resize(1);
        add_multi_point_noise(scene.x, 0.5, "gen_absolute_pose_scale_two_cameras_noise");

        CameraPose pose;
        double scale = 1.0;
        std::vector<std::vector<char>> inliers;
        RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext, scene.cameras,
                                                                     opt, &pose, &scale, &inliers);

        REQUIRE(stats.num_inliers >= N);
        REQUIRE_SMALL(rotation_error(pose, scene.pose), 1e-2);
        REQUIRE_SMALL(scale - scene.scale, 5e-2);
    }

    return true;
}

bool test_draw_sample_distinct_centers() {
    // Two cameras with distinct centers, where all but one observation belong to the first
    const std::vector<size_t> num_pts_camera = {40, 1};
    const std::vector<size_t> center_group = {0, 1};
    const size_t sample_sz = 4;

    RNG_t rng = 0;
    std::vector<std::pair<size_t, size_t>> sample(sample_sz);
    for (size_t iter = 0; iter < 100; ++iter) {
        draw_sample_distinct_centers(sample_sz, num_pts_camera, center_group, &sample, rng);

        size_t num_second = 0;
        for (size_t k = 0; k < sample_sz; ++k) {
            REQUIRE(sample[k].second < num_pts_camera[sample[k].first]);
            if (center_group[sample[k].first] == 1) {
                num_second++;
            }
        }
        REQUIRE_EQ_M(num_second, size_t(1), test_rng::case_id("sample", iter));
    }

    // Cameras which share a center are grouped together
    std::vector<Point3D> camera_centers = {Point3D(0.0, 0.0, 0.0), Point3D(0.5, 0.0, 0.0), Point3D(0.0, 0.0, 0.0)};
    std::vector<size_t> groups;
    REQUIRE_EQ(group_camera_centers(camera_centers, &groups), size_t(2));
    REQUIRE_EQ(groups[0], groups[2]);
    REQUIRE(groups[0] != groups[1]);

    // So are centers which agree only up to rounding: a rig rotating about a center away from
    // the origin recovers that center from -R' * t with a few ulps of noise, which must not
    // pass for a baseline
    std::vector<Point3D> rounded_centers;
    for (const CameraPose &ext : rotating_rig_extrinsics(3, Eigen::Vector3d(0.3, -0.2, 0.5))) {
        rounded_centers.push_back(ext.center());
    }
    rounded_centers[1] += 1e-15 * Eigen::Vector3d(1.0, -1.0, 1.0);
    REQUIRE_EQ(group_camera_centers(rounded_centers, &groups), size_t(1));

    // A second center without observations leaves no sample which spans two centers, and the
    // sampler has to return an ordinary sample rather than search for one forever
    const std::vector<size_t> unobserved_second = {40, 0};
    for (size_t iter = 0; iter < 20; ++iter) {
        draw_sample_distinct_centers(sample_sz, unobserved_second, center_group, &sample, rng);
        for (size_t k = 0; k < sample_sz; ++k) {
            REQUIRE_EQ_M(sample[k].first, size_t(0), test_rng::case_id("unobserved", iter));
            REQUIRE_M(sample[k].second < unobserved_second[0], test_rng::case_id("unobserved", iter));
        }
    }

    return true;
}

} // namespace test::gen_absolute_scale

using namespace test::gen_absolute_scale;
std::vector<Test> register_gen_absolute_scale_test() {
    return {TEST(test_gen_absolute_scale_jacobian),
            TEST(test_gen_absolute_scale_weighted_jacobian),
            TEST(test_gen_absolute_scale_bearing_jacobian),
            TEST(test_gen_absolute_scale_bearing_weighted_jacobian),
            TEST(test_gen_absolute_scale_loss_gradient),
            TEST(test_gen_absolute_scale_refinement),
            TEST(test_gen_absolute_scale_refinement_exact),
            TEST(test_gen_absolute_scale_refinement_monotonic),
            TEST(test_gen_absolute_scale_matches_rigid),
            TEST(test_gen_absolute_scale_single_center_refinement),
            TEST(test_gen_absolute_pose_scale_ransac),
            TEST(test_gen_absolute_pose_scale_bearings_ransac),
            TEST(test_gen_absolute_pose_scale_large_ratio),
            TEST(test_gen_absolute_pose_scale_outliers),
            TEST(test_gen_absolute_pose_scale_degenerate_rig),
            TEST(test_draw_sample_distinct_centers)};
}
