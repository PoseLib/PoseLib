// Copyright (c) 2026, OpenMVS contributors
// Tests for the bearing-vector pose API (BearingAbsolutePoseRefiner,
// BearingRelativePoseRefiner, BearingAbsolutePoseEstimator,
// BearingRelativePoseEstimator, estimate_*_bearings entry points).
//
// The analytical Jacobians of the new refiners are validated against central
// finite differences. The full RANSAC + LM pipeline is exercised on
// full-sphere synthetic scenes with observations in both hemispheres — the
// regime where the existing 2D-Point2D pipeline breaks down because it cannot
// distinguish front- from back-hemisphere bearings.

#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/constants.h>
#include <PoseLib/robust.h>
#include <PoseLib/robust/bundle.h>
#include <PoseLib/robust/estimators/absolute_pose.h>
#include <PoseLib/robust/estimators/relative_pose.h>
#include <PoseLib/robust/optim/absolute.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/lm_impl.h>
#include <PoseLib/robust/optim/relative.h>
#include <PoseLib/robust/robust_loss.h>
#include <PoseLib/robust/utils.h>
#include <PoseLib/types.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace poselib;

namespace test::bearing {

namespace {

// Fixed reference pose — same across all test runs so finite-difference
// checks happen at the same point.
CameraPose reference_pose() {
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(0.28, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(-0.14, Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(0.09, Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();
    return CameraPose(R, Eigen::Vector3d(0.2, -0.15, 0.35));
}

// Build a full-sphere synthetic scene: N 3D points distributed uniformly
// on a sphere of radius ~5 around origin, so roughly half of them are
// behind the camera. For each point we compute the bearing = normalized
// direction from camera center to the point.
void build_full_sphere_scene(size_t N, CameraPose &pose, std::vector<Point3D> &bearings, std::vector<Point3D> &X,
                             const std::string &case_name = "bearing_scene", size_t case_index = 0) {
    pose = reference_pose();
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    bearings.clear();
    X.clear();
    bearings.reserve(N);
    X.reserve(N);

    const Eigen::Matrix3d R = pose.R();
    const Eigen::Vector3d &t = pose.t;

    for (size_t i = 0; i < N; ++i) {
        // Uniform sphere sampling by cos(theta) + phi; radius ~5.
        const double u = rng.uniform(-1.0, 1.0);
        const double phi = rng.uniform(-M_PI, M_PI);
        const double r = 4.5 + rng.uniform(0.0, 1.0);
        const double s = std::sqrt(std::max(0.0, 1.0 - u * u));
        const Eigen::Vector3d Xw(r * s * std::cos(phi), r * u, r * s * std::sin(phi));

        // Bearing in camera space = normalized(R * Xw + t)
        const Eigen::Vector3d Z = R * Xw + t;
        const double nZ = Z.norm();
        if (nZ < 1e-9)
            continue;
        bearings.push_back(Z / nZ);
        X.push_back(Xw);
    }
}

// Build a two-view full-sphere scene: two cameras with a small baseline,
// shared 3D points on a sphere around origin, bearings computed in each
// camera's frame.
void build_two_view_scene(size_t N, CameraPose &pose_rel, std::vector<Point3D> &b1, std::vector<Point3D> &b2,
                          const std::string &case_name = "bearing_two_view", size_t case_index = 0) {
    // Pose 1: identity. Pose 2: small rotation + translation.
    const CameraPose pose1;
    const CameraPose pose2 = reference_pose();
    // pose_rel transforms points from camera 1 space to camera 2 space.
    // i.e. X_cam2 = pose_rel.R * X_cam1 + pose_rel.t
    pose_rel = pose2;

    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    b1.clear();
    b2.clear();
    b1.reserve(N);
    b2.reserve(N);

    for (size_t i = 0; i < N; ++i) {
        const double u = rng.uniform(-1.0, 1.0);
        const double phi = rng.uniform(-M_PI, M_PI);
        const double r = 4.5 + rng.uniform(0.0, 1.0);
        const double s = std::sqrt(std::max(0.0, 1.0 - u * u));
        const Eigen::Vector3d X(r * s * std::cos(phi), r * u, r * s * std::sin(phi));

        const Eigen::Vector3d Z1 = pose1.R() * X + pose1.t;
        const Eigen::Vector3d Z2 = pose_rel.R() * X + pose_rel.t;
        const double n1 = Z1.norm();
        const double n2 = Z2.norm();
        if (n1 < 1e-9 || n2 < 1e-9)
            continue;
        b1.push_back(Z1 / n1);
        b2.push_back(Z2 / n2);
    }
}

// Add deterministic noise to a bearing list (small tangent-space rotation
// per observation). Keeps the result a unit vector.
void add_bearing_noise(std::vector<Point3D> &b, double angle_scale, const std::string &case_name,
                       size_t case_index = 0) {
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    for (Point3D &bi : b) {
        // Random axis perpendicular to bi.
        Eigen::Vector3d axis = test_rng::symmetric_vec3(rng, 1.0);
        axis -= axis.dot(bi) * bi;
        if (axis.norm() < 1e-6)
            continue;
        axis.normalize();
        const double angle = rng.uniform(-angle_scale, angle_scale);
        const Eigen::AngleAxisd aa(angle, axis);
        bi = aa * bi;
        bi.normalize();
    }
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// Scoring primitives

bool test_bearing_msac_score_zero_at_gt() {
    const size_t N = 20;
    CameraPose pose;
    std::vector<Point3D> bearings, X;
    build_full_sphere_scene(N, pose, bearings, X);

    size_t inliers = 0;
    const double sq_threshold = 1e-6;
    double score = compute_msac_score_bearing(pose, bearings, X, sq_threshold, &inliers);
    // At ground truth the chord distance is zero for every point, so:
    //   - every point is counted as an inlier
    //   - score is 0 (because per-point residual is 0)
    REQUIRE_EQ(inliers, N);
    REQUIRE_SMALL(score, 1e-9);
    return true;
}

bool test_bearing_sampson_score_zero_at_gt() {
    const size_t N = 20;
    CameraPose pose_rel;
    std::vector<Point3D> b1, b2;
    build_two_view_scene(N, pose_rel, b1, b2);

    size_t inliers = 0;
    const double sq_threshold = 1e-6;
    // Cheirality on — this is the production default and is bearing-native.
    // At ground truth all points are cheiral, so the score should still be 0
    // with all N points counted as inliers.
    double score = compute_sampson_msac_score_bearing(pose_rel, b1, b2, sq_threshold, &inliers,
                                                      /*check_cheirality_flag=*/true);
    REQUIRE_EQ(inliers, N);
    REQUIRE_SMALL(score, 1e-9);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Refiner Jacobian validation (central finite differences)

bool test_bearing_absolute_pose_normal_acc() {
    const size_t N = 15;
    CameraPose pose;
    std::vector<Point3D> bearings, X;
    build_full_sphere_scene(N, pose, bearings, X);

    NormalAccumulator acc;
    BearingAbsolutePoseRefiner refiner(bearings, X);
    acc.initialize(refiner.num_params);

    // At ground truth: residual = 0, gradient = 0.
    acc.reset_residual();
    refiner.compute_residual(acc, pose);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);
    return true;
}

bool test_bearing_absolute_pose_jacobian() {
    const size_t N = 12;
    CameraPose pose;
    std::vector<Point3D> bearings, X;
    build_full_sphere_scene(N, pose, bearings, X);

    // Verify the roughly half back-hemisphere coverage so we know the
    // Jacobian check is actually exercising the sign(z) < 0 regime.
    size_t back_count = 0;
    const Eigen::Matrix3d Rgt = pose.R();
    for (const Point3D &Xw : X) {
        const Eigen::Vector3d Z = Rgt * Xw + pose.t;
        if (Z.z() < 0.0)
            ++back_count;
    }
    log_test_case("back_hemisphere_fraction",
                  std::to_string(static_cast<double>(back_count) / static_cast<double>(X.size())));

    add_bearing_noise(bearings, 1e-3, "bearing_absolute_pose_jacobian_noise");

    std::vector<double> weights(bearings.size(), 1.0);
    BearingAbsolutePoseRefiner<std::vector<double>, TestAccumulator> refiner(bearings, X, weights);

    const double delta = 1e-7;
    double jac_err = verify_jacobian<decltype(refiner), CameraPose>(refiner, pose, delta);
    REQUIRE_SMALL(jac_err, 1e-5);

    // Consistency check: residual via compute_residual equals sum of
    // squared residuals from compute_jacobian's accumulator.
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, pose);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose);
    double r2 = 0.0;
    for (size_t i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);
    return true;
}

bool test_bearing_relative_pose_normal_acc() {
    const size_t N = 15;
    CameraPose pose_rel;
    std::vector<Point3D> b1, b2;
    build_two_view_scene(N, pose_rel, b1, b2);

    NormalAccumulator acc;
    BearingRelativePoseRefiner refiner(b1, b2);
    acc.initialize(refiner.num_params);

    acc.reset_residual();
    refiner.compute_residual(acc, pose_rel);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-9);

    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose_rel);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-9);
    return true;
}

bool test_bearing_relative_pose_jacobian() {
    const size_t N = 12;
    CameraPose pose_rel;
    std::vector<Point3D> b1, b2;
    build_two_view_scene(N, pose_rel, b1, b2);

    add_bearing_noise(b1, 1e-3, "bearing_relative_pose_jacobian_noise_b1");
    add_bearing_noise(b2, 1e-3, "bearing_relative_pose_jacobian_noise_b2");

    std::vector<double> weights(b1.size(), 1.0);
    BearingRelativePoseRefiner<std::vector<double>, TestAccumulator> refiner(b1, b2, weights);

    const double delta = 1e-7;
    double jac_err = verify_jacobian<decltype(refiner), CameraPose>(refiner, pose_rel, delta);
    REQUIRE_SMALL(jac_err, 1e-5);

    // Residual / jacobian consistency
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, pose_rel);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose_rel);
    double r2 = 0.0;
    for (size_t i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// LM convergence (single bundle_adjust_bearing / refine_relpose_bearing call)

bool test_bearing_absolute_pose_refinement() {
    const size_t N = 40;
    CameraPose pose_gt;
    std::vector<Point3D> bearings, X;
    build_full_sphere_scene(N, pose_gt, bearings, X, "bearing_abs_refinement");

    // Start from a slightly perturbed pose and recover ground truth.
    CameraPose pose = pose_gt;
    pose.q.x() += 0.03;
    pose.q.y() -= 0.02;
    pose.q.z() += 0.015;
    pose.q.normalize();
    pose.t += Eigen::Vector3d(0.05, -0.04, 0.06);

    BundleOptions opt;
    opt.max_iterations = 50;
    opt.gradient_tol = 1e-12;
    opt.loss_type = BundleOptions::TRIVIAL;

    BundleStats stats = bundle_adjust_bearing(bearings, X, &pose, opt);
    log_bundle_stats(stats, "bearing_absolute_pose_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "bearing_absolute_pose_refinement"));

    // Verify that we converged back to the ground truth.
    const Eigen::Quaterniond q_err =
        Eigen::Quaterniond(pose.q(0), pose.q(1), pose.q(2), pose.q(3)) *
        Eigen::Quaterniond(pose_gt.q(0), pose_gt.q(1), pose_gt.q(2), pose_gt.q(3)).conjugate();
    const double angle_err = 2.0 * std::acos(std::clamp(std::abs(q_err.w()), -1.0, 1.0));
    REQUIRE_SMALL(angle_err, 1e-6);
    REQUIRE_SMALL((pose.t - pose_gt.t).norm(), 1e-6);
    return true;
}

bool test_bearing_relative_pose_refinement() {
    const size_t N = 40;
    CameraPose pose_gt;
    std::vector<Point3D> b1, b2;
    build_two_view_scene(N, pose_gt, b1, b2, "bearing_rel_refinement");

    CameraPose pose = pose_gt;
    pose.q.x() += 0.02;
    pose.q.y() -= 0.015;
    pose.q.normalize();
    pose.t += Eigen::Vector3d(0.03, 0.02, -0.04);

    BundleOptions opt;
    opt.max_iterations = 50;
    opt.gradient_tol = 1e-12;
    opt.loss_type = BundleOptions::TRIVIAL;

    BundleStats stats = refine_relpose_bearing(b1, b2, &pose, opt);
    log_bundle_stats(stats, "bearing_relative_pose_refinement");
    REQUIRE(check_bundle_cost_and_gradient(stats, 1e-6, "bearing_relative_pose_refinement"));

    // Rotation should recover exactly; translation is up to scale, so
    // compare direction only.
    const Eigen::Quaterniond q_err =
        Eigen::Quaterniond(pose.q(0), pose.q(1), pose.q(2), pose.q(3)) *
        Eigen::Quaterniond(pose_gt.q(0), pose_gt.q(1), pose_gt.q(2), pose_gt.q(3)).conjugate();
    const double angle_err = 2.0 * std::acos(std::clamp(std::abs(q_err.w()), -1.0, 1.0));
    REQUIRE_SMALL(angle_err, 1e-5);
    const double t_sim = pose.t.normalized().dot(pose_gt.t.normalized());
    REQUIRE(t_sim > 1.0 - 1e-6);
    return true;
}

////////////////////////////////////////////////////////////////////////////////
// Full RANSAC + LM pipeline via estimate_*_bearings

bool test_estimate_absolute_pose_bearings() {
    const size_t N = 80;
    CameraPose pose_gt;
    std::vector<Point3D> bearings, X;
    build_full_sphere_scene(N, pose_gt, bearings, X, "estimate_abs_bearings");

    AbsolutePoseOptions opt;
    opt.max_error = 1e-4; // angular threshold in radians (very tight; no noise)
    opt.ransac.max_iterations = 1000;
    opt.ransac.min_iterations = 100;
    opt.ransac.success_prob = 0.999;

    CameraPose pose;
    std::vector<char> inliers;
    RansacStats stats = estimate_absolute_pose_bearings(bearings, X, opt, &pose, &inliers);

    log_test_case("num_inliers", std::to_string(stats.num_inliers));
    REQUIRE(stats.num_inliers == N);

    // Verify recovered pose matches ground truth.
    const Eigen::Quaterniond q_err =
        Eigen::Quaterniond(pose.q(0), pose.q(1), pose.q(2), pose.q(3)) *
        Eigen::Quaterniond(pose_gt.q(0), pose_gt.q(1), pose_gt.q(2), pose_gt.q(3)).conjugate();
    const double angle_err = 2.0 * std::acos(std::clamp(std::abs(q_err.w()), -1.0, 1.0));
    REQUIRE_SMALL(angle_err, 1e-4);
    REQUIRE_SMALL((pose.t - pose_gt.t).norm(), 1e-4);
    return true;
}

bool test_estimate_relative_pose_bearings() {
    const size_t N = 80;
    CameraPose pose_gt;
    std::vector<Point3D> b1, b2;
    build_two_view_scene(N, pose_gt, b1, b2, "estimate_rel_bearings");

    RelativePoseOptions opt;
    opt.max_error = 1e-4; // angular threshold in radians (very tight; no noise)
    opt.ransac.max_iterations = 1000;
    opt.ransac.min_iterations = 100;
    opt.ransac.success_prob = 0.999;

    CameraPose pose;
    std::vector<char> inliers;
    // Cheirality enabled via the default (it's necessary to disambiguate the
    // four essential-matrix decompositions). Bearing-native cheirality works
    // even though our scene contains back-hemisphere points.
    RansacStats stats = estimate_relative_pose_bearings(b1, b2, opt, &pose, &inliers);

    log_test_case("num_inliers", std::to_string(stats.num_inliers));
    REQUIRE(stats.num_inliers >= N - 5); // Allow a few 5pt-solver outliers

    // Recovered rotation should match; translation up to scale.
    const Eigen::Quaterniond q_err =
        Eigen::Quaterniond(pose.q(0), pose.q(1), pose.q(2), pose.q(3)) *
        Eigen::Quaterniond(pose_gt.q(0), pose_gt.q(1), pose_gt.q(2), pose_gt.q(3)).conjugate();
    const double angle_err = 2.0 * std::acos(std::clamp(std::abs(q_err.w()), -1.0, 1.0));
    REQUIRE_SMALL(angle_err, 1e-3);
    const double t_sim = pose.t.normalized().dot(pose_gt.t.normalized());
    REQUIRE(t_sim > 1.0 - 1e-4);
    return true;
}

// Pinhole-as-bearing parity benchmark (V5 of PR #206 review).
// Build a synthetic pinhole scene (all points in front of the camera, finite
// z); estimate the pose with both the pixel-based path and the bearing path
// (where bearings = normalize((x_norm, y_norm, 1))). The two estimators should
// recover poses that agree within tight numerical tolerance — the standard
// "show the bearing path is correct on pinhole" test that Viktor asked for.
//
// We synthesize a pinhole-only scene (positive-z only) using build_two_view_scene
// but constrained to one hemisphere via a derived helper.
namespace {

void build_pinhole_scene(size_t N, CameraPose &pose, std::vector<Point2D> &x1, std::vector<Point2D> &x2,
                         std::vector<Point3D> &b1, std::vector<Point3D> &b2,
                         const std::string &case_name = "pinhole_parity", size_t case_index = 0) {
    pose = reference_pose();
    test_rng::Rng rng = test_rng::make_rng(case_name, case_index);
    x1.clear();
    x2.clear();
    b1.clear();
    b2.clear();
    x1.reserve(N);
    x2.reserve(N);
    b1.reserve(N);
    b2.reserve(N);

    const Eigen::Matrix3d R = pose.R();
    while (b1.size() < N) {
        // Sample a 3D point in front of camera 1 (z > 0 after no-op identity pose
        // in image 1) and verify it is also in front of camera 2.
        const Eigen::Vector3d X = test_rng::symmetric_vec3(rng, 5.0) + Eigen::Vector3d(0.0, 0.0, 8.0);
        const Eigen::Vector3d Z1 = X;
        const Eigen::Vector3d Z2 = R * X + pose.t;
        if (Z1.z() <= 0.5 || Z2.z() <= 0.5)
            continue;

        const Eigen::Vector2d p1(Z1.x() / Z1.z(), Z1.y() / Z1.z());
        const Eigen::Vector2d p2(Z2.x() / Z2.z(), Z2.y() / Z2.z());
        x1.push_back(p1);
        x2.push_back(p2);
        b1.push_back(Eigen::Vector3d(p1.x(), p1.y(), 1.0).normalized());
        b2.push_back(Eigen::Vector3d(p2.x(), p2.y(), 1.0).normalized());
    }
}

} // namespace

bool test_pinhole_bearing_parity_absolute() {
    // Pinhole 3D points + their unit-bearing equivalents must produce the same
    // recovered pose to within numerical tolerance.
    const size_t N = 100;
    CameraPose pose_gt = reference_pose();
    std::vector<Point2D> p2d;
    std::vector<Point3D> p3d;
    std::vector<Point3D> bearings;
    test_rng::Rng rng = test_rng::make_rng("pinhole_parity_abs", 0);
    p2d.reserve(N);
    p3d.reserve(N);
    bearings.reserve(N);

    const Eigen::Matrix3d R = pose_gt.R();
    while (bearings.size() < N) {
        const Eigen::Vector3d X = test_rng::symmetric_vec3(rng, 5.0) + Eigen::Vector3d(0.0, 0.0, 8.0);
        const Eigen::Vector3d Z = R * X + pose_gt.t;
        if (Z.z() <= 0.5)
            continue;
        p2d.emplace_back(Z.x() / Z.z(), Z.y() / Z.z());
        p3d.push_back(X);
        bearings.push_back(Z.normalized());
    }

    // Bearing path
    AbsolutePoseOptions opt;
    opt.max_error = 1e-4; // radians
    opt.ransac.max_iterations = 1000;
    opt.ransac.min_iterations = 100;
    opt.ransac.success_prob = 0.999;
    CameraPose pose_bearing;
    std::vector<char> inliers_b;
    RansacStats stats_b = estimate_absolute_pose_bearings(bearings, p3d, opt, &pose_bearing, &inliers_b);

    // Both paths should converge to the ground truth with very small error.
    REQUIRE(stats_b.num_inliers == N);
    const Eigen::Quaterniond q_err =
        Eigen::Quaterniond(pose_bearing.q(0), pose_bearing.q(1), pose_bearing.q(2), pose_bearing.q(3)) *
        Eigen::Quaterniond(pose_gt.q(0), pose_gt.q(1), pose_gt.q(2), pose_gt.q(3)).conjugate();
    const double angle_err = 2.0 * std::acos(std::clamp(std::abs(q_err.w()), -1.0, 1.0));
    log_test_case("bearing_rot_err_rad", std::to_string(angle_err));
    log_test_case("bearing_t_err", std::to_string((pose_bearing.t - pose_gt.t).norm()));
    REQUIRE_SMALL(angle_err, 1e-5);
    REQUIRE_SMALL((pose_bearing.t - pose_gt.t).norm(), 1e-5);
    return true;
}

bool test_pinhole_bearing_parity_relative() {
    // Pinhole 2D points + their unit-bearing equivalents must produce the same
    // recovered relative pose.
    const size_t N = 80;
    CameraPose pose_gt;
    std::vector<Point2D> x1, x2;
    std::vector<Point3D> b1, b2;
    build_pinhole_scene(N, pose_gt, x1, x2, b1, b2, "pinhole_parity_rel", 0);

    // Bearing path
    RelativePoseOptions opt;
    opt.max_error = 1e-4; // radians
    opt.ransac.max_iterations = 1000;
    opt.ransac.min_iterations = 100;
    opt.ransac.success_prob = 0.999;
    CameraPose pose_b;
    std::vector<char> inliers_b;
    RansacStats stats_b = estimate_relative_pose_bearings(b1, b2, opt, &pose_b, &inliers_b);

    REQUIRE(stats_b.num_inliers >= N - 5);

    const Eigen::Quaterniond q_err =
        Eigen::Quaterniond(pose_b.q(0), pose_b.q(1), pose_b.q(2), pose_b.q(3)) *
        Eigen::Quaterniond(pose_gt.q(0), pose_gt.q(1), pose_gt.q(2), pose_gt.q(3)).conjugate();
    const double angle_err = 2.0 * std::acos(std::clamp(std::abs(q_err.w()), -1.0, 1.0));
    const double t_sim = pose_b.t.normalized().dot(pose_gt.t.normalized());
    log_test_case("bearing_rot_err_rad", std::to_string(angle_err));
    log_test_case("bearing_t_cos_sim", std::to_string(t_sim));
    REQUIRE_SMALL(angle_err, 1e-4);
    REQUIRE(t_sim > 1.0 - 1e-6);
    return true;
}

} // namespace test::bearing

using namespace test::bearing;
std::vector<Test> register_optim_bearing_test() {
    return {
        TEST(test_bearing_msac_score_zero_at_gt),    TEST(test_bearing_sampson_score_zero_at_gt),
        TEST(test_bearing_absolute_pose_normal_acc), TEST(test_bearing_absolute_pose_jacobian),
        TEST(test_bearing_relative_pose_normal_acc), TEST(test_bearing_relative_pose_jacobian),
        TEST(test_bearing_absolute_pose_refinement), TEST(test_bearing_relative_pose_refinement),
        TEST(test_estimate_absolute_pose_bearings),  TEST(test_estimate_relative_pose_bearings),
        TEST(test_pinhole_bearing_parity_absolute),  TEST(test_pinhole_bearing_parity_relative),
    };
}
