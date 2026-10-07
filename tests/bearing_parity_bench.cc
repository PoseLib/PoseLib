// Standalone bearing-vector vs pinhole parity benchmark for PR #206 review.
// Drives both estimate_absolute_pose (pixel) and estimate_absolute_pose_bearings
// (bearing) on synthetic pinhole scenes — and the two relative-pose paths
// likewise — and reports per-scene rotation/translation/runtime deltas.
//
// Build via: cmake --build build --target bearing_parity_bench
// Run from build/: ./tests/bearing_parity_bench

#include <Eigen/Dense>
#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace poselib;

namespace {

CameraPose random_pose(std::mt19937 &rng) {
    std::uniform_real_distribution<double> a(-0.5, 0.5);
    const Eigen::Matrix3d R =
        (Eigen::AngleAxisd(a(rng), Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(a(rng), Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(a(rng), Eigen::Vector3d::UnitZ()))
            .toRotationMatrix();
    Eigen::Vector3d t(a(rng), a(rng), a(rng));
    return CameraPose(R, t);
}

double rot_err_rad(const CameraPose &a, const CameraPose &b) {
    const Eigen::Quaterniond qa(a.q(0), a.q(1), a.q(2), a.q(3));
    const Eigen::Quaterniond qb(b.q(0), b.q(1), b.q(2), b.q(3));
    const double w = std::abs((qa * qb.conjugate()).w());
    return 2.0 * std::acos(std::min(1.0, w));
}

void run_absolute_scene(int trial, std::mt19937 &rng, double &px_rot, double &px_t, double &b_rot, double &b_t,
                        double &px_ms, double &b_ms) {
    const size_t N = 100;
    const double focal = 800.0;
    const double noise_px = 0.5;

    CameraPose pose_gt = random_pose(rng);
    const Eigen::Matrix3d R = pose_gt.R();

    std::uniform_real_distribution<double> ud(-3.0, 3.0);
    std::normal_distribution<double> nd(0.0, noise_px);

    std::vector<Point2D> pix;
    std::vector<Point3D> pts;
    std::vector<Point3D> bearings;
    pix.reserve(N);
    pts.reserve(N);
    bearings.reserve(N);
    while (pix.size() < N) {
        Eigen::Vector3d X(ud(rng), ud(rng), ud(rng) + 7.0);
        const Eigen::Vector3d Z = R * X + pose_gt.t;
        if (Z.z() <= 0.5)
            continue;
        const double xn = Z.x() / Z.z() + nd(rng) / focal;
        const double yn = Z.y() / Z.z() + nd(rng) / focal;
        pix.emplace_back(focal * xn, focal * yn);
        pts.push_back(X);
        bearings.push_back(Eigen::Vector3d(xn, yn, 1.0).normalized());
    }

    // Pinhole path.
    Image image_px;
    image_px.camera.model_id = CameraModelId::SIMPLE_PINHOLE;
    image_px.camera.params = {focal, 0.0, 0.0};
    AbsolutePoseOptions opt_px;
    opt_px.max_error = 2.0; // pixels
    opt_px.ransac.max_iterations = 1000;
    opt_px.ransac.success_prob = 0.999;
    std::vector<char> in_px;
    auto t0 = std::chrono::high_resolution_clock::now();
    estimate_absolute_pose(pix, pts, opt_px, &image_px, &in_px);
    auto t1 = std::chrono::high_resolution_clock::now();
    px_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    px_rot = rot_err_rad(image_px.pose, pose_gt);
    px_t = (image_px.pose.t - pose_gt.t).norm();

    // Bearing path.
    AbsolutePoseOptions opt_b;
    opt_b.max_error = 2.0 / focal; // ≈ angular threshold in radians
    opt_b.ransac.max_iterations = 1000;
    opt_b.ransac.success_prob = 0.999;
    CameraPose pose_b;
    std::vector<char> in_b;
    t0 = std::chrono::high_resolution_clock::now();
    estimate_absolute_pose_bearings(bearings, pts, opt_b, &pose_b, &in_b);
    t1 = std::chrono::high_resolution_clock::now();
    b_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    b_rot = rot_err_rad(pose_b, pose_gt);
    b_t = (pose_b.t - pose_gt.t).norm();
    (void)trial;
}

void run_relative_scene(int trial, std::mt19937 &rng, double &px_rot, double &px_t, double &b_rot, double &b_t,
                        double &px_ms, double &b_ms) {
    const size_t N = 80;
    const double focal = 800.0;
    const double noise_px = 0.5;

    CameraPose pose_gt = random_pose(rng);
    pose_gt.t.normalize(); // unit translation
    const Eigen::Matrix3d R = pose_gt.R();

    std::uniform_real_distribution<double> ud(-3.0, 3.0);
    std::normal_distribution<double> nd(0.0, noise_px);

    std::vector<Point2D> x1, x2;
    std::vector<Point3D> b1, b2;
    while (x1.size() < N) {
        Eigen::Vector3d X(ud(rng), ud(rng), ud(rng) + 7.0);
        const Eigen::Vector3d Z1 = X;
        const Eigen::Vector3d Z2 = R * X + pose_gt.t;
        if (Z1.z() <= 0.5 || Z2.z() <= 0.5)
            continue;
        const double xn1 = Z1.x() / Z1.z() + nd(rng) / focal;
        const double yn1 = Z1.y() / Z1.z() + nd(rng) / focal;
        const double xn2 = Z2.x() / Z2.z() + nd(rng) / focal;
        const double yn2 = Z2.y() / Z2.z() + nd(rng) / focal;
        x1.emplace_back(focal * xn1, focal * yn1);
        x2.emplace_back(focal * xn2, focal * yn2);
        b1.push_back(Eigen::Vector3d(xn1, yn1, 1.0).normalized());
        b2.push_back(Eigen::Vector3d(xn2, yn2, 1.0).normalized());
    }

    Camera cam1;
    cam1.model_id = CameraModelId::SIMPLE_PINHOLE;
    cam1.params = {focal, 0.0, 0.0};
    Camera cam2 = cam1;

    RelativePoseOptions opt_px;
    opt_px.max_error = 2.0; // pixels
    opt_px.ransac.max_iterations = 1000;
    opt_px.ransac.success_prob = 0.999;
    CameraPose pose_px;
    std::vector<char> in_px;
    auto t0 = std::chrono::high_resolution_clock::now();
    estimate_relative_pose(x1, x2, cam1, cam2, opt_px, &pose_px, &in_px);
    auto t1 = std::chrono::high_resolution_clock::now();
    px_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    px_rot = rot_err_rad(pose_px, pose_gt);
    px_t = std::acos(std::min(1.0, std::abs(pose_px.t.normalized().dot(pose_gt.t.normalized()))));

    RelativePoseOptions opt_b;
    opt_b.max_error = 2.0 / focal; // angular threshold in radians
    opt_b.ransac.max_iterations = 1000;
    opt_b.ransac.success_prob = 0.999;
    CameraPose pose_b;
    std::vector<char> in_b;
    t0 = std::chrono::high_resolution_clock::now();
    estimate_relative_pose_bearings(b1, b2, opt_b, &pose_b, &in_b);
    t1 = std::chrono::high_resolution_clock::now();
    b_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    b_rot = rot_err_rad(pose_b, pose_gt);
    b_t = std::acos(std::min(1.0, std::abs(pose_b.t.normalized().dot(pose_gt.t.normalized()))));
    (void)trial;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

double mean(const std::vector<double> &v) {
    double s = 0.0;
    for (double x : v)
        s += x;
    return s / v.size();
}

} // namespace

int main() {
    const int trials = 100;
    std::mt19937 rng(42);

    std::vector<double> abs_px_rot, abs_px_t, abs_b_rot, abs_b_t, abs_px_ms, abs_b_ms;
    std::vector<double> rel_px_rot, rel_px_t, rel_b_rot, rel_b_t, rel_px_ms, rel_b_ms;

    for (int i = 0; i < trials; ++i) {
        double pr, pt, br, bt, pm, bm;
        run_absolute_scene(i, rng, pr, pt, br, bt, pm, bm);
        abs_px_rot.push_back(pr);
        abs_px_t.push_back(pt);
        abs_b_rot.push_back(br);
        abs_b_t.push_back(bt);
        abs_px_ms.push_back(pm);
        abs_b_ms.push_back(bm);

        run_relative_scene(i, rng, pr, pt, br, bt, pm, bm);
        rel_px_rot.push_back(pr);
        rel_px_t.push_back(pt);
        rel_b_rot.push_back(br);
        rel_b_t.push_back(bt);
        rel_px_ms.push_back(pm);
        rel_b_ms.push_back(bm);
    }

    std::printf("Bearing-parity benchmark — %d trials, pinhole f=800, noise=0.5px\n\n", trials);
    std::printf("Absolute pose (PnP):\n");
    std::printf("  pinhole path : med rot %.2e rad | med t %.2e | mean rt %.2f ms\n", median(abs_px_rot),
                median(abs_px_t), mean(abs_px_ms));
    std::printf("  bearing path : med rot %.2e rad | med t %.2e | mean rt %.2f ms\n", median(abs_b_rot),
                median(abs_b_t), mean(abs_b_ms));

    std::printf("\nRelative pose (5pt):\n");
    std::printf("  pinhole path : med rot %.2e rad | med t-angle %.2e rad | mean rt %.2f ms\n", median(rel_px_rot),
                median(rel_px_t), mean(rel_px_ms));
    std::printf("  bearing path : med rot %.2e rad | med t-angle %.2e rad | mean rt %.2f ms\n", median(rel_b_rot),
                median(rel_b_t), mean(rel_b_ms));

    return 0;
}
