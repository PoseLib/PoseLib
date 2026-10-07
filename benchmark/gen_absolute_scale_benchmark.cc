// Benchmark for the robust generalized absolute pose and scale estimator.
// Sweeps rig size, number of correspondences, pixel noise, outlier share and the scale ratio
// between the rig and the 3D points, and reports runtime, success rate, median errors and the
// number of RANSAC iterations for each configuration. Where the rig is already at the scale
// of the points the rigid estimate_generalized_absolute_pose is run on the same input, so the
// cost of the extra unknown is visible.
//
// Build via: cmake --build build --target gen_absolute_scale_benchmark
// Run from build/: ./benchmark/gen_absolute_scale_benchmark [trials]

#include <Eigen/Dense>
#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

using namespace poselib;

namespace {

// The rig centers sit on a ring of this diameter in the rig coordinate system, so the extent
// of the rig in the frame of the 3D points is scale * kRigDiameter for every camera count.
constexpr double kRigDiameter = 1.0;
constexpr double kMinDepth = 5.0;
constexpr double kMaxDepth = 12.0;

struct Config {
    size_t num_cams;
    size_t num_points;
    double noise_px;
    double outlier_ratio;
    double scale;
};

struct Scene {
    CameraPose pose;
    double scale;
    double rig_extent;
    std::vector<CameraPose> camera_ext;
    std::vector<Camera> cameras;
    std::vector<std::vector<Point2D>> x;
    std::vector<std::vector<Point3D>> X;
};

struct Trial {
    double runtime_ms;
    double rigid_runtime_ms;
    double rot_err_deg;
    double trans_err;
    double scale_err_rel;
    double iterations;
    bool success;
};

CameraPose random_pose(std::mt19937 &rng) {
    std::uniform_real_distribution<double> angle(-0.6, 0.6);
    std::uniform_real_distribution<double> offset(-1.5, 1.5);
    const Eigen::Matrix3d R = (Eigen::AngleAxisd(angle(rng), Eigen::Vector3d::UnitY()) *
                               Eigen::AngleAxisd(angle(rng), Eigen::Vector3d::UnitX()) *
                               Eigen::AngleAxisd(angle(rng), Eigen::Vector3d::UnitZ()))
                                  .toRotationMatrix();
    return CameraPose(R, Eigen::Vector3d(offset(rng), offset(rng), offset(rng)));
}

// Cameras spread over a ring, each looking outwards, so that no two of them share a center and
// the sample constraint of two distinct centers is satisfiable for every pair.
std::vector<CameraPose> make_rig(size_t num_cams) {
    std::vector<CameraPose> camera_ext;
    camera_ext.reserve(num_cams);
    for (size_t k = 0; k < num_cams; ++k) {
        const double theta = 2.0 * M_PI * static_cast<double>(k) / static_cast<double>(num_cams);
        const Eigen::Matrix3d R = Eigen::AngleAxisd(-theta, Eigen::Vector3d::UnitY()).toRotationMatrix();
        const Eigen::Vector3d center = 0.5 * kRigDiameter * Eigen::Vector3d(std::sin(theta), 0.0, std::cos(theta));
        camera_ext.emplace_back(R, -R * center);
    }
    return camera_ext;
}

double rig_extent(const std::vector<CameraPose> &camera_ext) {
    double extent = 0.0;
    for (size_t i = 0; i < camera_ext.size(); ++i) {
        for (size_t j = i + 1; j < camera_ext.size(); ++j) {
            extent = std::max(extent, (camera_ext[i].center() - camera_ext[j].center()).norm());
        }
    }
    return extent;
}

Scene generate_scene(const Config &config, const Camera &camera, std::mt19937 &rng) {
    Scene scene;
    scene.pose = random_pose(rng);
    scene.scale = config.scale;
    scene.camera_ext = make_rig(config.num_cams);
    scene.rig_extent = config.scale * rig_extent(scene.camera_ext);
    scene.cameras.assign(config.num_cams, camera);
    scene.x.assign(config.num_cams, {});
    scene.X.assign(config.num_cams, {});

    std::uniform_real_distribution<double> u(0.05, 0.95);
    std::uniform_real_distribution<double> depth(kMinDepth, kMaxDepth);
    std::normal_distribution<double> noise(0.0, config.noise_px);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    const ScaledCameraPose scaled_pose(scene.pose, config.scale);
    for (size_t k = 0; k < config.num_cams; ++k) {
        const CameraPose full_pose = scaled_pose.camera_pose(scene.camera_ext[k]);
        // Spread the correspondences over the rig as evenly as the count allows
        const size_t num_pts = config.num_points / config.num_cams + (k < config.num_points % config.num_cams ? 1 : 0);
        scene.x[k].reserve(num_pts);
        scene.X[k].reserve(num_pts);
        for (size_t i = 0; i < num_pts; ++i) {
            const Eigen::Vector2d xi(u(rng) * camera.width, u(rng) * camera.height);
            Eigen::Vector3d Xi;
            camera.unproject(xi, &Xi);
            Xi *= depth(rng);
            scene.X[k].push_back(full_pose.apply_inverse(Xi));

            if (unit(rng) < config.outlier_ratio) {
                scene.x[k].emplace_back(unit(rng) * camera.width, unit(rng) * camera.height);
            } else if (config.noise_px > 0.0) {
                scene.x[k].emplace_back(xi(0) + noise(rng), xi(1) + noise(rng));
            } else {
                scene.x[k].push_back(xi);
            }
        }
    }
    return scene;
}

double rotation_error_deg(const CameraPose &pose, const CameraPose &pose_gt) {
    const double c = 0.5 * ((pose.R().transpose() * pose_gt.R()).trace() - 1.0);
    return std::acos(std::max(-1.0, std::min(1.0, c))) * 180.0 / M_PI;
}

template <typename T> T median(std::vector<T> values) {
    if (values.empty()) {
        return T(0);
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double mean(const std::vector<double> &values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = 0.0;
    for (double value : values) {
        sum += value;
    }
    return sum / static_cast<double>(values.size());
}

Trial run_trial(const Scene &scene, const AbsolutePoseOptions &opt, bool run_rigid) {
    Trial trial{};

    CameraPose pose;
    double scale = 1.0;
    std::vector<std::vector<char>> inliers;

    const auto start = std::chrono::steady_clock::now();
    const RansacStats stats = estimate_generalized_absolute_pose_scale(scene.x, scene.X, scene.camera_ext,
                                                                       scene.cameras, opt, &pose, &scale, &inliers);
    const auto stop = std::chrono::steady_clock::now();
    trial.runtime_ms = std::chrono::duration<double, std::milli>(stop - start).count();

    trial.rot_err_deg = rotation_error_deg(pose, scene.pose);
    trial.trans_err = (pose.t - scene.pose.t).norm();
    trial.scale_err_rel = std::abs(scale - scene.scale) / scene.scale;
    trial.iterations = static_cast<double>(stats.iterations);
    trial.success = trial.rot_err_deg < 0.5 && trial.trans_err < 0.01 * scene.rig_extent && trial.scale_err_rel < 0.01;

    trial.rigid_runtime_ms = 0.0;
    if (run_rigid) {
        CameraPose rigid_pose;
        std::vector<std::vector<char>> rigid_inliers;
        const auto rigid_start = std::chrono::steady_clock::now();
        estimate_generalized_absolute_pose(scene.x, scene.X, scene.camera_ext, scene.cameras, opt, &rigid_pose,
                                           &rigid_inliers);
        const auto rigid_stop = std::chrono::steady_clock::now();
        trial.rigid_runtime_ms = std::chrono::duration<double, std::milli>(rigid_stop - rigid_start).count();
    }
    return trial;
}

} // namespace

int main(int argc, char *argv[]) {
    const size_t num_trials = argc > 1 ? static_cast<size_t>(std::atoi(argv[1])) : 20;

    Camera camera;
    camera.initialize_from_txt("0 PINHOLE 1200 800 800.0 800.0 600.0 400.0");

    const std::vector<size_t> camera_counts = {2, 4, 8};
    const std::vector<size_t> point_counts = {200, 2000};
    const std::vector<double> noise_levels = {0.0, 0.5, 2.0};
    const std::vector<double> outlier_ratios = {0.0, 0.3, 0.6};
    const std::vector<double> scales = {0.1, 1.0, 10.0};

    std::printf("Generalized absolute pose and scale, %zu trials per configuration\n", num_trials);
    std::printf("Rig of diameter %.1f in rig units, points at depth %.0f to %.0f, PINHOLE 1200x800 f=800\n",
                kRigDiameter, kMinDepth, kMaxDepth);
    std::printf("Inlier threshold max(1, 3*noise) px, RANSAC 100 to 10000 iterations\n");
    std::printf("success: rotation < 0.5 deg, translation < 1%% of the rig extent (scale * %.1f), scale < 1%%\n",
                kRigDiameter);
    std::printf("rigid ms: estimate_generalized_absolute_pose on the same input, where the rig is at the scale of "
                "the points\n\n");

    std::printf("%4s %6s %6s %6s %6s | %9s %9s %8s %9s %9s %9s %8s\n", "cams", "pts", "noise", "outl", "scale", "ms",
                "rigid ms", "success", "rot deg", "t %ext", "scale %", "iters");
    std::printf("---------------------------------+---------------------------------------------------------------"
                "-----------------\n");

    size_t config_index = 0;
    for (size_t num_cams : camera_counts) {
        for (size_t num_points : point_counts) {
            for (double noise_px : noise_levels) {
                for (double outlier_ratio : outlier_ratios) {
                    for (double scale : scales) {
                        const Config config{num_cams, num_points, noise_px, outlier_ratio, scale};
                        const bool run_rigid = scale == 1.0;

                        AbsolutePoseOptions opt;
                        opt.max_error = std::max(1.0, 3.0 * noise_px);
                        opt.ransac.min_iterations = 100;
                        opt.ransac.max_iterations = 10000;
                        opt.ransac.seed = 0;

                        std::vector<double> runtimes, rigid_runtimes, rot_errs, trans_errs, scale_errs, iterations;
                        size_t num_success = 0;
                        for (size_t trial_index = 0; trial_index < num_trials; ++trial_index) {
                            // Deterministic across runs and independent per configuration
                            std::mt19937 rng(static_cast<unsigned int>(1000 * config_index + trial_index + 1));
                            const Scene scene = generate_scene(config, camera, rng);
                            const Trial trial = run_trial(scene, opt, run_rigid);

                            runtimes.push_back(trial.runtime_ms);
                            rigid_runtimes.push_back(trial.rigid_runtime_ms);
                            rot_errs.push_back(trial.rot_err_deg);
                            trans_errs.push_back(100.0 * trial.trans_err / scene.rig_extent);
                            scale_errs.push_back(100.0 * trial.scale_err_rel);
                            iterations.push_back(trial.iterations);
                            num_success += trial.success ? 1 : 0;
                        }

                        char rigid_column[16];
                        if (run_rigid) {
                            std::snprintf(rigid_column, sizeof(rigid_column), "%9.2f", mean(rigid_runtimes));
                        } else {
                            std::snprintf(rigid_column, sizeof(rigid_column), "%9s", "-");
                        }

                        std::printf("%4zu %6zu %6.1f %6.1f %6.1f | %9.2f %s %7.0f%% %9.4f %9.2f %9.3f %8.0f\n",
                                    num_cams, num_points, noise_px, outlier_ratio, scale, mean(runtimes), rigid_column,
                                    100.0 * static_cast<double>(num_success) / static_cast<double>(num_trials),
                                    median(rot_errs), median(trans_errs), median(scale_errs), mean(iterations));
                        std::fflush(stdout);
                        config_index++;
                    }
                }
            }
        }
    }

    return 0;
}
