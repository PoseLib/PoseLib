#include "test.h"

#include <PoseLib/solvers/gen_relpose_6pt_42.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <vector>

using namespace poselib;

namespace test::structureless_resection {

namespace {

double rotation_error(const CameraPose &a, const CameraPose &b) {
    const Eigen::Matrix3d dR = a.R() * b.R().transpose();
    const double cos_theta = std::clamp(0.5 * (dR.trace() - 1.0), -1.0, 1.0);
    return std::acos(cos_theta);
}

double translation_error(const CameraPose &a, const CameraPose &b) { return (a.t - b.t).norm(); }

struct Semigen42Case {
    CameraPose gt_pose;
    Eigen::Vector3d p2_off;
    std::vector<Eigen::Vector3d> x1_ref;
    std::vector<Eigen::Vector3d> x2_ref;
    std::vector<Eigen::Vector3d> x1_off;
    std::vector<Eigen::Vector3d> x2_off;
};

Semigen42Case make_semigen42_case() {
    const Eigen::AngleAxisd aa_x(0.25, Eigen::Vector3d(1.0, 0.0, 0.0).normalized());
    const Eigen::AngleAxisd aa_y(-0.18, Eigen::Vector3d(0.0, 1.0, 0.0).normalized());
    const Eigen::AngleAxisd aa_z(0.12, Eigen::Vector3d(0.0, 0.0, 1.0).normalized());

    Semigen42Case data;
    data.gt_pose = CameraPose((aa_z * aa_y * aa_x).toRotationMatrix(), Eigen::Vector3d(0.18, -0.11, 0.35));
    data.p2_off = Eigen::Vector3d(0.22, -0.06, 0.04);

    const std::array<Eigen::Vector3d, 2> X_ref = {
        Eigen::Vector3d(0.25, -0.05, 4.20),
        Eigen::Vector3d(-0.18, 0.16, 3.80),
    };
    const std::array<Eigen::Vector3d, 4> X_off = {
        Eigen::Vector3d(0.32, 0.12, 4.60),
        Eigen::Vector3d(-0.28, -0.08, 4.10),
        Eigen::Vector3d(0.08, -0.20, 3.60),
        Eigen::Vector3d(-0.06, 0.24, 4.40),
    };

    data.x1_ref.reserve(X_ref.size());
    data.x2_ref.reserve(X_ref.size());
    data.x1_off.reserve(X_off.size());
    data.x2_off.reserve(X_off.size());

    for (const Eigen::Vector3d &X : X_ref) {
        const Eigen::Vector3d y_ref = data.gt_pose.apply(X);
        data.x1_ref.push_back(X.normalized());
        data.x2_ref.push_back(y_ref.normalized());
    }

    for (const Eigen::Vector3d &X : X_off) {
        const Eigen::Vector3d y_ref = data.gt_pose.apply(X);
        const Eigen::Vector3d y_off = y_ref - data.p2_off;
        data.x1_off.push_back(X.normalized());
        data.x2_off.push_back(y_off.normalized());
    }

    return data;
}

bool contains_pose_close_to_gt(const std::vector<CameraPose> &solutions, const CameraPose &gt_pose) {
    double best_rot = std::numeric_limits<double>::infinity();
    double best_trans = std::numeric_limits<double>::infinity();
    for (const CameraPose &pose : solutions) {
        best_rot = std::min(best_rot, rotation_error(pose, gt_pose));
        best_trans = std::min(best_trans, translation_error(pose, gt_pose));
        if (rotation_error(pose, gt_pose) <= 1e-5 && translation_error(pose, gt_pose) <= 1e-5) {
            return true;
        }
    }

    std::cout << "Best rotation error: " << best_rot << ", best translation error: " << best_trans << "\n";
    return false;
}

} // namespace

bool test_gen_relpose_6pt_42_reordered_direct() {
    const Semigen42Case data = make_semigen42_case();

    std::vector<CameraPose> solutions;
    const int num_solutions =
        gen_relpose_6pt_42(data.x1_off, data.x2_off, data.x1_ref, data.x2_ref, data.p2_off, &solutions);

    REQUIRE(num_solutions > 0);
    REQUIRE_EQ(num_solutions, static_cast<int>(solutions.size()));
    REQUIRE(contains_pose_close_to_gt(solutions, data.gt_pose));
    return true;
}

bool test_gen_relpose_6pt_42_reordered_benchmark_layout() {
    const Semigen42Case data = make_semigen42_case();

    std::vector<Eigen::Vector3d> x1_all = {data.x1_ref[0], data.x1_ref[1]};
    std::vector<Eigen::Vector3d> x2_all = {data.x2_ref[0], data.x2_ref[1]};
    x1_all.insert(x1_all.end(), data.x1_off.begin(), data.x1_off.end());
    x2_all.insert(x2_all.end(), data.x2_off.begin(), data.x2_off.end());

    std::vector<Eigen::Vector3d> x1_ref = {x1_all[0], x1_all[1]};
    std::vector<Eigen::Vector3d> x2_ref = {x2_all[0], x2_all[1]};
    std::vector<Eigen::Vector3d> x1_off = {x1_all[2], x1_all[3], x1_all[4], x1_all[5]};
    std::vector<Eigen::Vector3d> x2_off = {x2_all[2], x2_all[3], x2_all[4], x2_all[5]};

    std::vector<CameraPose> solutions;
    const int num_solutions = gen_relpose_6pt_42(x1_off, x2_off, x1_ref, x2_ref, data.p2_off, &solutions);

    REQUIRE(num_solutions > 0);
    REQUIRE_EQ(num_solutions, static_cast<int>(solutions.size()));
    REQUIRE(contains_pose_close_to_gt(solutions, data.gt_pose));
    return true;
}

} // namespace test::structureless_resection

std::vector<Test> register_structureless_resection_test() {
    return {
        TEST(test::structureless_resection::test_gen_relpose_6pt_42_reordered_direct),
        TEST(test::structureless_resection::test_gen_relpose_6pt_42_reordered_benchmark_layout),
    };
}
