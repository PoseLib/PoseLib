#include "test.h"

#include <PoseLib/solvers/gen_relpose_6pt_51.h>
#include <PoseLib/solvers/gen_relpose_6pt.h>
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

struct StructurelessTwoRefCase {
    CameraPose pose_query;
    CameraPose pose_ref1;
    CameraPose pose_ref2;
    std::vector<Eigen::Vector3d> x_query1;
    std::vector<Eigen::Vector3d> x_ref1;
    std::vector<Eigen::Vector3d> x_query2;
    std::vector<Eigen::Vector3d> x_ref2;
};

struct Structureless6ptCase {
    CameraPose pose_query;
    std::vector<CameraPose> pose_ref;
    std::vector<Eigen::Vector3d> x_query;
    std::vector<Eigen::Vector3d> x_ref;
};

CameraPose make_pose_from_center(const Eigen::AngleAxisd &aa_x, const Eigen::AngleAxisd &aa_y, const Eigen::AngleAxisd &aa_z,
                                 const Eigen::Vector3d &center) {
    const Eigen::Matrix3d R = (aa_z * aa_y * aa_x).toRotationMatrix();
    return CameraPose(R, -R * center);
}

Eigen::Vector3d bearing_from_pose(const CameraPose &pose, const Eigen::Vector3d &X) { return pose.apply(X).normalized(); }

CameraPose scale_pose_translation(const CameraPose &pose, double scale) { return CameraPose(pose.q, scale * pose.t); }

std::vector<CameraPose> scale_pose_translations(const std::vector<CameraPose> &poses, double scale) {
    std::vector<CameraPose> scaled;
    scaled.reserve(poses.size());
    for (const CameraPose &pose : poses) {
        scaled.push_back(scale_pose_translation(pose, scale));
    }
    return scaled;
}

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

StructurelessTwoRefCase make_structureless_42_case() {
    StructurelessTwoRefCase data;
    data.pose_query = make_pose_from_center(Eigen::AngleAxisd(0.20, Eigen::Vector3d::UnitX()),
                                            Eigen::AngleAxisd(-0.14, Eigen::Vector3d::UnitY()),
                                            Eigen::AngleAxisd(0.11, Eigen::Vector3d::UnitZ()),
                                            Eigen::Vector3d(0.10, -0.08, 0.05));
    data.pose_ref1 = make_pose_from_center(Eigen::AngleAxisd(-0.06, Eigen::Vector3d::UnitX()),
                                           Eigen::AngleAxisd(0.17, Eigen::Vector3d::UnitY()),
                                           Eigen::AngleAxisd(-0.09, Eigen::Vector3d::UnitZ()),
                                           Eigen::Vector3d(-0.18, 0.07, 0.02));
    data.pose_ref2 = make_pose_from_center(Eigen::AngleAxisd(0.08, Eigen::Vector3d::UnitX()),
                                           Eigen::AngleAxisd(-0.10, Eigen::Vector3d::UnitY()),
                                           Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitZ()),
                                           Eigen::Vector3d(0.22, 0.03, -0.04));

    const std::array<Eigen::Vector3d, 4> X_ref1 = {
        Eigen::Vector3d(0.30, 0.08, 4.20),
        Eigen::Vector3d(-0.22, -0.15, 4.60),
        Eigen::Vector3d(0.14, -0.09, 3.90),
        Eigen::Vector3d(-0.10, 0.19, 4.40),
    };
    const std::array<Eigen::Vector3d, 2> X_ref2 = {
        Eigen::Vector3d(0.06, -0.04, 4.10),
        Eigen::Vector3d(-0.16, 0.13, 4.70),
    };

    data.x_query1.reserve(X_ref1.size());
    data.x_ref1.reserve(X_ref1.size());
    data.x_query2.reserve(X_ref2.size());
    data.x_ref2.reserve(X_ref2.size());

    for (const Eigen::Vector3d &X : X_ref1) {
        data.x_query1.push_back(bearing_from_pose(data.pose_query, X));
        data.x_ref1.push_back(bearing_from_pose(data.pose_ref1, X));
    }
    for (const Eigen::Vector3d &X : X_ref2) {
        data.x_query2.push_back(bearing_from_pose(data.pose_query, X));
        data.x_ref2.push_back(bearing_from_pose(data.pose_ref2, X));
    }

    return data;
}

StructurelessTwoRefCase make_structureless_51_case() {
    StructurelessTwoRefCase data;
    data.pose_query = CameraPose(
        (Eigen::AngleAxisd(0.10, Eigen::Vector3d::UnitZ()) * Eigen::AngleAxisd(-0.18, Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(0.25, Eigen::Vector3d::UnitX()))
            .toRotationMatrix(),
        Eigen::Vector3d(0.18, -0.11, 0.35));
    data.pose_ref1 = CameraPose();
    const Eigen::Matrix3d R_ref2 = Eigen::Matrix3d::Identity();
    data.pose_ref2 = CameraPose(R_ref2, Eigen::Vector3d(-0.22, 0.06, -0.04));

    const std::array<Eigen::Vector3d, 5> X_ref1 = {
        Eigen::Vector3d(0.30, 0.12, 4.60),
        Eigen::Vector3d(-0.28, -0.08, 4.10),
        Eigen::Vector3d(0.08, -0.20, 3.60),
        Eigen::Vector3d(-0.06, 0.24, 4.40),
        Eigen::Vector3d(0.18, 0.05, 4.85),
    };
    const std::array<Eigen::Vector3d, 1> X_ref2 = {
        Eigen::Vector3d(-0.14, 0.09, 4.55),
    };

    data.x_query1.reserve(X_ref1.size());
    data.x_ref1.reserve(X_ref1.size());
    data.x_query2.reserve(X_ref2.size());
    data.x_ref2.reserve(X_ref2.size());

    for (const Eigen::Vector3d &X : X_ref1) {
        data.x_query1.push_back(bearing_from_pose(data.pose_query, X));
        data.x_ref1.push_back(bearing_from_pose(data.pose_ref1, X));
    }
    for (const Eigen::Vector3d &X : X_ref2) {
        data.x_query2.push_back(bearing_from_pose(data.pose_query, X));
        data.x_ref2.push_back(bearing_from_pose(data.pose_ref2, X));
    }

    return data;
}

StructurelessTwoRefCase make_structureless_33_case() {
    StructurelessTwoRefCase data;
    data.pose_query = make_pose_from_center(Eigen::AngleAxisd(0.14, Eigen::Vector3d::UnitX()),
                                            Eigen::AngleAxisd(-0.11, Eigen::Vector3d::UnitY()),
                                            Eigen::AngleAxisd(0.10, Eigen::Vector3d::UnitZ()),
                                            Eigen::Vector3d(0.09, -0.07, 0.04));
    data.pose_ref1 = make_pose_from_center(Eigen::AngleAxisd(-0.03, Eigen::Vector3d::UnitX()),
                                           Eigen::AngleAxisd(0.13, Eigen::Vector3d::UnitY()),
                                           Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitZ()),
                                           Eigen::Vector3d(-0.16, 0.06, 0.00));
    data.pose_ref2 = make_pose_from_center(Eigen::AngleAxisd(0.07, Eigen::Vector3d::UnitX()),
                                           Eigen::AngleAxisd(-0.09, Eigen::Vector3d::UnitY()),
                                           Eigen::AngleAxisd(0.04, Eigen::Vector3d::UnitZ()),
                                           Eigen::Vector3d(0.19, 0.04, -0.03));

    const std::array<Eigen::Vector3d, 3> X_ref1 = {
        Eigen::Vector3d(0.24, 0.11, 4.15),
        Eigen::Vector3d(-0.18, -0.10, 4.70),
        Eigen::Vector3d(0.10, -0.16, 3.85),
    };
    const std::array<Eigen::Vector3d, 3> X_ref2 = {
        Eigen::Vector3d(-0.12, 0.20, 4.35),
        Eigen::Vector3d(0.05, -0.06, 4.55),
        Eigen::Vector3d(-0.20, 0.07, 4.85),
    };

    data.x_query1.reserve(X_ref1.size());
    data.x_ref1.reserve(X_ref1.size());
    data.x_query2.reserve(X_ref2.size());
    data.x_ref2.reserve(X_ref2.size());

    for (const Eigen::Vector3d &X : X_ref1) {
        data.x_query1.push_back(bearing_from_pose(data.pose_query, X));
        data.x_ref1.push_back(bearing_from_pose(data.pose_ref1, X));
    }
    for (const Eigen::Vector3d &X : X_ref2) {
        data.x_query2.push_back(bearing_from_pose(data.pose_query, X));
        data.x_ref2.push_back(bearing_from_pose(data.pose_ref2, X));
    }

    return data;
}

Structureless6ptCase make_structureless_6pt_case() {
    Structureless6ptCase data;
    data.pose_query = make_pose_from_center(Eigen::AngleAxisd(0.18, Eigen::Vector3d::UnitX()),
                                            Eigen::AngleAxisd(-0.13, Eigen::Vector3d::UnitY()),
                                            Eigen::AngleAxisd(0.07, Eigen::Vector3d::UnitZ()),
                                            Eigen::Vector3d(0.08, -0.06, 0.05));

    data.pose_ref = {
        make_pose_from_center(Eigen::AngleAxisd(-0.04, Eigen::Vector3d::UnitX()),
                              Eigen::AngleAxisd(0.11, Eigen::Vector3d::UnitY()),
                              Eigen::AngleAxisd(-0.05, Eigen::Vector3d::UnitZ()),
                              Eigen::Vector3d(-0.22, 0.08, 0.02)),
        make_pose_from_center(Eigen::AngleAxisd(0.06, Eigen::Vector3d::UnitX()),
                              Eigen::AngleAxisd(-0.07, Eigen::Vector3d::UnitY()),
                              Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitZ()),
                              Eigen::Vector3d(0.15, -0.04, -0.03)),
        make_pose_from_center(Eigen::AngleAxisd(-0.02, Eigen::Vector3d::UnitX()),
                              Eigen::AngleAxisd(0.14, Eigen::Vector3d::UnitY()),
                              Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitZ()),
                              Eigen::Vector3d(-0.12, -0.07, 0.01)),
        make_pose_from_center(Eigen::AngleAxisd(0.05, Eigen::Vector3d::UnitX()),
                              Eigen::AngleAxisd(-0.09, Eigen::Vector3d::UnitY()),
                              Eigen::AngleAxisd(0.06, Eigen::Vector3d::UnitZ()),
                              Eigen::Vector3d(0.24, 0.06, -0.02)),
        make_pose_from_center(Eigen::AngleAxisd(-0.08, Eigen::Vector3d::UnitX()),
                              Eigen::AngleAxisd(0.10, Eigen::Vector3d::UnitY()),
                              Eigen::AngleAxisd(-0.02, Eigen::Vector3d::UnitZ()),
                              Eigen::Vector3d(-0.05, 0.12, 0.04)),
        make_pose_from_center(Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitX()),
                              Eigen::AngleAxisd(-0.12, Eigen::Vector3d::UnitY()),
                              Eigen::AngleAxisd(0.09, Eigen::Vector3d::UnitZ()),
                              Eigen::Vector3d(0.18, 0.02, -0.05)),
    };

    const std::array<Eigen::Vector3d, 6> X = {
        Eigen::Vector3d(0.26, 0.09, 4.25),
        Eigen::Vector3d(-0.19, -0.12, 4.55),
        Eigen::Vector3d(0.11, -0.18, 3.95),
        Eigen::Vector3d(-0.08, 0.21, 4.45),
        Eigen::Vector3d(0.17, 0.05, 4.90),
        Eigen::Vector3d(-0.15, 0.08, 4.65),
    };

    data.x_query.reserve(X.size());
    data.x_ref.reserve(X.size());
    for (size_t i = 0; i < X.size(); ++i) {
        data.x_query.push_back(bearing_from_pose(data.pose_query, X[i]));
        data.x_ref.push_back(bearing_from_pose(data.pose_ref[i], X[i]));
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

bool test_structureless_resection_42_recovers_query_pose() {
    const StructurelessTwoRefCase data = make_structureless_42_case();

    const std::vector<CameraPose> solutions = structureless_resection_42(
        data.x_query1, data.x_ref1, data.pose_ref1, data.x_query2, data.x_ref2, data.pose_ref2);

    REQUIRE(!solutions.empty());
    REQUIRE(contains_pose_close_to_gt(solutions, data.pose_query));
    return true;
}

bool test_structureless_resection_42_matches_manual_solver_transform() {
    const StructurelessTwoRefCase data = make_structureless_42_case();

    CameraPose pose_ref1_from_ref2 = data.pose_ref1;
    pose_ref1_from_ref2 = pose_ref1_from_ref2.compose(data.pose_ref2.inverse());

    std::vector<Eigen::Vector3d> x2_off;
    x2_off.reserve(data.x_ref1.size());
    for (const Eigen::Vector3d &x : data.x_ref1) {
        x2_off.push_back(pose_ref1_from_ref2.derotate(x.normalized()));
    }

    std::vector<CameraPose> raw_solutions;
    const int num_raw =
        gen_relpose_6pt_42(data.x_query1, x2_off, data.x_query2, data.x_ref2, pose_ref1_from_ref2.center(), &raw_solutions);

    const std::vector<CameraPose> wrapper_solutions = structureless_resection_42(
        data.x_query1, data.x_ref1, data.pose_ref1, data.x_query2, data.x_ref2, data.pose_ref2);

    REQUIRE(num_raw > 0);
    REQUIRE_EQ(raw_solutions.size(), wrapper_solutions.size());

    std::vector<CameraPose> manual_world_solutions;
    manual_world_solutions.reserve(raw_solutions.size());
    for (const CameraPose &pose_ref2_from_query : raw_solutions) {
        CameraPose pose_query_from_world = pose_ref2_from_query.inverse();
        pose_query_from_world = pose_query_from_world.compose(data.pose_ref2);
        manual_world_solutions.push_back(pose_query_from_world);
    }

    for (size_t i = 0; i < manual_world_solutions.size(); ++i) {
        REQUIRE(rotation_error(manual_world_solutions[i], wrapper_solutions[i]) <= 1e-10);
        REQUIRE(translation_error(manual_world_solutions[i], wrapper_solutions[i]) <= 1e-10);
    }
    return true;
}

bool test_structureless_resection_42_handles_scaled_reference_frame() {
    const StructurelessTwoRefCase data = make_structureless_42_case();
    const double scale = 37.5;

    const CameraPose scaled_pose_query = scale_pose_translation(data.pose_query, scale);
    const CameraPose scaled_pose_ref1 = scale_pose_translation(data.pose_ref1, scale);
    const CameraPose scaled_pose_ref2 = scale_pose_translation(data.pose_ref2, scale);

    const std::vector<CameraPose> solutions = structureless_resection_42(
        data.x_query1, data.x_ref1, scaled_pose_ref1, data.x_query2, data.x_ref2, scaled_pose_ref2);

    REQUIRE(!solutions.empty());
    REQUIRE(contains_pose_close_to_gt(solutions, scaled_pose_query));
    return true;
}

bool test_structureless_resection_51_recovers_query_pose() {
    const StructurelessTwoRefCase data = make_structureless_51_case();

    const std::vector<CameraPose> solutions = structureless_resection_51(
        data.x_query1, data.x_ref1, data.pose_ref1, data.x_query2, data.x_ref2, data.pose_ref2);

    REQUIRE(!solutions.empty());
    REQUIRE(contains_pose_close_to_gt(solutions, data.pose_query));
    return true;
}

bool test_structureless_resection_33_recovers_query_pose() {
    const StructurelessTwoRefCase data = make_structureless_33_case();

    const std::vector<CameraPose> solutions = structureless_resection_33(
        data.x_query1, data.x_ref1, data.pose_ref1, data.x_query2, data.x_ref2, data.pose_ref2);

    REQUIRE(!solutions.empty());
    REQUIRE(contains_pose_close_to_gt(solutions, data.pose_query));
    return true;
}

bool test_structureless_resection_6pt_recovers_query_pose() {
    const Structureless6ptCase data = make_structureless_6pt_case();

    const std::vector<CameraPose> solutions = structureless_resection_6pt(data.x_query, data.x_ref, data.pose_ref);

    REQUIRE(!solutions.empty());
    REQUIRE(contains_pose_close_to_gt(solutions, data.pose_query));
    return true;
}

bool test_structureless_resection_6pt_handles_scaled_reference_frame() {
    const Structureless6ptCase data = make_structureless_6pt_case();
    const double scale = 19.0;

    const CameraPose scaled_pose_query = scale_pose_translation(data.pose_query, scale);
    const std::vector<CameraPose> scaled_pose_ref = scale_pose_translations(data.pose_ref, scale);

    const std::vector<CameraPose> solutions = structureless_resection_6pt(data.x_query, data.x_ref, scaled_pose_ref);

    REQUIRE(!solutions.empty());
    REQUIRE(contains_pose_close_to_gt(solutions, scaled_pose_query));
    return true;
}

} // namespace test::structureless_resection

std::vector<Test> register_structureless_resection_test() {
    return {
        TEST(test::structureless_resection::test_gen_relpose_6pt_42_reordered_direct),
        TEST(test::structureless_resection::test_gen_relpose_6pt_42_reordered_benchmark_layout),
        TEST(test::structureless_resection::test_structureless_resection_42_recovers_query_pose),
        TEST(test::structureless_resection::test_structureless_resection_42_matches_manual_solver_transform),
        TEST(test::structureless_resection::test_structureless_resection_42_handles_scaled_reference_frame),
        TEST(test::structureless_resection::test_structureless_resection_51_recovers_query_pose),
        TEST(test::structureless_resection::test_structureless_resection_33_recovers_query_pose),
        TEST(test::structureless_resection::test_structureless_resection_6pt_recovers_query_pose),
        TEST(test::structureless_resection::test_structureless_resection_6pt_handles_scaled_reference_frame),
    };
}
