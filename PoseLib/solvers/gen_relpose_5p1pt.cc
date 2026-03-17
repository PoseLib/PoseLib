#include "gen_relpose_5p1pt.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/solvers/relpose_5pt.h"

#include <Eigen/Dense>

namespace poselib {
namespace {

bool normalize_bearings(const std::vector<Eigen::Vector3d> &input, std::vector<Eigen::Vector3d> *output) {
    output->clear();
    output->reserve(input.size());
    for (const Eigen::Vector3d &x : input) {
        const double norm = x.norm();
        if (norm <= 0.0) {
            output->clear();
            return false;
        }
        output->push_back(x / norm);
    }
    return true;
}

} // namespace

int gen_relpose_5p1pt(const std::vector<Eigen::Vector3d> &p1, const std::vector<Eigen::Vector3d> &x1,
                      const std::vector<Eigen::Vector3d> &p2, const std::vector<Eigen::Vector3d> &x2,
                      std::vector<CameraPose> *output) {

    output->clear();
    relpose_5pt(x1, x2, output);

    for (size_t k = 0; k < output->size(); ++k) {
        CameraPose &pose = (*output)[k];

        // the translation is given by
        //  t = p2 - R_5pt*p1 + gamma * t_5pt = a + gamma * b

        // we need to solve for gamma using our extra correspondence
        //  R * (p1 + lambda_1 * x1) + a + gamma * b = lambda_2 * x2 + p2

        Eigen::Matrix3d R = pose.R();
        Eigen::Vector3d a = p2[0] - R * p1[0];
        Eigen::Vector3d b = pose.t;

        // vector used to eliminate lambda1 and lambda2
        Eigen::Vector3d w = x2[5].cross(R * x1[5]);

        const double c0 = w.dot(p2[5] - R * p1[5] - a);
        const double c1 = w.dot(b);

        const double gamma = c0 / c1;

        pose.t = a + gamma * b;
        // TODO: Cheirality check for the last point
    }

    return output->size();
}

std::vector<CameraPose> structureless_resection_51(const std::vector<Eigen::Vector3d> &x_query1,
                                                   const std::vector<Eigen::Vector3d> &x_ref1,
                                                   const CameraPose &pose_ref1,
                                                   const std::vector<Eigen::Vector3d> &x_query2,
                                                   const std::vector<Eigen::Vector3d> &x_ref2,
                                                   const CameraPose &pose_ref2) {
    constexpr int kNumRef1 = 5;
    constexpr int kNumRef2 = 1;
    constexpr int kNumObs = 6;

    if (x_query1.size() != kNumRef1 || x_ref1.size() != kNumRef1 || x_query2.size() != kNumRef2 ||
        x_ref2.size() != kNumRef2) {
        return {};
    }

    std::vector<Eigen::Vector3d> x1_group1;
    std::vector<Eigen::Vector3d> x1_group2;
    std::vector<Eigen::Vector3d> x2_group1;
    if (!normalize_bearings(x_query1, &x1_group1) || !normalize_bearings(x_query2, &x1_group2) ||
        !normalize_bearings(x_ref1, &x2_group1)) {
        return {};
    }

    CameraPose pose_ref2_from_ref1 = pose_ref2;
    pose_ref2_from_ref1 = pose_ref2_from_ref1.compose(pose_ref1.inverse());
    const Eigen::Vector3d p2_group2 = pose_ref2_from_ref1.center();
    const double p2_scale = p2_group2.norm();
    if (p2_scale <= 1e-12) {
        return {};
    }

    std::vector<Eigen::Vector3d> x2_group2;
    if (!normalize_bearings(x_ref2, &x2_group2)) {
        return {};
    }

    std::vector<Eigen::Vector3d> p1(kNumObs, Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> x1;
    std::vector<Eigen::Vector3d> p2;
    std::vector<Eigen::Vector3d> x2;
    x1.reserve(kNumObs);
    p2.reserve(kNumObs);
    x2.reserve(kNumObs);

    for (int i = 0; i < kNumRef1; ++i) {
        x1.push_back(x1_group1[i]);
        x2.push_back(x2_group1[i]);
        p2.emplace_back(Eigen::Vector3d::Zero());
    }

    x1.push_back(x1_group2[0]);
    x2.push_back(pose_ref2_from_ref1.derotate(x2_group2[0]));
    p2.push_back(p2_group2 / p2_scale);

    std::vector<CameraPose> raw_solutions;
    gen_relpose_5p1pt(p1, x1, p2, x2, &raw_solutions);

    std::vector<CameraPose> solutions;
    solutions.reserve(raw_solutions.size());
    for (CameraPose pose_ref1_from_query : raw_solutions) {
        pose_ref1_from_query.t *= p2_scale;
        const CameraPose pose_query_from_ref1 = pose_ref1_from_query.inverse();
        CameraPose pose_query_from_world = pose_query_from_ref1;
        pose_query_from_world = pose_query_from_world.compose(pose_ref1);
        solutions.push_back(pose_query_from_world);
    }

    return solutions;
}

} // namespace poselib
