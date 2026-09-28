#include "PoseLib/misc/essential.h"
#include "PoseLib/solvers/gen_relpose_5p1pt.h"
#include "test.h"
#include "test_rng.h"

using namespace poselib;

namespace {

// Rig with cameras spread over a distance comparable to the scene depth. The 6th correspondence comes from a different
// camera, so checking its cheirality w.r.t. the first camera pair would often reject the true pose.
bool test_gen_relpose_5p1pt() {
    auto rng = test_rng::make_rng("gen_relpose_5p1pt");
    const int num_samples = 1000;
    int num_found = 0;
    for (int sample = 0; sample < num_samples; ++sample) {
        const Eigen::Matrix3d R =
            Eigen::AngleAxisd(rng.uniform(-0.3, 0.3), test_rng::symmetric_vec3(rng).normalized()).toRotationMatrix();
        const Eigen::Vector3d t = test_rng::symmetric_vec3(rng);

        // Camera centers in the rig frame. Odd samples have the 6th correspondence differ on only one side.
        const Eigen::Vector3d cam_a = 2.0 * test_rng::symmetric_vec3(rng).normalized();
        const Eigen::Vector3d cam_b = 2.0 * test_rng::symmetric_vec3(rng).normalized();
        const Eigen::Vector3d cam_b2 = sample % 2 == 0 ? cam_b : cam_a;

        std::vector<Eigen::Vector3d> p1, x1, p2, x2;
        for (int i = 0; i < 6; ++i) {
            const Eigen::Vector3d c1 = i < 5 ? cam_a : cam_b;
            const Eigen::Vector3d c2 = i < 5 ? cam_a : cam_b2;
            const Eigen::Vector3d X = Eigen::Vector3d(0, 0, 5) + test_rng::symmetric_vec3(rng);
            p1.push_back(c1);
            x1.push_back((X - c1).normalized());
            p2.push_back(c2);
            x2.push_back((R * X + t - c2).normalized());
        }

        CameraPoseVector solutions;
        const int count = gen_relpose_5p1pt(p1, x1, p2, x2, &solutions);
        REQUIRE_EQ(count, solutions.size());

        double min_error = std::numeric_limits<double>::infinity();
        for (const CameraPose &pose : solutions) {
            // Every returned solution must have all six points in front of their own cameras
            REQUIRE(check_cheirality(pose, p1, x1, p2, x2));
            min_error = std::min(min_error, (pose.R() - R).norm() + (pose.t - t).norm());
        }
        if (min_error < 1e-6) {
            ++num_found;
        }
    }
    // The 5pt solver itself occasionally loses accuracy on random minimal samples
    REQUIRE(num_found >= 0.95 * num_samples);
    return true;
}

} // namespace

std::vector<Test> register_solvers_gen_relpose_test() { return {TEST(test_gen_relpose_5p1pt)}; }
