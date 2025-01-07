#include "gen_relpose_5p1pt.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/solvers/relpose_5pt.h"

#include <Eigen/Dense>

namespace poselib {

int gen_relpose_5p1pt(const std::vector<Eigen::Vector3_t> &p1, const std::vector<Eigen::Vector3_t> &x1,
                      const std::vector<Eigen::Vector3_t> &p2, const std::vector<Eigen::Vector3_t> &x2,
                      std::vector<CameraPose> *output) {

    output->clear();
    relpose_5pt(x1, x2, output);

    for (size_t k = 0; k < output->size(); ++k) {
        CameraPose &pose = (*output)[k];

        // the translation is given by
        //  t = p2 - R_5pt*p1 + gamma * t_5pt = a + gamma * b

        // we need to solve for gamma using our extra correspondence
        //  R * (p1 + lambda_1 * x1) + a + gamma * b = lambda_2 * x2 + p2

        Eigen::Matrix3_t R = pose.R();
        Eigen::Vector3_t a = p2[0] - R * p1[0];
        Eigen::Vector3_t b = pose.t;

        // vector used to eliminate lambda1 and lambda2
        Eigen::Vector3_t w = x2[5].cross(R * x1[5]);

        const real_t c0 = w.dot(p2[5] - R * p1[5] - a);
        const real_t c1 = w.dot(b);

        const real_t gamma = c0 / c1;

        pose.t = a + gamma * b;
        // TODO: Cheirality check for the last point
    }

    return output->size();
}

} // namespace poselib