#include "gen_relpose_5p1pt.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/solvers/relpose_5pt.h"

#include <Eigen/Dense>

namespace poselib {

int gen_relpose_5p1pt(const std::vector<Vector3> &p1, const std::vector<Vector3> &x1, const std::vector<Vector3> &p2,
                      const std::vector<Vector3> &x2, std::vector<CameraPose> *output) {

    output->clear();
    relpose_5pt(x1, x2, output);

    for (size_t k = 0; k < output->size(); ++k) {
        CameraPose &pose = (*output)[k];

        // the translation is given by
        //  t = p2 - R_5pt*p1 + gamma * t_5pt = a + gamma * b

        // we need to solve for gamma using our extra correspondence
        //  R * (p1 + lambda_1 * x1) + a + gamma * b = lambda_2 * x2 + p2

        Matrix3x3 R = pose.R();
        Vector3 a = p2[0] - R * p1[0];
        Vector3 b = pose.t;

        // vector used to eliminate lambda1 and lambda2
        Vector3 w = x2[5].cross(R * x1[5]);

        const real c0 = w.dot(p2[5] - R * p1[5] - a);
        const real c1 = w.dot(b);

        const real gamma = c0 / c1;

        pose.t = a + gamma * b;
        // TODO: Cheirality check for the last point
    }

    return output->size();
}

} // namespace poselib