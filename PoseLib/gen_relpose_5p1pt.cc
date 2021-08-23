#include "gen_relpose_5p1pt.h"
#include "relpose_5pt.h"
#include "misc/essential.h"
#include <Eigen/Dense>

namespace pose_lib {

int gen_relpose_5p1pt(const std::vector<Eigen::Vector3d> &p1, const std::vector<Eigen::Vector3d> &x1,
                      const std::vector<Eigen::Vector3d> &p2, const std::vector<Eigen::Vector3d> &x2, std::vector<CameraPose> *output) {

    output->clear();
    relpose_5pt(x1, x2, output);    

    for(size_t k = 0; k < output->size(); ++k) {
        CameraPose &pose = (*output)[k];

        // the translation is given by
        //  t = p2 - R_5pt*p1 + gamma * t_5pt = a + gamma * b
        
        // we need to solve for gamma using our extra correspondence
        //  R * (p1 + lambda_1 * x1) + a + gamma * b = lambda_2 * x2 + p2



        Eigen::Vector3d a = p2[0] - pose.R * p1[0];
        Eigen::Vector3d b = pose.t;

        // vector used to eliminate lambda1 and lambda2
        Eigen::Vector3d w = x2[5].cross(pose.R * x1[5]);

        const double c0 = w.dot(p2[5] - pose.R * p1[5] - a);
        const double c1 = w.dot(b);

        const double gamma = c0 / c1;
        
        pose.t = a + gamma * b;
    }

    return output->size();
}

} // namespace pose_lib