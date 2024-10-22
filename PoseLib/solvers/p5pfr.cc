#include "p5pfr.h"

#include "p5lp_radial.h"

#include <Eigen/Dense>

namespace poselib {

int p5pfr(const std::vector<Eigen::Vector2d> &points2d, const std::vector<Eigen::Vector3d> &points3d,
         std::vector<CameraPose> *output_poses, std::vector<double> *output_focals, std::vector<double> *output_dist, bool normalize_input) {
    if (normalize_input) {
        double focal0 = 0.0;
        for (int i = 0; i < 5; ++i) {
            focal0 += points2d[i].norm();
        }
        focal0 /= 5;

        std::vector<Eigen::Vector2d> scaled_points2d;
        scaled_points2d.reserve(5);
        for (int i = 0; i < 5; ++i) {
            scaled_points2d.push_back(points2d[i] / focal0);
        }

        int n_sols = p5pfr(scaled_points2d, points3d, output_poses, output_focals, output_dist, false);

        for (int i = 0; i < n_sols; ++i) {
            (*output_focals)[i] *= focal0;
        }
        return n_sols;
    }

    std::vector<CameraPose> poses_radial;
    p5lp_radial(points2d, points3d, &poses_radial);

    Eigen::Matrix<double, 2, 3> A;
    Eigen::Vector2d b;
    Eigen::Matrix3d AtA;
    Eigen::Vector3d Atb;

    output_poses->clear();
    output_focals->clear();
    for (size_t i = 0; i < poses_radial.size(); ++i) {
        AtA.setZero();
        Atb.setZero();

        // f * (RX(1:2) + t(1:2)) = (RX(3) + t(3)) * x / (1+r2*k)
        // (f + r2*f*k) * (RX(1:2) + t(1:2)) = (RX(3) + t(3)) * x
        
        // [RX(1:2) + t(1:2), r2*(RX(1:2) + t(1:2)), -x] * [f; fk; t3] = RX(3)*x

        CameraPose p = poses_radial[i];
        for (int k = 0; k < 5; ++k) {
            double r2 = points2d[k].squaredNorm();
            Eigen::Vector3d RX = p.rotate(points3d[k]);
            Eigen::Vector2d RX2t = RX.topRows(2) + p.t.topRows(2);
            A << RX2t, r2*RX2t, -points2d[k];
            b << RX(2) * points2d[k];
            AtA += A.transpose() * A;
            Atb += A.transpose() * b;
        }

        // Solve for focal length and t3
        Eigen::Vector3d sol = AtA.inverse() * Atb;
        double focal = sol(0);
        double dist = sol(1) / (focal * focal * focal);
        p.t(2) = sol(2);

        // Correct sign
        if (focal < 0) {
            focal = -focal;
            Eigen::Matrix3d R = p.R();
            R.row(0) = -R.row(0);
            R.row(1) = -R.row(1);
            p.q = rotmat_to_quat(R);
            p.t(0) = -p.t(0);
            p.t(1) = -p.t(1);
        }
        output_poses->emplace_back(p);
        output_focals->emplace_back(focal);
        output_dist->emplace_back(dist);
    }

    return output_poses->size();
}

} // namespace poselib