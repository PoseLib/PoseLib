#include <Eigen/Dense>
#include "p5pf.h"
#include "p5lp_radial.h"

namespace poselib {


    int p5pf(const std::vector<Eigen::Vector2d>& points2d,
        const std::vector<Eigen::Vector3d>& points3d,
        std::vector<CameraPose>* output_poses,
        std::vector<double>* output_focals,
        bool normalize_input,
        bool model_distortion)
    {
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

            int n_sols = p5pf(scaled_points2d, points3d, output_poses, output_focals, false, model_distortion);

            for (int i = 0; i < n_sols; ++i) {
                (*output_focals)[i] *= focal0;
            }
            return n_sols;
        }

        std::vector<CameraPose> poses_radial;
        p5lp_radial(points2d, points3d, &poses_radial);

        Eigen::Matrix2d A, AtA;
        Eigen::Vector2d b, Atb;

        Eigen::Matrix<double,2,3> A3;
        Eigen::Matrix<double,3,3> AtA3;
        Eigen::Vector2d b3;
        Eigen::Vector3d Atb3;


        output_poses->clear();
        output_focals->clear();
        for(size_t i = 0; i < poses_radial.size(); ++i) {
            AtA.setZero();
            Atb.setZero();
            
            CameraPose p = poses_radial[i];
            double focal;
            if(model_distortion) {
                AtA3.setZero();
                Atb3.setZero();

                for (int k = 0; k < 5; ++k) {
                    Eigen::Vector3d RX = p.rotate(points3d[k]);
                    Eigen::Vector2d RX2 = RX.topRows(2) + p.t.topRows(2);
                    A3 << RX2, points2d[k].squaredNorm() * RX2, -points2d[k];
                    b3 << RX(2) * points2d[k];
                    AtA3 += A3.transpose() * A3;
                    Atb3 += A3.transpose() * b3;
                }

                // Solve for focal length and t3
                Eigen::Vector3d sol = AtA3.inverse() * Atb3;
                focal = sol(0);
                // dist = sol(1) / focal;
                p.t(2) = sol(2);
            } else {
                for (int k = 0; k < 5; ++k) {
                    Eigen::Vector3d RX = p.rotate(points3d[k]);
                    A << RX.topRows(2) + p.t.topRows(2), -points2d[k];
                    b << RX(2) * points2d[k];
                    AtA += A.transpose() * A;
                    Atb += A.transpose() * b;
                }

                // Solve for focal length and t3
                Eigen::Vector2d sol = AtA.inverse() * Atb;
                focal = sol(0);
                p.t(2) = sol(1);
            }

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
        }

        return output_poses->size();
    }

}