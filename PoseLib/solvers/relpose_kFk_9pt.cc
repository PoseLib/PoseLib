//
// Created by kocur on 15-May-24.
//
#include "PoseLib/camera_pose.h"

#include <Eigen/Dense>
#include <vector>

namespace poselib {
int relpose_kFk_9pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                    std::vector<ProjectiveImagePair> *models) {
    int sample_number_ = 9;
    Eigen::MatrixXd D1(sample_number_, 9);
    Eigen::MatrixXd D2(sample_number_, 9);
    D2.setZero();
    Eigen::MatrixXd D3(sample_number_, 9);
    D3.setZero();

    // form a linear system: i-th row of A(=a) represents
    // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0

    for (int i = 0; i < sample_number_; i++) {
        const double u1 = x1[i](0), v1 = x1[i](1), u2 = x2[i](0), v2 = x2[i](1), uv1 = u1 * u1 + v1 * v1,
                     uv2 = u2 * u2 + v2 * v2;

        // If not weighted least-squares is applied
        D1(i, 0) = u1 * u2;
        D1(i, 1) = u2 * v1;
        D1(i, 2) = u2;
        D1(i, 3) = u1 * v2;
        D1(i, 4) = v1 * v2;
        D1(i, 5) = v2;
        D1(i, 6) = u1;
        D1(i, 7) = v1;
        D1(i, 8) = 1.0;

        D2(i, 2) = u2 * uv1;
        D2(i, 5) = v2 * uv1;
        D2(i, 6) = u1 * uv2;
        D2(i, 7) = v1 * uv2;
        D2(i, 8) = uv1 + uv2;

        D3(i, 8) = uv1 * uv2;
    }
    Eigen::MatrixXd M(9, 6);
    Eigen::MatrixXd D4(9, 6);
    D4 << D2.col(2), D2.col(5), D2.col(6), D2.col(7), D2.col(8), D3.col(8);
    M = -D1.partialPivLu().solve(D4);

    Eigen::MatrixXd D = Eigen::MatrixXd::Zero(6, 6);
    D(0, 5) = 1.0;
    D.row(1) = M.row(2);
    D.row(2) = M.row(5);
    D.row(3) = M.row(6);
    D.row(4) = M.row(7);
    D.row(5) = M.row(8);

    Eigen::EigenSolver<Eigen::MatrixXd> es(D);
    const Eigen::VectorXcd &eigenvalues = es.eigenvalues();
    Eigen::Matrix<double, 9, 1> f;
    Eigen::MatrixXd C(9, 9);

    models->reserve(6);

    for (int i = 0; i < 6; i++) {
        // Only consider real solutions.
        if (std::abs(eigenvalues(i).imag()) >= 1e-12) {
            continue;
        }

        double lambda = 1.0 / eigenvalues(i).real(); // distortion parameter
        C = D1 + lambda * D2 + lambda * lambda * D3;

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeFullV);

        f = svd.matrixV().col(8);

        Eigen::Matrix3d F;
        F << f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7], f[8];

        Camera cam = Camera("DIVISION", std::vector<double>{1.0, 1.0, 0.0, 0.0, lambda}, -1, -1);
        models->emplace_back(ProjectiveImagePair(F, cam, cam));
    }

    return models->size();
}
} // namespace poselib