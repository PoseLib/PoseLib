#include "relpose_7pt.h"

#include "PoseLib/misc/univariate.h"

#include <Eigen/Dense>

namespace poselib {

int relpose_7pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                std::vector<Eigen::Matrix3d> *fundamental_matrices) {

    // Compute nullspace to epipolar constraints
    Eigen::Matrix<double, 9, 7> epipolar_constraints;
    for (size_t i = 0; i < 7; ++i) {
        epipolar_constraints.col(i) << x1[i](0) * x2[i], x1[i](1) * x2[i], x1[i](2) * x2[i];
    }
    Eigen::Matrix<double, 9, 9> Q = epipolar_constraints.fullPivHouseholderQr().matrixQ();
    Eigen::Matrix<double, 9, 2> N = Q.rightCols(2);

    // coefficients for det(F(x)) = 0
    const double c3 = N(0, 0) * N(4, 0) * N(8, 0) - N(0, 0) * N(5, 0) * N(7, 0) - N(1, 0) * N(3, 0) * N(8, 0) +
                      N(1, 0) * N(5, 0) * N(6, 0) + N(2, 0) * N(3, 0) * N(7, 0) - N(2, 0) * N(4, 0) * N(6, 0);
    const double c2 = N(0, 0) * N(4, 0) * N(8, 1) + N(0, 0) * N(4, 1) * N(8, 0) - N(0, 0) * N(5, 0) * N(7, 1) -
                      N(0, 0) * N(5, 1) * N(7, 0) + N(0, 1) * N(4, 0) * N(8, 0) - N(0, 1) * N(5, 0) * N(7, 0) -
                      N(1, 0) * N(3, 0) * N(8, 1) - N(1, 0) * N(3, 1) * N(8, 0) + N(1, 0) * N(5, 0) * N(6, 1) +
                      N(1, 0) * N(5, 1) * N(6, 0) - N(1, 1) * N(3, 0) * N(8, 0) + N(1, 1) * N(5, 0) * N(6, 0) +
                      N(2, 0) * N(3, 0) * N(7, 1) + N(2, 0) * N(3, 1) * N(7, 0) - N(2, 0) * N(4, 0) * N(6, 1) -
                      N(2, 0) * N(4, 1) * N(6, 0) + N(2, 1) * N(3, 0) * N(7, 0) - N(2, 1) * N(4, 0) * N(6, 0);
    const double c1 = N(0, 0) * N(4, 1) * N(8, 1) - N(0, 0) * N(5, 1) * N(7, 1) + N(0, 1) * N(4, 0) * N(8, 1) +
                      N(0, 1) * N(4, 1) * N(8, 0) - N(0, 1) * N(5, 0) * N(7, 1) - N(0, 1) * N(5, 1) * N(7, 0) -
                      N(1, 0) * N(3, 1) * N(8, 1) + N(1, 0) * N(5, 1) * N(6, 1) - N(1, 1) * N(3, 0) * N(8, 1) -
                      N(1, 1) * N(3, 1) * N(8, 0) + N(1, 1) * N(5, 0) * N(6, 1) + N(1, 1) * N(5, 1) * N(6, 0) +
                      N(2, 0) * N(3, 1) * N(7, 1) - N(2, 0) * N(4, 1) * N(6, 1) + N(2, 1) * N(3, 0) * N(7, 1) +
                      N(2, 1) * N(3, 1) * N(7, 0) - N(2, 1) * N(4, 0) * N(6, 1) - N(2, 1) * N(4, 1) * N(6, 0);
    const double c0 = N(0, 1) * N(4, 1) * N(8, 1) - N(0, 1) * N(5, 1) * N(7, 1) - N(1, 1) * N(3, 1) * N(8, 1) +
                      N(1, 1) * N(5, 1) * N(6, 1) + N(2, 1) * N(3, 1) * N(7, 1) - N(2, 1) * N(4, 1) * N(6, 1);

    // Solve the cubic
    double inv_c3 = 1.0 / c3;
    double roots[3];
    int n_roots = univariate::solve_cubic_real(c2 * inv_c3, c1 * inv_c3, c0 * inv_c3, roots);

    // Reshape back into 3x3 matrices
    fundamental_matrices->clear();
    fundamental_matrices->reserve(n_roots);
    for (int i = 0; i < n_roots; ++i) {
        Eigen::Matrix<double, 9, 1> f = N.col(0) * roots[i] + N.col(1);
        f.normalize();
        fundamental_matrices->push_back(Eigen::Map<Eigen::Matrix3d>(f.data()));
    }

    return n_roots;
}

} // namespace poselib