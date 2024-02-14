#ifndef POSELIB_P3P_COMMON_H
#define POSELIB_P3P_COMMON_H

#include <cmath>

namespace poselib {

inline double cubic_trigonometric_solution(const double alpha, const double beta, const double k2, const double alpha3) {
    const double H = std::sqrt(-alpha3);
    const double I = std::sqrt(-alpha / 3.0);
    const double J = std::acos(-beta / (2.0 * H));
    const double K = std::cos(J / 3.0);
    return 2.0 * I * K - k2 / 3.0;  
}

 
inline double cubic_cardano_solution(const double beta, const double G, const double k2) {
    const double M = std::cbrt(-0.5 * beta + sqrt(G));
    const double N = -std::cbrt(0.5 * beta + sqrt(G));
    return M + N - k2 / 3.0;
}

bool inline root2real(double b, double c, double &r1, double &r2) {
    double THRESHOLD = -1.0e-12;
    double v = b * b - 4.0 * c;
    if (v < THRESHOLD) {
        r1 = r2 = -0.5 * b;
        return v >= 0;
    }
    if (v > THRESHOLD && v < 0.0) {
        r1 = -0.5 * b;
        r2 = -2;
        return true;
    }

    double y = std::sqrt(v);
    if (b < 0) {
        r1 = 0.5 * (-b + y);
        r2 = 0.5 * (-b - y);
    } else {
        r1 = 2.0 * c / (-b + y);
        r2 = 2.0 * c / (-b - y);
    }
    return true;
}

inline std::array<Eigen::Vector3d, 2> compute_pq(Eigen::Matrix3d C) {
    std::array<Eigen::Vector3d, 2> pq;
    Eigen::Matrix3d C_adj;

    C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
    C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
    C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
    C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
    C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
    C_adj(1, 0) = C_adj(0, 1);
    C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
    C_adj(2, 0) = C_adj(0, 2);
    C_adj(2, 1) = C_adj(1, 2);

    Eigen::Vector3d v;
    if (C_adj(0, 0) > C_adj(1, 1)) {
        if (C_adj(0, 0) > C_adj(2, 2)) {
            v = C_adj.col(0) / std::sqrt(C_adj(0, 0));
        } else {
            v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
        }
    } else if (C_adj(1, 1) > C_adj(2, 2)) {
        v = C_adj.col(1) / std::sqrt(C_adj(1, 1));
    } else {
        v = C_adj.col(2) / std::sqrt(C_adj(2, 2));
    }

    Eigen::Matrix3d D = C;
    D(0, 1) -= v(2);
    D(0, 2) += v(1);
    D(1, 2) -= v(0);
    D(1, 0) += v(2);
    D(2, 0) -= v(1);
    D(2, 1) += v(0);

    pq[0] = D.col(0);
    pq[1] = D.row(0);

    return pq;
}

// Performs a few newton steps on the equations
inline void refine_lambda(double &lambda1, double &lambda2, double &lambda3, const double a12, const double a13,
                          const double a23, const double b12, const double b13, const double b23) {

    for (int iter = 0; iter < 5; ++iter) {
        double r1 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
        double r2 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
        double r3 = (lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);
        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-10)
            return;
        double x11 = lambda1 - lambda2 * b12;
        double x12 = lambda2 - lambda1 * b12;
        double x21 = lambda1 - lambda3 * b13;
        double x23 = lambda3 - lambda1 * b13;
        double x32 = lambda2 - lambda3 * b23;
        double x33 = lambda3 - lambda2 * b23;
        double detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant
        // This uses the closed form of the inverse for the jacobean.
        // Due to the zero elements this actually becomes quite nice.
        lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
    }
}

} // namespace poselib

#endif // POSELIB_P3P_COMMON_H
