#ifndef POSELIB_P3P_COMMON_H
#define POSELIB_P3P_COMMON_H

#include "PoseLib/real_matrix.h"

#include <cmath>

namespace poselib {

bool inline root2real(Real b, Real c, Real &r1, Real &r2) {
    Real THRESHOLD = -1.0e-12;
    Real v = b * b - 4.0 * c;
    if (v < THRESHOLD) {
        r1 = r2 = -0.5 * b;
        return v >= 0;
    }
    if (v > THRESHOLD && v < 0.0) {
        r1 = -0.5 * b;
        r2 = -2;
        return true;
    }

    Real y = std::sqrt(v);
    if (b < 0) {
        r1 = 0.5 * (-b + y);
        r2 = 0.5 * (-b - y);
    } else {
        r1 = 2.0 * c / (-b + y);
        r2 = 2.0 * c / (-b - y);
    }
    return true;
}

inline std::array<Vector3, 2> compute_pq(Matrix3x3 C) {
    std::array<Vector3, 2> pq;
    Matrix3x3 C_adj;

    C_adj(0, 0) = C(1, 2) * C(2, 1) - C(1, 1) * C(2, 2);
    C_adj(1, 1) = C(0, 2) * C(2, 0) - C(0, 0) * C(2, 2);
    C_adj(2, 2) = C(0, 1) * C(1, 0) - C(0, 0) * C(1, 1);
    C_adj(0, 1) = C(0, 1) * C(2, 2) - C(0, 2) * C(2, 1);
    C_adj(0, 2) = C(0, 2) * C(1, 1) - C(0, 1) * C(1, 2);
    C_adj(1, 0) = C_adj(0, 1);
    C_adj(1, 2) = C(0, 0) * C(1, 2) - C(0, 2) * C(1, 0);
    C_adj(2, 0) = C_adj(0, 2);
    C_adj(2, 1) = C_adj(1, 2);

    Vector3 v;
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

    C(0, 1) -= v(2);
    C(0, 2) += v(1);
    C(1, 2) -= v(0);
    C(1, 0) += v(2);
    C(2, 0) -= v(1);
    C(2, 1) += v(0);

    pq[0] = C.col(0);
    pq[1] = C.row(0);

    return pq;
}

// Performs a few newton steps on the equations
inline void refine_lambda(Real &lambda1, Real &lambda2, Real &lambda3, const Real a12, const Real a13, const Real a23,
                          const Real b12, const Real b13, const Real b23) {

    for (int iter = 0; iter < 5; ++iter) {
        Real r1 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda2 * b12 + lambda2 * lambda2 - a12);
        Real r2 = (lambda1 * lambda1 - 2.0 * lambda1 * lambda3 * b13 + lambda3 * lambda3 - a13);
        Real r3 = (lambda2 * lambda2 - 2.0 * lambda2 * lambda3 * b23 + lambda3 * lambda3 - a23);
        if (std::abs(r1) + std::abs(r2) + std::abs(r3) < 1e-10)
            return;
        Real x11 = lambda1 - lambda2 * b12;
        Real x12 = lambda2 - lambda1 * b12;
        Real x21 = lambda1 - lambda3 * b13;
        Real x23 = lambda3 - lambda1 * b13;
        Real x32 = lambda2 - lambda3 * b23;
        Real x33 = lambda3 - lambda2 * b23;
        Real detJ = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant
        // This uses the closed form of the inverse for the jacobean.
        // Due to the zero elements this actually becomes quite nice.
        lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * detJ;
        lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * detJ;
        lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * detJ;
    }
}

} // namespace poselib

#endif // POSELIB_P3P_COMMON_H
