#include "relpose_monodepth_3pt.h"

namespace poselib {

std::pair<Eigen::MatrixXd, Eigen::VectorXd> solver_p3p_mono_3d(const Eigen::VectorXd &data) {

    const double *d = data.data();
    Eigen::VectorXd coeffs(18);
    coeffs[0] = std::pow(d[6], 2) - 2 * d[6] * d[7] + std::pow(d[7], 2) + std::pow(d[9], 2) - 2 * d[9] * d[10] +
                std::pow(d[10], 2);
    coeffs[1] = -std::pow(d[0], 2) + 2 * d[0] * d[1] - std::pow(d[1], 2) - std::pow(d[3], 2) + 2 * d[3] * d[4] -
                std::pow(d[4], 2);
    coeffs[2] = 2 * std::pow(d[6], 2) * d[15] - 2 * d[6] * d[7] * d[15] + 2 * std::pow(d[9], 2) * d[15] -
                2 * d[9] * d[10] * d[15] - 2 * d[6] * d[7] * d[16] + 2 * std::pow(d[7], 2) * d[16] -
                2 * d[9] * d[10] * d[16] + 2 * std::pow(d[10], 2) * d[16];
    coeffs[3] = std::pow(d[6], 2) * std::pow(d[15], 2) + std::pow(d[9], 2) * std::pow(d[15], 2) -
                2 * d[6] * d[7] * d[15] * d[16] - 2 * d[9] * d[10] * d[15] * d[16] +
                std::pow(d[7], 2) * std::pow(d[16], 2) + std::pow(d[10], 2) * std::pow(d[16], 2) + std::pow(d[15], 2) -
                2 * d[15] * d[16] + std::pow(d[16], 2);
    coeffs[4] = -2 * std::pow(d[0], 2) * d[12] + 2 * d[0] * d[1] * d[12] - 2 * std::pow(d[3], 2) * d[12] +
                2 * d[3] * d[4] * d[12] + 2 * d[0] * d[1] * d[13] - 2 * std::pow(d[1], 2) * d[13] +
                2 * d[3] * d[4] * d[13] - 2 * std::pow(d[4], 2) * d[13];
    coeffs[5] = -std::pow(d[0], 2) * std::pow(d[12], 2) - std::pow(d[3], 2) * std::pow(d[12], 2) +
                2 * d[0] * d[1] * d[12] * d[13] + 2 * d[3] * d[4] * d[12] * d[13] -
                std::pow(d[1], 2) * std::pow(d[13], 2) - std::pow(d[4], 2) * std::pow(d[13], 2) - std::pow(d[12], 2) +
                2 * d[12] * d[13] - std::pow(d[13], 2);
    coeffs[6] = std::pow(d[6], 2) - 2 * d[6] * d[8] + std::pow(d[8], 2) + std::pow(d[9], 2) - 2 * d[9] * d[11] +
                std::pow(d[11], 2);
    coeffs[7] = -std::pow(d[0], 2) + 2 * d[0] * d[2] - std::pow(d[2], 2) - std::pow(d[3], 2) + 2 * d[3] * d[5] -
                std::pow(d[5], 2);
    coeffs[8] = 2 * std::pow(d[6], 2) * d[15] - 2 * d[6] * d[8] * d[15] + 2 * std::pow(d[9], 2) * d[15] -
                2 * d[9] * d[11] * d[15] - 2 * d[6] * d[8] * d[17] + 2 * std::pow(d[8], 2) * d[17] -
                2 * d[9] * d[11] * d[17] + 2 * std::pow(d[11], 2) * d[17];
    coeffs[9] = std::pow(d[6], 2) * std::pow(d[15], 2) + std::pow(d[9], 2) * std::pow(d[15], 2) -
                2 * d[6] * d[8] * d[15] * d[17] - 2 * d[9] * d[11] * d[15] * d[17] +
                std::pow(d[8], 2) * std::pow(d[17], 2) + std::pow(d[11], 2) * std::pow(d[17], 2) + std::pow(d[15], 2) -
                2 * d[15] * d[17] + std::pow(d[17], 2);
    coeffs[10] = -2 * std::pow(d[0], 2) * d[12] + 2 * d[0] * d[2] * d[12] - 2 * std::pow(d[3], 2) * d[12] +
                 2 * d[3] * d[5] * d[12] + 2 * d[0] * d[2] * d[14] - 2 * std::pow(d[2], 2) * d[14] +
                 2 * d[3] * d[5] * d[14] - 2 * std::pow(d[5], 2) * d[14];
    coeffs[11] = -std::pow(d[0], 2) * std::pow(d[12], 2) - std::pow(d[3], 2) * std::pow(d[12], 2) +
                 2 * d[0] * d[2] * d[12] * d[14] + 2 * d[3] * d[5] * d[12] * d[14] -
                 std::pow(d[2], 2) * std::pow(d[14], 2) - std::pow(d[5], 2) * std::pow(d[14], 2) - std::pow(d[12], 2) +
                 2 * d[12] * d[14] - std::pow(d[14], 2);
    coeffs[12] = std::pow(d[7], 2) - 2 * d[7] * d[8] + std::pow(d[8], 2) + std::pow(d[10], 2) - 2 * d[10] * d[11] +
                 std::pow(d[11], 2);
    coeffs[13] = -std::pow(d[1], 2) + 2 * d[1] * d[2] - std::pow(d[2], 2) - std::pow(d[4], 2) + 2 * d[4] * d[5] -
                 std::pow(d[5], 2);
    coeffs[14] = 2 * std::pow(d[7], 2) * d[16] - 2 * d[7] * d[8] * d[16] + 2 * std::pow(d[10], 2) * d[16] -
                 2 * d[10] * d[11] * d[16] - 2 * d[7] * d[8] * d[17] + 2 * std::pow(d[8], 2) * d[17] -
                 2 * d[10] * d[11] * d[17] + 2 * std::pow(d[11], 2) * d[17];
    coeffs[15] = std::pow(d[7], 2) * std::pow(d[16], 2) + std::pow(d[10], 2) * std::pow(d[16], 2) -
                 2 * d[7] * d[8] * d[16] * d[17] - 2 * d[10] * d[11] * d[16] * d[17] +
                 std::pow(d[8], 2) * std::pow(d[17], 2) + std::pow(d[11], 2) * std::pow(d[17], 2) + std::pow(d[16], 2) -
                 2 * d[16] * d[17] + std::pow(d[17], 2);
    coeffs[16] = -2 * std::pow(d[1], 2) * d[13] + 2 * d[1] * d[2] * d[13] - 2 * std::pow(d[4], 2) * d[13] +
                 2 * d[4] * d[5] * d[13] + 2 * d[1] * d[2] * d[14] - 2 * std::pow(d[2], 2) * d[14] +
                 2 * d[4] * d[5] * d[14] - 2 * std::pow(d[5], 2) * d[14];
    coeffs[17] = -std::pow(d[1], 2) * std::pow(d[13], 2) - std::pow(d[4], 2) * std::pow(d[13], 2) +
                 2 * d[1] * d[2] * d[13] * d[14] + 2 * d[4] * d[5] * d[13] * d[14] -
                 std::pow(d[2], 2) * std::pow(d[14], 2) - std::pow(d[5], 2) * std::pow(d[14], 2) - std::pow(d[13], 2) +
                 2 * d[13] * d[14] - std::pow(d[14], 2);

    Eigen::MatrixXd C0(3, 3);
    C0 << coeffs[0], coeffs[2], coeffs[3], coeffs[6], coeffs[8], coeffs[9], coeffs[12], coeffs[14], coeffs[15];

    Eigen::MatrixXd C1(3, 3);
    C1 << coeffs[1], coeffs[4], coeffs[5], coeffs[7], coeffs[10], coeffs[11], coeffs[13], coeffs[16], coeffs[17];

    Eigen::MatrixXd C2 = -C0.fullPivLu().solve(C1);

    double k0 = C2(0, 0);
    double k1 = C2(0, 1);
    double k2 = C2(0, 2);
    double k3 = C2(1, 0);
    double k4 = C2(1, 1);
    double k5 = C2(1, 2);
    double k6 = C2(2, 0);
    double k7 = C2(2, 1);
    double k8 = C2(2, 2);

    double c4 = 1.0 / (k3 * k3 - k0 * k6);
    double c3 = c4 * (2 * k3 * k4 - k1 * k6 - k0 * k7);
    double c2 = c4 * (k4 * k4 - k0 * k8 - k1 * k7 - k2 * k6 + 2 * k3 * k5);
    double c1 = c4 * (2 * k4 * k5 - k2 * k7 - k1 * k8);
    double c0 = c4 * (k5 * k5 - k2 * k8);

    double roots[4];
    int n_roots = univariate::solve_quartic_real(c3, c2, c1, c0, roots);
    int m = 0;
    Eigen::MatrixXd sols(3, n_roots);
    for (int ii = 0; ii < n_roots; ii++) {
        double ss = k6 * roots[ii] * roots[ii] + k7 * roots[ii] + k8;
        if (ss < 0.001)
            continue;
        sols(1, ii) = roots[ii];
        sols(0, ii) = std::sqrt(ss);
        sols(2, ii) = (k3 * roots[ii] * roots[ii] + k4 * roots[ii] + k5) / ss;
        ++m;
    }
    sols.conservativeResize(3, m);
    return {sols, coeffs};
}

inline void refine_suv(double &s, double &u, double &v, const Eigen::VectorXd c) {
    for (int iter = 0; iter < 5; ++iter) {
        Eigen::Vector3d r;
        r(0) = c(0) * s * v * v + c(1) * u * u + c(2) * s * v + c(3) * s + c(4) * u + c(5);
        r(1) = c(6) * s * v * v + c(7) * u * u + c(8) * s * v + c(9) * s + c(10) * u + c(11);
        r(2) = c(12) * s * v * v + c(13) * u * u + c(14) * s * v + c(15) * s + c(16) * u + c(17);
        if (std::abs(r(0)) + std::abs(r(1)) + std::abs(r(2)) < 1e-10)
            return;
        Eigen::Matrix3d J;
        J(0, 0) = c(0) * v * v + c(2) * v + c(3);
        J(0, 1) = 2.0 * c(1) * u + c(4);
        J(0, 2) = 2.0 * c(0) * s * v + c(2) * s;

        J(1, 0) = c(6) * v * v + c(8) * v + c(9);
        J(1, 1) = 2.0 * c(7) * u + c(10);
        J(1, 2) = 2.0 * c(6) * s * v + c(8) * s;

        J(2, 0) = c(12) * v * v + c(14) * v + c(15);
        J(2, 1) = 2.0 * c(13) * u + c(16);
        J(2, 2) = 2.0 * c(12) * s * v + c(14) * s;

        Eigen::Vector3d delta_lambda = (J.transpose() * J).ldlt().solve(-J.transpose() * r);

        s += delta_lambda(0);
        u += delta_lambda(1);
        v += delta_lambda(2);
    }
}

int relpose_monodepth_3pt(const std::vector<Eigen::Vector3d> &x1h, const std::vector<Eigen::Vector3d> &x2h,
                          const std::vector<double> &depth1, const std::vector<double> &depth2,
                          std::vector<MonoDepthTwoViewGeometry> *rel_pose) {
    rel_pose->clear();
    rel_pose->reserve(4);

    Eigen::VectorXd datain(18);
    datain << x1h[0][0], x1h[1][0], x1h[2][0], x1h[0][1], x1h[1][1], x1h[2][1], x2h[0][0], x2h[1][0], x2h[2][0],
        x2h[0][1], x2h[1][1], x2h[2][1], depth1[0], depth1[1], depth1[2], depth2[0], depth2[1], depth2[2];

    auto [sols, cc] = solver_p3p_mono_3d(datain);
    size_t num_sols = 0;
    for (int k = 0; k < sols.cols(); ++k) {

        double s = sols(0, k);
        double u = sols(1, k);
        double v = sols(2, k);

        if (depth2[0] + v <= 0 || depth2[1] + v <= 0 || depth2[2] + v <= 0 || depth1[0] + u <= 0 ||
            depth1[1] + u <= 0 || depth1[2] + u <= 0)
            continue;

        double s2 = s * s;
        refine_suv(s2, u, v, cc);
        s = std::sqrt(s2);

        Eigen::Vector3d v1 = s * (depth2[0] + v) * x2h[0] - s * (depth2[1] + v) * x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0] + v) * x2h[0] - s * (depth2[2] + v) * x2h[2];

        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0] + u) * x1h[0] - (depth1[1] + u) * x1h[1];
        Eigen::Vector3d u2 = (depth1[0] + u) * x1h[0] - (depth1[2] + u) * x1h[2];

        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;
        Eigen::Vector3d t = s * (depth2[0] + v) * x2h[0] - (depth1[0] + u) * rot * x1h[0];

        MonoDepthTwoViewGeometry pose = MonoDepthTwoViewGeometry(rot, t, s);

        pose.shift1 = u;
        pose.shift2 = v;

        rel_pose->emplace_back(pose);
        num_sols++;
    }

    return num_sols;
}
} // namespace poselib
