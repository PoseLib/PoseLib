#include "relpose_monodepth_3pt_shared_focal.h"

namespace poselib {
int relpose_monodepth_3pt_shared_focal(const std::vector<Eigen::Vector3d> &x1h, const std::vector<Eigen::Vector3d> &x2h,
                                       const std::vector<double> &depth1, const std::vector<double> &depth2,
                                       std::vector<MonoDepthImagePair> *models) {
    models->clear();
    models->reserve(4);

    Eigen::Matrix3d X1;
    X1.col(0) = depth1[0] * x1h[0];
    X1.col(1) = depth1[1] * x1h[1];
    X1.col(2) = depth1[2] * x1h[2];

    Eigen::Matrix3d X2;
    X2.col(0) = depth2[0] * x2h[0];
    X2.col(1) = depth2[1] * x2h[1];
    X2.col(2) = x2h[2];

    double a[17];

    a[0] = X1(0, 0);
    a[1] = X1(0, 1);
    a[2] = X1(0, 2);
    a[3] = X1(1, 0);
    a[4] = X1(1, 1);
    a[5] = X1(1, 2);
    a[6] = X1(2, 0);
    a[7] = X1(2, 1);
    a[8] = X1(2, 2);

    a[9] = X2(0, 0);
    a[10] = X2(0, 1);
    a[11] = X2(0, 2);
    a[12] = X2(1, 0);
    a[13] = X2(1, 1);
    a[14] = X2(1, 2);
    a[15] = X2(2, 0);
    a[16] = X2(2, 1);

    double b[12];
    b[0] = a[0] - a[1];
    b[1] = a[3] - a[4];
    b[2] = a[6] - a[7];
    b[3] = a[0] - a[2];
    b[4] = a[3] - a[5];
    b[5] = a[6] - a[8];
    b[6] = a[1] - a[2];
    b[7] = a[4] - a[5];
    b[8] = a[7] - a[8];
    b[9] = a[9] - a[10];
    b[10] = a[12] - a[13];
    b[11] = a[15] - a[16];

    double c[18];
    c[0] = -std::pow(b[11], 2);
    c[1] = std::pow(b[2], 2);
    c[2] = -std::pow(b[9], 2) - std::pow(b[10], 2);
    c[3] = std::pow(b[0], 2) + std::pow(b[1], 2);

    c[4] = -1.0;
    c[5] = 2 * a[15];
    c[6] = -std::pow(a[15], 2);
    c[7] = std::pow(b[5], 2);
    c[8] = -std::pow(a[11], 2) - std::pow(a[14], 2);
    c[9] = 2 * a[9] * a[11] + 2 * a[12] * a[14];
    c[10] = -std::pow(a[9], 2) - std::pow(a[12], 2);
    c[11] = std::pow(b[3], 2) + std::pow(b[4], 2);

    c[12] = 2 * a[16] - 2 * a[15];
    c[13] = std::pow(a[15], 2) - std::pow(a[16], 2);
    c[14] = std::pow(b[8], 2) - std::pow(b[5], 2);
    c[15] = 2 * a[10] * a[11] - 2 * a[9] * a[11] - 2 * a[12] * a[14] + 2 * a[13] * a[14];
    c[16] = std::pow(a[9], 2) - std::pow(a[10], 2) + std::pow(a[12], 2) - std::pow(a[13], 2);
    c[17] = -std::pow(b[3], 2) - std::pow(b[4], 2) + std::pow(b[6], 2) + std::pow(b[7], 2);

    double d[21];

    d[6] = 1 / (a[6] - a[7]);
    d[0] = (-c[3] * c[8]) * d[6];
    d[1] = (-c[3] * c[9]) * d[6];
    d[2] = (c[2] * c[11] - c[3] * c[10]) * d[6];
    d[3] = (-c[3] * c[4] - c[1] * c[8]) * d[6];
    d[4] = (-c[3] * c[5] - c[1] * c[9]) * d[6];
    d[5] = (c[2] * c[7] - c[3] * c[6] + c[0] * c[11] - c[1] * c[10]) * d[6];
    d[7] = (a[6] * a[16] - 2 * a[6] * a[15] + a[7] * a[15] + a[8] * a[15] - a[8] * a[16]) * d[6];

    d[8] = 1 / (2 * (a[6] - a[7]) * (a[15] - a[16]));
    d[9] = (-c[3] * c[15]) * d[8];
    d[10] = (c[2] * c[17] - c[3] * c[16]) * d[8];
    d[11] = (-c[3] * c[12] - c[1] * c[15]) * d[8];
    d[12] = (c[2] * c[14] - c[3] * c[13] + c[0] * c[17] - c[1] * c[16]) * d[8];

    d[13] = 1 / (a[6] + a[7] - 2 * a[8]);
    d[14] = (a[8] * a[15] - a[7] * a[15] - a[6] * a[16] + a[8] * a[16]) * d[13];
    d[15] = (c[8] * c[17]) * d[13];
    d[16] = (c[9] * c[17] - c[11] * c[15]) * d[13];
    d[17] = (c[10] * c[17] - c[11] * c[16]) * d[13];
    d[18] = (c[4] * c[17] + c[8] * c[14]) * d[13];
    d[19] = (c[5] * c[17] - c[7] * c[15] + c[9] * c[14] - c[11] * c[12]) * d[13];
    d[20] = (c[6] * c[17] - c[7] * c[16] + c[10] * c[14] - c[11] * c[13]) * d[13];

    Eigen::MatrixXd C0(3, 3);
    C0 << d[2], d[5], d[7], d[10], d[12], 1.0, d[17], d[20], d[14];

    Eigen::MatrixXd C1(3, 4);
    C1 << d[0] - d[9], d[3] - d[11], d[1] - d[10], d[4] - d[12], 0, 0, d[9], d[11], d[15] - d[9], d[18] - d[11],
        d[16] - d[10], d[19] - d[12];

    Eigen::MatrixXd C2 = -C0.partialPivLu().solve(C1);

    Eigen::MatrixXd AM(4, 4);
    AM << 0, 0, 1.0, 0, 0, 0, 0, 1.0, C2(0, 0), C2(0, 1), C2(0, 2), C2(0, 3), C2(1, 0), C2(1, 1), C2(1, 2), C2(1, 3);

    Eigen::EigenSolver<Eigen::Matrix<double, 4, 4>> es(AM, false);
    Eigen::ArrayXcd D = es.eigenvalues();

    for (int k = 0; k < 4; ++k) {

        if (abs(D(k).imag()) > 0.001 || D(k).real() < 0.0)
            continue;

        double d3 = 1.0 / D(k).real();

        Eigen::MatrixXd A0(2, 2);
        A0 << (d[3] - d[11]) * d3 * d3 + (d[4] - d[12]) * d3 + d[5], d[7], d[12] + d[11] * d3, 1.0;

        Eigen::VectorXd A1(2);
        A1 << (d[0] - d[9]) * d3 * d3 + (d[1] - d[10]) * d3 + d[2], d[10] + d[9] * d3;
        Eigen::VectorXd A2 = -A0.partialPivLu().solve(A1);

        if (A2(0) < 0.0)
            continue;

        double s2 = -(c[1] * A2(0) + c[3]) / (c[0] * A2(0) + c[2]);
        if (s2 < 0.001)
            continue;

        double s = std::sqrt(s2);
        double f = std::sqrt(A2(0));

        Eigen::Matrix3d Kinv;
        Kinv << 1.0 / f, 0, 0, 0, 1.0 / f, 0, 0, 0, 1;

        Eigen::Vector3d v1 = s * (depth2[0]) * Kinv * x2h[0] - s * (depth2[1]) * Kinv * x2h[1];
        Eigen::Vector3d v2 = s * (depth2[0]) * Kinv * x2h[0] - s * (d3)*Kinv * x2h[2];
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0]) * Kinv * x1h[0] - (depth1[1]) * Kinv * x1h[1];
        Eigen::Vector3d u2 = (depth1[0]) * Kinv * x1h[0] - (depth1[2]) * Kinv * x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0]) * rot * Kinv * x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0]) * Kinv * x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        MonoDepthTwoViewGeometry pose = MonoDepthTwoViewGeometry(rot, trans, s);
        Camera camera = Camera(SimplePinholeCameraModel::model_id, std::vector<double>{f, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera, camera);
    }

    return models->size();
}
} // namespace poselib
