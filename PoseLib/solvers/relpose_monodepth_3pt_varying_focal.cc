#include "relpose_monodepth_3pt_varying_focal.h"

namespace poselib {
int relpose_monodepth_3pt_varying_focal(const std::vector<Eigen::Vector3d> &x1h,
                                        const std::vector<Eigen::Vector3d> &x2h, const std::vector<double> &depth1,
                                        const std::vector<double> &depth2, std::vector<MonoDepthImagePair> *models) {
    models->clear();
    models->reserve(1);

    double a[18];
    a[0] = x1h[0][0] * depth1[0];
    a[1] = x1h[1][0] * depth1[1];
    a[2] = x1h[2][0] * depth1[2];
    a[3] = x1h[0][1] * depth1[0];
    a[4] = x1h[1][1] * depth1[1];
    a[5] = x1h[2][1] * depth1[2];
    a[6] = depth1[0];
    a[7] = depth1[1];
    a[8] = depth1[2];

    a[9] = x2h[0][0] * depth2[0];
    a[10] = x2h[1][0] * depth2[1];
    a[11] = x2h[2][0] * depth2[2];
    a[12] = x2h[0][1] * depth2[0];
    a[13] = x2h[1][1] * depth2[1];
    a[14] = x2h[2][1] * depth2[2];
    a[15] = depth2[0];
    a[16] = depth2[1];
    a[17] = depth2[2];

    double b[18];
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
    b[12] = a[9] - a[11];
    b[13] = a[12] - a[14];
    b[14] = a[15] - a[17];
    b[15] = a[10] - a[11];
    b[16] = a[13] - a[14];
    b[17] = a[16] - a[17];

    Eigen::Matrix3d A;
    A << std::pow(b[0], 2) + std::pow(b[1], 2), -std::pow(b[9], 2) - std::pow(b[10], 2), -std::pow(b[11], 2),
        std::pow(b[3], 2) + std::pow(b[4], 2), -std::pow(b[12], 2) - std::pow(b[13], 2), -std::pow(b[14], 2),
        std::pow(b[6], 2) + std::pow(b[7], 2), -std::pow(b[15], 2) - std::pow(b[16], 2), -std::pow(b[17], 2);
    Eigen::Vector3d B;
    B << b[2] * b[2], b[5] * b[5], b[8] * b[8];
    Eigen::Vector3d sol = -A.partialPivLu().solve(B);

    if (sol(0) > 0 && sol(1) > 0 && sol(2) > 0) {
        double f = std::sqrt(sol(0));
        double s = std::sqrt(sol(2));
        double w = std::sqrt(sol(1) / sol(2));

        Eigen::Matrix3d K1inv;
        K1inv << f, 0, 0, 0, f, 0, 0, 0, 1;

        Eigen::Matrix3d K2inv;
        K2inv << w, 0, 0, 0, w, 0, 0, 0, 1;

        Eigen::Vector3d v1 = s * ((depth2[0]) * K2inv * x2h[0] - (depth2[1]) * K2inv * x2h[1]);
        Eigen::Vector3d v2 = s * ((depth2[0]) * K2inv * x2h[0] - (depth2[2]) * K2inv * x2h[2]);
        Eigen::Matrix3d Y;
        Y << v1, v2, v1.cross(v2);

        Eigen::Vector3d u1 = (depth1[0]) * K1inv * x1h[0] - (depth1[1]) * K1inv * x1h[1];
        Eigen::Vector3d u2 = (depth1[0]) * K1inv * x1h[0] - (depth1[2]) * K1inv * x1h[2];
        Eigen::Matrix3d X;
        X << u1, u2, u1.cross(u2);
        X = X.inverse().eval();

        Eigen::Matrix3d rot = Y * X;

        Eigen::Vector3d trans1 = (depth1[0]) * rot * K1inv * x1h[0];
        Eigen::Vector3d trans2 = s * (depth2[0]) * K2inv * x2h[0];
        Eigen::Vector3d trans = trans2 - trans1;

        double focal1 = 1.0 / f;
        double focal2 = 1.0 / w;

        MonoDepthTwoViewGeometry pose = MonoDepthTwoViewGeometry(rot, trans, s);
        Camera camera1 = Camera(SimplePinholeCameraModel::model_id, std::vector<double>{focal1, 0.0, 0.0}, -1, -1);
        Camera camera2 = Camera(SimplePinholeCameraModel::model_id, std::vector<double>{focal2, 0.0, 0.0}, -1, -1);
        models->emplace_back(pose, camera1, camera2);
    }

    return models->size();
}
} // namespace poselib
