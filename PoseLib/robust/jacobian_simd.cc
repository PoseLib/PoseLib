#include "PoseLib/robust/jacobian_simd.h"

#include "PoseLib/camera_pose.h"
#include "PoseLib/misc/colmap_models.h"
#include "PoseLib/misc/essential.h"
#include "PoseLib/robust/jacobian_impl.h"
#include "PoseLib/robust/robust_loss.h"
#include "PoseLib/types.h"

namespace poselib {

static double sum_pd(__m256d v) {
    double tmp[4];
    _mm256_storeu_pd(tmp, v);
    return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

template <>
size_t CameraJacobianAccumulatorSIMD<NullCameraModel, TruncatedLoss, UniformWeightVector>::accumulate(
    const CameraPose &pose, Eigen::Matrix<double, 6, 6> &JtJ, Eigen::Matrix<double, 6, 1> &Jtr) const {
    Eigen::Matrix3d Rmat = pose.R();
    const __m256d R[3][3]{{
                              _mm256_set1_pd(Rmat(0, 0)),
                              _mm256_set1_pd(Rmat(0, 1)),
                              _mm256_set1_pd(Rmat(0, 2)),
                          },
                          {
                              _mm256_set1_pd(Rmat(1, 0)),
                              _mm256_set1_pd(Rmat(1, 1)),
                              _mm256_set1_pd(Rmat(1, 2)),
                          },
                          {
                              _mm256_set1_pd(Rmat(2, 0)),
                              _mm256_set1_pd(Rmat(2, 1)),
                              _mm256_set1_pd(Rmat(2, 2)),
                          }};
    const __m256d t[3]{_mm256_set1_pd(pose.t(0)), _mm256_set1_pd(pose.t(1)), _mm256_set1_pd(pose.t(2))};

    size_t num_residuals = 0;
    size_t K = x.rows();
    const __m256d sq_thresholds = _mm256_set1_pd(loss_fn.squared_thr);

    __m256d JtJ_sum00 = _mm256_setzero_pd(), JtJ_sum10 = _mm256_setzero_pd(), JtJ_sum11 = _mm256_setzero_pd(),
            JtJ_sum20 = _mm256_setzero_pd(), JtJ_sum21 = _mm256_setzero_pd(), JtJ_sum22 = _mm256_setzero_pd(),
            JtJ_sum30 = _mm256_setzero_pd(), JtJ_sum31 = _mm256_setzero_pd(), JtJ_sum32 = _mm256_setzero_pd(),
            JtJ_sum33 = _mm256_setzero_pd(), JtJ_sum40 = _mm256_setzero_pd(), JtJ_sum41 = _mm256_setzero_pd(),
            JtJ_sum42 = _mm256_setzero_pd(), JtJ_sum43 = _mm256_setzero_pd(), JtJ_sum44 = _mm256_setzero_pd(),
            JtJ_sum50 = _mm256_setzero_pd(), JtJ_sum51 = _mm256_setzero_pd(), JtJ_sum52 = _mm256_setzero_pd(),
            JtJ_sum53 = _mm256_setzero_pd(), JtJ_sum54 = _mm256_setzero_pd(), JtJ_sum55 = _mm256_setzero_pd();

    __m256d Jtr_sum[6] = {_mm256_setzero_pd()};

    const double *X_p = X.data();
    const double *x_p = x.data();
    for (size_t k = 0; k < K - 3; k += 4) {

        const __m256d X[3]{
            _mm256_loadu_pd(X_p + k),
            _mm256_loadu_pd(X_p + K + k),
            _mm256_loadu_pd(X_p + 2 * K + k),
        };

        __m256d Z[3]{
            _mm256_fmadd_pd(R[0][0], X[0], t[0]),
            _mm256_fmadd_pd(R[1][0], X[0], t[1]),
            _mm256_fmadd_pd(R[2][0], X[0], t[2]),
        };
        for (size_t i = 1; i < 3; ++i) {
            Z[0] = _mm256_fmadd_pd(R[0][i], X[i], Z[0]);
            Z[1] = _mm256_fmadd_pd(R[1][i], X[i], Z[1]);
            Z[2] = _mm256_fmadd_pd(R[2][i], X[i], Z[2]);
        }

        const __m256d valid_z_mask = _mm256_cmp_pd(Z[2], _mm256_setzero_pd(), _CMP_GT_OQ);

        if (_mm256_testz_pd(valid_z_mask, valid_z_mask)) {
            continue;
        }

        const __m256d inv_z = _mm256_div_pd(_mm256_set1_pd(1.0), Z[2]);
        const __m256d z[2]{_mm256_mul_pd(Z[0], inv_z), _mm256_mul_pd(Z[1], inv_z)};

        const __m256d r[2]{_mm256_sub_pd(z[0], _mm256_loadu_pd(x_p + k)),
                           _mm256_sub_pd(z[1], _mm256_loadu_pd(x_p + K + k))};
        const __m256d r_squared = _mm256_add_pd(_mm256_mul_pd(r[0], r[0]), _mm256_mul_pd(r[1], r[1]));
        const __m256d valid_mask = _mm256_and_pd(valid_z_mask, _mm256_cmp_pd(r_squared, sq_thresholds, _CMP_LT_OQ));

        if (_mm256_testz_pd(valid_mask, valid_mask)) {
            continue;
        }

        num_residuals += _mm_popcnt_u32(_mm256_movemask_pd(valid_mask));

        const __m256d dZ[2][3]{{
                                   _mm256_mul_pd(inv_z, _mm256_sub_pd(R[0][0], _mm256_mul_pd(z[0], R[2][0]))),
                                   _mm256_mul_pd(inv_z, _mm256_sub_pd(R[0][1], _mm256_mul_pd(z[0], R[2][1]))),
                                   _mm256_mul_pd(inv_z, _mm256_sub_pd(R[0][2], _mm256_mul_pd(z[0], R[2][2]))),
                               },
                               {
                                   _mm256_mul_pd(inv_z, _mm256_sub_pd(R[1][0], _mm256_mul_pd(z[1], R[2][0]))),
                                   _mm256_mul_pd(inv_z, _mm256_sub_pd(R[1][1], _mm256_mul_pd(z[1], R[2][1]))),
                                   _mm256_mul_pd(inv_z, _mm256_sub_pd(R[1][2], _mm256_mul_pd(z[1], R[2][2]))),
                               }};

        const __m256d dZtdZ_0_0 = _mm256_and_pd(
            valid_mask, _mm256_add_pd(_mm256_mul_pd(dZ[0][0], dZ[0][0]), _mm256_mul_pd(dZ[1][0], dZ[1][0])));
        const __m256d dZtdZ_1_0 = _mm256_and_pd(
            valid_mask, _mm256_add_pd(_mm256_mul_pd(dZ[0][1], dZ[0][0]), _mm256_mul_pd(dZ[1][1], dZ[1][0])));
        const __m256d dZtdZ_1_1 = _mm256_and_pd(
            valid_mask, _mm256_add_pd(_mm256_mul_pd(dZ[0][1], dZ[0][1]), _mm256_mul_pd(dZ[1][1], dZ[1][1])));
        const __m256d dZtdZ_2_0 = _mm256_and_pd(
            valid_mask, _mm256_add_pd(_mm256_mul_pd(dZ[0][2], dZ[0][0]), _mm256_mul_pd(dZ[1][2], dZ[1][0])));
        const __m256d dZtdZ_2_1 = _mm256_and_pd(
            valid_mask, _mm256_add_pd(_mm256_mul_pd(dZ[0][2], dZ[0][1]), _mm256_mul_pd(dZ[1][2], dZ[1][1])));
        const __m256d dZtdZ_2_2 = _mm256_and_pd(
            valid_mask, _mm256_add_pd(_mm256_mul_pd(dZ[0][2], dZ[0][2]), _mm256_mul_pd(dZ[1][2], dZ[1][2])));

        const __m256d XZZ021 = _mm256_mul_pd(X[0], dZtdZ_2_1);
        const __m256d XZZ210 = _mm256_mul_pd(X[2], dZtdZ_1_0);
        const __m256d XZZ120 = _mm256_mul_pd(X[1], dZtdZ_2_0);

        const __m256d JtJ_30 = _mm256_sub_pd(XZZ120, XZZ210);
        const __m256d JtJ_41 = _mm256_sub_pd(XZZ210, XZZ021);
        const __m256d JtJ_52 = _mm256_sub_pd(XZZ021, XZZ120);

        JtJ_sum30 += JtJ_30;
        JtJ_sum41 += JtJ_41;
        JtJ_sum52 += JtJ_52;

        const __m256d JtJ_40 = _mm256_sub_pd(_mm256_mul_pd(X[1], dZtdZ_2_1), _mm256_mul_pd(X[2], dZtdZ_1_1));
        const __m256d JtJ_50 = _mm256_sub_pd(_mm256_mul_pd(X[1], dZtdZ_2_2), _mm256_mul_pd(X[2], dZtdZ_2_1));
        const __m256d JtJ_31 = _mm256_sub_pd(_mm256_mul_pd(X[2], dZtdZ_0_0), _mm256_mul_pd(X[0], dZtdZ_2_0));
        const __m256d JtJ_51 = _mm256_sub_pd(_mm256_mul_pd(X[2], dZtdZ_2_0), _mm256_mul_pd(X[0], dZtdZ_2_2));
        const __m256d JtJ_32 = _mm256_sub_pd(_mm256_mul_pd(X[0], dZtdZ_1_0), _mm256_mul_pd(X[1], dZtdZ_0_0));
        const __m256d JtJ_42 = _mm256_sub_pd(_mm256_mul_pd(X[0], dZtdZ_1_1), _mm256_mul_pd(X[1], dZtdZ_1_0));

        JtJ_sum40 += JtJ_40;
        JtJ_sum50 += JtJ_50;
        JtJ_sum31 += JtJ_31;
        JtJ_sum51 += JtJ_51;
        JtJ_sum32 += JtJ_32;
        JtJ_sum42 += JtJ_42;

        JtJ_sum00 += _mm256_sub_pd(_mm256_mul_pd(X[1], JtJ_50), _mm256_mul_pd(X[2], JtJ_40));
        JtJ_sum10 += _mm256_sub_pd(_mm256_mul_pd(X[1], JtJ_51), _mm256_mul_pd(X[2], JtJ_41));
        JtJ_sum20 += _mm256_sub_pd(_mm256_mul_pd(X[1], JtJ_52), _mm256_mul_pd(X[2], JtJ_42));
        JtJ_sum11 += _mm256_sub_pd(_mm256_mul_pd(X[2], JtJ_31), _mm256_mul_pd(X[0], JtJ_51));
        JtJ_sum21 += _mm256_sub_pd(_mm256_mul_pd(X[2], JtJ_32), _mm256_mul_pd(X[0], JtJ_52));
        JtJ_sum22 += _mm256_sub_pd(_mm256_mul_pd(X[0], JtJ_42), _mm256_mul_pd(X[1], JtJ_32));

        JtJ_sum33 += dZtdZ_0_0;
        JtJ_sum43 += dZtdZ_1_0;
        JtJ_sum53 += dZtdZ_2_0;
        JtJ_sum44 += dZtdZ_1_1;
        JtJ_sum54 += dZtdZ_2_1;
        JtJ_sum55 += dZtdZ_2_2;

        // Accumulate into JtJ and Jtr

        const __m256d masked_r[2]{_mm256_and_pd(valid_mask, r[0]), _mm256_and_pd(valid_mask, r[1])};

        Jtr_sum[0] = _mm256_add_pd(
            Jtr_sum[0], _mm256_add_pd(_mm256_mul_pd(masked_r[0], _mm256_sub_pd(_mm256_mul_pd(X[1], dZ[0][2]),
                                                                               _mm256_mul_pd(X[2], dZ[0][1]))),
                                      _mm256_mul_pd(masked_r[1], _mm256_sub_pd(_mm256_mul_pd(X[1], dZ[1][2]),
                                                                               _mm256_mul_pd(X[2], dZ[1][1])))));

        Jtr_sum[1] = _mm256_sub_pd(
            Jtr_sum[1], _mm256_add_pd(_mm256_mul_pd(masked_r[0], _mm256_sub_pd(_mm256_mul_pd(X[0], dZ[0][2]),
                                                                               _mm256_mul_pd(X[2], dZ[0][0]))),
                                      _mm256_mul_pd(masked_r[1], _mm256_sub_pd(_mm256_mul_pd(X[0], dZ[1][2]),
                                                                               _mm256_mul_pd(X[2], dZ[1][0])))));

        Jtr_sum[2] = _mm256_add_pd(
            Jtr_sum[2], _mm256_add_pd(_mm256_mul_pd(masked_r[0], _mm256_sub_pd(_mm256_mul_pd(X[0], dZ[0][1]),
                                                                               _mm256_mul_pd(X[1], dZ[0][0]))),
                                      _mm256_mul_pd(masked_r[1], _mm256_sub_pd(_mm256_mul_pd(X[0], dZ[1][1]),
                                                                               _mm256_mul_pd(X[1], dZ[1][0])))));

        Jtr_sum[3] = _mm256_add_pd(
            Jtr_sum[3], _mm256_add_pd(_mm256_mul_pd(dZ[0][0], masked_r[0]), _mm256_mul_pd(dZ[1][0], masked_r[1])));
        Jtr_sum[4] = _mm256_add_pd(
            Jtr_sum[4], _mm256_add_pd(_mm256_mul_pd(dZ[0][1], masked_r[0]), _mm256_mul_pd(dZ[1][1], masked_r[1])));
        Jtr_sum[5] = _mm256_add_pd(
            Jtr_sum[5], _mm256_add_pd(_mm256_mul_pd(dZ[0][2], masked_r[0]), _mm256_mul_pd(dZ[1][2], masked_r[1])));
    }

    JtJ(0, 0) += sum_pd(JtJ_sum00);
    JtJ(1, 0) += sum_pd(JtJ_sum10);
    JtJ(1, 1) += sum_pd(JtJ_sum11);
    JtJ(2, 0) += sum_pd(JtJ_sum20);
    JtJ(2, 1) += sum_pd(JtJ_sum21);
    JtJ(2, 2) += sum_pd(JtJ_sum22);
    JtJ(3, 0) += sum_pd(JtJ_sum30);
    JtJ(3, 1) += sum_pd(JtJ_sum31);
    JtJ(3, 2) += sum_pd(JtJ_sum32);
    JtJ(3, 3) += sum_pd(JtJ_sum33);
    JtJ(4, 0) += sum_pd(JtJ_sum40);
    JtJ(4, 1) += sum_pd(JtJ_sum41);
    JtJ(4, 2) += sum_pd(JtJ_sum42);
    JtJ(4, 3) += sum_pd(JtJ_sum43);
    JtJ(4, 4) += sum_pd(JtJ_sum44);
    JtJ(5, 0) += sum_pd(JtJ_sum50);
    JtJ(5, 1) += sum_pd(JtJ_sum51);
    JtJ(5, 2) += sum_pd(JtJ_sum52);
    JtJ(5, 3) += sum_pd(JtJ_sum53);
    JtJ(5, 4) += sum_pd(JtJ_sum54);
    JtJ(5, 5) += sum_pd(JtJ_sum55);

    for (size_t i = 0; i < 6; ++i) {
        Jtr(i) += sum_pd(Jtr_sum[i]);
    }

    // Handle the remaining points
    for (size_t k = K - K % 4; k < K; ++k) {
        const Eigen::Vector3d Z = (Rmat * X.row(k).transpose() + pose.t);
        const Eigen::Vector2d z = Z.hnormalized();
        if (Z(2) < 0.0)
            continue;

        Eigen::Vector2d r = z - x.row(k).transpose();
        const double r_squared = r.squaredNorm();
        const double weight = loss_fn.weight(r_squared);
        if (weight == 0.0)
            continue;

        num_residuals++;
        Eigen::Matrix<double, 2, 3> dZ;
        dZ.block<2, 2>(0, 0).setIdentity();
        dZ.col(2) = -z;
        dZ *= 1.0 / Z(2);
        dZ *= Rmat;
        const double X0 = X(k, 0);
        const double X1 = X(k, 1);
        const double X2 = X(k, 2);
        const double dZtdZ_0_0 = dZ.col(0).dot(dZ.col(0));
        const double dZtdZ_1_0 = dZ.col(1).dot(dZ.col(0));
        const double dZtdZ_1_1 = dZ.col(1).dot(dZ.col(1));
        const double dZtdZ_2_0 = dZ.col(2).dot(dZ.col(0));
        const double dZtdZ_2_1 = dZ.col(2).dot(dZ.col(1));
        const double dZtdZ_2_2 = dZ.col(2).dot(dZ.col(2));
        JtJ(0, 0) += X2 * (X2 * dZtdZ_1_1 - X1 * dZtdZ_2_1) + X1 * (X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1);
        JtJ(1, 0) += -X2 * (X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1) - X1 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
        JtJ(2, 0) += X1 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0) - X2 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
        JtJ(3, 0) += X1 * dZtdZ_2_0 - X2 * dZtdZ_1_0;
        JtJ(4, 0) += X1 * dZtdZ_2_1 - X2 * dZtdZ_1_1;
        JtJ(5, 0) += X1 * dZtdZ_2_2 - X2 * dZtdZ_2_1;
        JtJ(1, 1) += X2 * (X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0) + X0 * (X0 * dZtdZ_2_2 - X2 * dZtdZ_2_0);
        JtJ(2, 1) += -X2 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) - X0 * (X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0);
        JtJ(3, 1) += X2 * dZtdZ_0_0 - X0 * dZtdZ_2_0;
        JtJ(4, 1) += X2 * dZtdZ_1_0 - X0 * dZtdZ_2_1;
        JtJ(5, 1) += X2 * dZtdZ_2_0 - X0 * dZtdZ_2_2;
        JtJ(2, 2) += X1 * (X1 * dZtdZ_0_0 - X0 * dZtdZ_1_0) + X0 * (X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0);
        JtJ(3, 2) += X0 * dZtdZ_1_0 - X1 * dZtdZ_0_0;
        JtJ(4, 2) += X0 * dZtdZ_1_1 - X1 * dZtdZ_1_0;
        JtJ(5, 2) += X0 * dZtdZ_2_1 - X1 * dZtdZ_2_0;
        JtJ(3, 3) += dZtdZ_0_0;
        JtJ(4, 3) += dZtdZ_1_0;
        JtJ(5, 3) += dZtdZ_2_0;
        JtJ(4, 4) += dZtdZ_1_1;
        JtJ(5, 4) += dZtdZ_2_1;
        JtJ(5, 5) += dZtdZ_2_2;
        Jtr(0) += (r(0) * (X1 * dZ(0, 2) - X2 * dZ(0, 1)) + r(1) * (X1 * dZ(1, 2) - X2 * dZ(1, 1)));
        Jtr(1) += (-r(0) * (X0 * dZ(0, 2) - X2 * dZ(0, 0)) - r(1) * (X0 * dZ(1, 2) - X2 * dZ(1, 0)));
        Jtr(2) += (r(0) * (X0 * dZ(0, 1) - X1 * dZ(0, 0)) + r(1) * (X0 * dZ(1, 1) - X1 * dZ(1, 0)));
        Jtr(3) += (dZ(0, 0) * r(0) + dZ(1, 0) * r(1));
        Jtr(4) += (dZ(0, 1) * r(0) + dZ(1, 1) * r(1));
        Jtr(5) += (dZ(0, 2) * r(0) + dZ(1, 2) * r(1));
    }

    return num_residuals;
}

template <>
double CameraJacobianAccumulatorSIMD<NullCameraModel, TruncatedLoss, UniformWeightVector>::residual(
    const CameraPose &pose) const {
    double cost = 0;

    size_t K = x.rows();

    double ts[3];
    for (size_t i = 0; i < 3; ++i) {
        ts[i] = pose.t(i);
    }

    __m256d P[3][3];
    const Eigen::Matrix3d R = pose.R();
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            P[i][j] = _mm256_set1_pd(R(i, j));
        }
    }
    __m256d costs = _mm256_setzero_pd();
    __m256d sq_thresholds = _mm256_set1_pd(loss_fn.squared_thr);

    const double *X0 = X.data();
    const double *x0 = x.data();

    for (size_t k = 0; k < K - 3; k += 4) {
        __m256d z[3];
        for (size_t i = 0; i < 3; ++i) {
            z[i] = _mm256_set1_pd(ts[i]);
        }
        for (size_t i = 0; i < 3; ++i) {
            __m256d X_i = _mm256_loadu_pd(X0 + K * i + k);
            for (size_t j = 0; j < 3; ++j) {
                z[j] = _mm256_fmadd_pd(P[j][i], X_i, z[j]);
            }
        }

        int cond = _mm256_movemask_pd(z[2]);
        if (cond == 0b1111) {
            continue;
        }

        __m256d inv_z2 = _mm256_div_pd(_mm256_set1_pd(1.0), z[2]);
        for (size_t i = 0; i < 2; ++i) {
            z[i] = _mm256_mul_pd(z[i], inv_z2);
        }
        for (size_t i = 0; i < 2; ++i) {
            z[i] = _mm256_sub_pd(z[i], _mm256_loadu_pd(x0 + K * i + k));
        }
        for (size_t i = 0; i < 2; ++i) {
            z[i] = _mm256_mul_pd(z[i], z[i]);
        }
        __m256d r_squared = _mm256_min_pd(_mm256_add_pd(z[0], z[1]), sq_thresholds);

        __m256d valid_mask = _mm256_cmp_pd(z[2], _mm256_setzero_pd(), _CMP_GT_OQ);
        __m256d valid_r_squared = _mm256_and_pd(valid_mask, r_squared);
        costs = _mm256_add_pd(costs, valid_r_squared);
    }
    cost = sum_pd(costs);

    for (size_t k = K - K % 4; k < K; ++k) {
        Eigen::Vector3d Z = pose.apply(X.row(k).transpose());
        if (Z(2) < 0)
            continue;
        const double inv_z = 1.0 / Z(2);
        const double r_squared = (Z.block<2, 1>(0, 0) * inv_z - x.row(k).transpose()).squaredNorm();
        cost += loss_fn.loss(r_squared);
    }

    return cost;
}

} // namespace poselib
