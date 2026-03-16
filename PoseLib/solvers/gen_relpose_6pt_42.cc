// Copyright (c) 2021, Viktor Larsson
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of the copyright holder nor the
//       names of its contributors may be used to endorse or promote products
//       derived from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include "gen_relpose_6pt_42.h"

#include "PoseLib/misc/essential.h"
#include "PoseLib/misc/quaternion.h"

#include <Eigen/Dense>

#include <array>
#include <complex>
#include <vector>

#define USE_FAST_EIGENVECTOR_SOLVER


namespace poselib {
namespace {

constexpr int kNumObs = 6;
constexpr int kNumRefObs = 2;
constexpr int kNumOffObs = 4;
constexpr int kNumDeg2 = 10;
constexpr int kNumDeg4 = 35;
constexpr int kNumDeg6 = 84;
constexpr int kNumCoeff = 1316;
constexpr int kNumSexticRows = 14;
constexpr int kNumQuarticRows = 4;
constexpr int kNumBaseMonomials42 = 10;
constexpr double kImagTol = 1e-8;

using Poly2 = std::array<double, kNumDeg2>;
using Poly4 = std::array<double, kNumDeg4>;
using Poly6 = std::array<double, kNumDeg6>;
using RowPoly = std::array<Poly2, 3>;
using RowSet = std::array<RowPoly, 5>;
using Matrix76 = Eigen::Matrix<double, 76, 76>;
using Matrix76x40 = Eigen::Matrix<double, 76, 40>;
using Matrix50x40 = Eigen::Matrix<double, 50, 40>;
using Matrix40 = Eigen::Matrix<double, 40, 40>;
using Matrix10 = Eigen::Matrix<double, kNumBaseMonomials42, kNumBaseMonomials42>;
using Matrix10x9 = Eigen::Matrix<double, kNumBaseMonomials42, kNumBaseMonomials42 - 1>;

struct TripletEntry {
    int row;
    int col;
    int coeff;
};

constexpr TripletEntry kC0Triplets[] = {
#include "gen_relpose_6pt_42_c0.inc"
};

constexpr TripletEntry kC1Triplets[] = {
#include "gen_relpose_6pt_42_c1.inc"
};

constexpr std::array<int, 40> kActionPermutation = {45, 29, 19, 14, 15, 9, 17, 18, 8, 20, 21, 5, 26, 24, 25, 7, 27, 28, 4, 30, 31, 32, 2, 41, 38, 36, 37, 6, 39, 40, 3, 42, 43, 44, 1, 46, 47, 48, 49, 0};
// Metadata exported from the finalized PysolverGen template for semigen_relpose_6pt_42.
constexpr std::array<int, kNumBaseMonomials42> kFastBaseBasisRows = {0, 1, 2, 3, 6, 12, 13, 23, 24, 25};
constexpr std::array<int, kNumBaseMonomials42> kFastTopBasisRows = {39, 22, 11, 5, 8, 18, 15, 34, 30, 27};
constexpr std::array<int, 40> kFastBasisBaseIndex = {
    0, 1, 2, 3, 3, 3, 4, 4, 4, 2,
    2, 2, 5, 6, 6, 6, 5, 5, 5, 1,
    1, 1, 1, 7, 8, 9, 9, 9, 8, 8,
    8, 7, 7, 7, 7, 0, 0, 0, 0, 0,
};
constexpr std::array<int, 40> kFastBasisActionPowers = {
    0, 0, 0, 0, 1, 2, 0, 1, 2, 1,
    2, 3, 0, 0, 1, 2, 1, 2, 3, 1,
    2, 3, 4, 0, 0, 0, 1, 2, 1, 2,
    3, 1, 2, 3, 4, 1, 2, 3, 4, 5,
};
constexpr int kFastExtractX1Base = 1;
constexpr int kFastExtractX2Base = 7;

std::vector<std::array<int, 3>> generate_exponents(int max_degree) {
    std::vector<std::array<int, 3>> exponents;
    for (int e1 = 0; e1 <= max_degree; ++e1) {
        for (int e2 = 0; e2 <= max_degree - e1; ++e2) {
            for (int e3 = 0; e3 <= max_degree - e1 - e2; ++e3) {
                exponents.push_back({e1, e2, e3});
            }
        }
    }
    return exponents;
}

int find_exponent_index(const std::vector<std::array<int, 3>> &exponents, int e1, int e2, int e3) {
    for (size_t i = 0; i < exponents.size(); ++i) {
        const auto &exp = exponents[i];
        if (exp[0] == e1 && exp[1] == e2 && exp[2] == e3) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

const std::array<std::array<int, kNumDeg2>, kNumDeg2> &mul_2_2_to_4() {
    static const auto table = []() {
        std::array<std::array<int, kNumDeg2>, kNumDeg2> result{};
        const auto deg2 = generate_exponents(2);
        const auto deg4 = generate_exponents(4);
        for (int i = 0; i < kNumDeg2; ++i) {
            for (int j = 0; j < kNumDeg2; ++j) {
                const auto &a = deg2[i];
                const auto &b = deg2[j];
                result[i][j] = find_exponent_index(deg4, a[0] + b[0], a[1] + b[1], a[2] + b[2]);
            }
        }
        return result;
    }();
    return table;
}

const std::array<std::array<int, kNumDeg2>, kNumDeg4> &mul_4_2_to_6() {
    static const auto table = []() {
        std::array<std::array<int, kNumDeg2>, kNumDeg4> result{};
        const auto deg2 = generate_exponents(2);
        const auto deg4 = generate_exponents(4);
        const auto deg6 = generate_exponents(6);
        for (int i = 0; i < kNumDeg4; ++i) {
            for (int j = 0; j < kNumDeg2; ++j) {
                const auto &a = deg4[i];
                const auto &b = deg2[j];
                result[i][j] = find_exponent_index(deg6, a[0] + b[0], a[1] + b[1], a[2] + b[2]);
            }
        }
        return result;
    }();
    return table;
}

const std::array<std::array<Poly2, 3>, 3> &rotation_coeffs2() {
    static const auto coeffs = []() {
        std::array<std::array<Poly2, 3>, 3> r{};

        r[0][0][0] = 1.0;
        r[0][0][9] = 1.0;
        r[0][0][5] = -1.0;
        r[0][0][2] = -1.0;

        r[0][1][8] = 2.0;
        r[0][1][1] = -2.0;

        r[0][2][3] = 2.0;
        r[0][2][7] = 2.0;

        r[1][0][1] = 2.0;
        r[1][0][8] = 2.0;

        r[1][1][0] = 1.0;
        r[1][1][9] = -1.0;
        r[1][1][5] = 1.0;
        r[1][1][2] = -1.0;

        r[1][2][6] = -2.0;
        r[1][2][4] = 2.0;

        r[2][0][3] = -2.0;
        r[2][0][7] = 2.0;

        r[2][1][6] = 2.0;
        r[2][1][4] = 2.0;

        r[2][2][0] = 1.0;
        r[2][2][9] = -1.0;
        r[2][2][5] = -1.0;
        r[2][2][2] = 1.0;

        return r;
    }();
    return coeffs;
}

Poly4 mul_deg2_deg2(const Poly2 &p, const Poly2 &q) {
    Poly4 out{};
    const auto &table = mul_2_2_to_4();
    for (int i = 0; i < kNumDeg2; ++i) {
        if (p[i] == 0.0) {
            continue;
        }
        for (int j = 0; j < kNumDeg2; ++j) {
            if (q[j] == 0.0) {
                continue;
            }
            out[table[i][j]] += p[i] * q[j];
        }
    }
    return out;
}

Poly6 mul_deg4_deg2(const Poly4 &p, const Poly2 &q) {
    Poly6 out{};
    const auto &table = mul_4_2_to_6();
    for (int i = 0; i < kNumDeg4; ++i) {
        if (p[i] == 0.0) {
            continue;
        }
        for (int j = 0; j < kNumDeg2; ++j) {
            if (q[j] == 0.0) {
                continue;
            }
            out[table[i][j]] += p[i] * q[j];
        }
    }
    return out;
}

Poly6 triple_product_deg2(const Poly2 &p, const Poly2 &q, const Poly2 &r) {
    return mul_deg4_deg2(mul_deg2_deg2(p, q), r);
}

Poly4 det2_deg2(const Poly2 &a, const Poly2 &b, const Poly2 &c, const Poly2 &d) {
    Poly4 out = mul_deg2_deg2(a, d);
    const Poly4 tmp = mul_deg2_deg2(b, c);
    for (int i = 0; i < kNumDeg4; ++i) {
        out[i] -= tmp[i];
    }
    return out;
}

Poly6 det3_deg2(const RowPoly &r0, const RowPoly &r1, const RowPoly &r2) {
    const Poly6 aei = triple_product_deg2(r0[0], r1[1], r2[2]);
    const Poly6 afh = triple_product_deg2(r0[0], r1[2], r2[1]);
    const Poly6 bdi = triple_product_deg2(r0[1], r1[0], r2[2]);
    const Poly6 bfg = triple_product_deg2(r0[1], r1[2], r2[0]);
    const Poly6 cdh = triple_product_deg2(r0[2], r1[0], r2[1]);
    const Poly6 ceg = triple_product_deg2(r0[2], r1[1], r2[0]);

    Poly6 out{};
    for (int i = 0; i < kNumDeg6; ++i) {
        out[i] = aei[i] - afh[i] - bdi[i] + bfg[i] + cdh[i] - ceg[i];
    }
    return out;
}

Poly2 bilinear_rotation_poly(const Eigen::Vector3d &u, const Eigen::Vector3d &v) {
    Poly2 out{};
    const auto &r = rotation_coeffs2();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const double weight = u(i) * v(j);
            if (weight == 0.0) {
                continue;
            }
            for (int k = 0; k < kNumDeg2; ++k) {
                out[k] += weight * r[i][j][k];
            }
        }
    }
    return out;
}

RowSet compute_rows_deg2(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                         const std::vector<Eigen::Vector3d> &p2, int kk) {
    RowSet rows{};
    int row_idx = 0;
    const Eigen::Vector3d &x1_kk = x1[kk];
    const Eigen::Vector3d &x2_kk = x2[kk];
    const Eigen::Vector3d &p2_kk = p2[kk];

    for (int k = 0; k < kNumObs; ++k) {
        if (k == kk) {
            continue;
        }

        const Eigen::Vector3d &x2_k = x2[k];
        const Eigen::Vector3d &x1_k = x1[k];
        const Eigen::Vector3d lam1_right = x1_kk.cross(x1_k);
        const Eigen::Vector3d lam2_left = x2_k.cross(x2_kk);
        const Eigen::Vector3d const_left = x2_k.cross(p2[k] - p2_kk);

        rows[row_idx][0] = bilinear_rotation_poly(x2_k, -lam1_right);
        rows[row_idx][1] = bilinear_rotation_poly(lam2_left, x1_k);
        rows[row_idx][2] = bilinear_rotation_poly(const_left, x1_k);
        ++row_idx;
    }

    return rows;
}

void store_sextic(std::array<Poly6, kNumSexticRows> &c6, int out_idx, const RowSet &rows, int i0, int i1, int i2) {
    c6[out_idx] = det3_deg2(rows[i0], rows[i1], rows[i2]);
}

void store_quartic(std::array<Poly4, kNumQuarticRows> &c4, int out_idx, const RowSet &rows, int i0, int i1) {
    c4[out_idx] = det2_deg2(rows[i0][0], rows[i0][1], rows[i1][0], rows[i1][1]);
}

std::array<double, kNumCoeff> compute_minimal_coefficients(const std::vector<Eigen::Vector3d> &x1_ref,
                                                           const std::vector<Eigen::Vector3d> &x2_ref,
                                                           const std::vector<Eigen::Vector3d> &x1_off,
                                                           const std::vector<Eigen::Vector3d> &x2_off,
                                                           const Eigen::Vector3d &p2_off) {
    std::vector<Eigen::Vector3d> x1;
    std::vector<Eigen::Vector3d> x2;
    std::vector<Eigen::Vector3d> p2;
    x1.reserve(kNumObs);
    x2.reserve(kNumObs);
    p2.reserve(kNumObs);

    for (int i = 0; i < kNumRefObs; ++i) {
        x1.push_back(x1_ref[i]);
        x2.push_back(x2_ref[i]);
        p2.emplace_back(Eigen::Vector3d::Zero());
    }
    for (int i = 0; i < kNumOffObs; ++i) {
        x1.push_back(x1_off[i]);
        x2.push_back(x2_off[i]);
        p2.push_back(p2_off);
    }

    std::array<Poly6, kNumSexticRows> c6{};
    std::array<Poly4, kNumQuarticRows> c4{};

    for (int kk = 0; kk < kNumObs; ++kk) {
        const RowSet rows = compute_rows_deg2(x1, x2, p2, kk);
        switch (kk) {
        case 0:
            store_sextic(c6, 0, rows, 0, 1, 2);
            store_sextic(c6, 1, rows, 0, 2, 4);
            store_sextic(c6, 2, rows, 1, 2, 3);
            break;
        case 1:
            store_sextic(c6, 3, rows, 0, 2, 3);
            store_sextic(c6, 4, rows, 0, 3, 4);
            store_sextic(c6, 5, rows, 1, 2, 3);
            store_sextic(c6, 6, rows, 1, 3, 4);
            break;
        case 2:
            store_sextic(c6, 7, rows, 0, 1, 3);
            store_sextic(c6, 8, rows, 1, 2, 4);
            store_quartic(c4, 0, rows, 2, 3);
            break;
        case 3:
            store_sextic(c6, 9, rows, 0, 2, 4);
            store_quartic(c4, 1, rows, 2, 4);
            break;
        case 4:
            store_sextic(c6, 10, rows, 0, 3, 4);
            store_quartic(c4, 2, rows, 2, 4);
            store_quartic(c4, 3, rows, 3, 4);
            break;
        case 5:
            store_sextic(c6, 11, rows, 0, 1, 2);
            store_sextic(c6, 12, rows, 0, 2, 4);
            store_sextic(c6, 13, rows, 1, 3, 4);
            break;
        default:
            break;
        }
    }

    std::array<double, kNumCoeff> coeffs{};
    int offset = 0;
    for (const auto &poly : c6) {
        for (double value : poly) {
            coeffs[offset++] = value;
        }
    }
    for (const auto &poly : c4) {
        for (double value : poly) {
            coeffs[offset++] = value;
        }
    }
    return coeffs;
}

void build_elimination_template(const std::array<double, kNumCoeff> &coeffs, Matrix76 *c0, Matrix76x40 *c1) {
    c0->setZero();
    c1->setZero();
    for (const TripletEntry &entry : kC0Triplets) {
        (*c0)(entry.row, entry.col) = coeffs[entry.coeff];
    }
    for (const TripletEntry &entry : kC1Triplets) {
        (*c1)(entry.row, entry.col) = coeffs[entry.coeff];
    }
}

#ifdef USE_FAST_EIGENVECTOR_SOLVER
void fast_eigenvector_solver_42(const double *eigv, int neig, const Matrix40 &am, Eigen::Matrix<double, 3, 40> &sols) {
    for (int i = 0; i < neig; ++i) {
        const double z = eigv[i];
        double zi[7];
        zi[0] = 1.0;
        for (int k = 1; k < 7; ++k) {
            zi[k] = zi[k - 1] * z;
        }

        Matrix10 A;
        A.setZero();
        for (int eq = 0; eq < kNumBaseMonomials42; ++eq) {
            const int row = kFastTopBasisRows[eq];
            for (int col = 0; col < 40; ++col) {
                A(eq, kFastBasisBaseIndex[col]) += am(row, col) * zi[kFastBasisActionPowers[col]];
            }
            A(eq, eq) -= zi[kFastBasisActionPowers[row] + 1];
        }

        Eigen::Matrix<double, kNumBaseMonomials42, 1> u;
        u(0) = 1.0;
        u.tail<kNumBaseMonomials42 - 1>() =
            A.rightCols<kNumBaseMonomials42 - 1>().colPivHouseholderQr().solve(-A.col(0));

        sols(0, i) = u(kFastExtractX1Base);
        sols(1, i) = u(kFastExtractX2Base);
        sols(2, i) = z;
    }
}
#endif

void root_refinement(const std::vector<Eigen::Vector3d> &p1, const std::vector<Eigen::Vector3d> &x1,
                     const std::vector<Eigen::Vector3d> &p2, const std::vector<Eigen::Vector3d> &x2,
                     std::vector<CameraPose> *output) {

    Eigen::Matrix<double, 6, 6> J;
    Eigen::Matrix<double, 6, 1> res;
    Eigen::Matrix<double, 6, 1> dp;

    std::vector<Eigen::Vector3d> qq1(6, Eigen::Vector3d::Zero()), qq2(6, Eigen::Vector3d::Zero());
    for (size_t pt_k = 0; pt_k < 6; ++pt_k) {
        qq1[pt_k] = x1[pt_k].cross(p1[pt_k]);
        qq2[pt_k] = x2[pt_k].cross(p2[pt_k]);
    }

    for (size_t pose_k = 0; pose_k < output->size(); ++pose_k) {
        CameraPose &pose = (*output)[pose_k];

        for (size_t iter = 0; iter < 5; ++iter) {
            for (size_t pt_k = 0; pt_k < 6; ++pt_k) {
                Eigen::Vector3d x2t = x2[pt_k].cross(pose.t);
                Eigen::Vector3d Rx1 = pose.rotate(x1[pt_k]);
                Eigen::Vector3d Rqq1 = pose.rotate(qq1[pt_k]);

                res(pt_k) = (x2t - qq2[pt_k]).dot(Rx1) - x2[pt_k].dot(Rqq1);
                J.block<1, 3>(pt_k, 0) = -x2t.cross(Rx1) + qq2[pt_k].cross(Rx1) + x2[pt_k].cross(Rqq1);
                J.block<1, 3>(pt_k, 3) = -x2[pt_k].cross(Rx1);
            }

            if (res.norm() < 1e-12) {
                break;
            }

            dp = J.partialPivLu().solve(res);

            const Eigen::Vector3d w = -dp.block<3, 1>(0, 0);
            pose.q = quat_step_pre(pose.q, w);
            pose.t = pose.t - dp.block<3, 1>(3, 0);
        }
    }
}

} // namespace

int gen_relpose_6pt_42(const std::vector<Eigen::Vector3d> &x1_ref, const std::vector<Eigen::Vector3d> &x2_ref,
                       const std::vector<Eigen::Vector3d> &x1_off, const std::vector<Eigen::Vector3d> &x2_off,
                       const Eigen::Vector3d &p2_off, std::vector<CameraPose> *output) {
    output->clear();
    if (x1_ref.size() != kNumRefObs || x2_ref.size() != kNumRefObs || x1_off.size() != kNumOffObs ||
        x2_off.size() != kNumOffObs) {
        return 0;
    }

    const std::array<double, kNumCoeff> coeffs =
        compute_minimal_coefficients(x1_ref, x2_ref, x1_off, x2_off, p2_off);

    Matrix76 c0;
    Matrix76x40 c1;
    build_elimination_template(coeffs, &c0, &c1);

    const Matrix76x40 c12 = c0.partialPivLu().solve(c1);

    Matrix50x40 rr;
    rr.topRows<10>() = -c12.bottomRows<10>();
    rr.bottomRows<40>().setIdentity();

    Matrix40 action_matrix;
    for (int i = 0; i < 40; ++i) {
        action_matrix.row(i) = rr.row(kActionPermutation[i]);
    }

    Eigen::Matrix<double, 3, 40> sols;
    sols.setZero();
    int n_roots = 0;

#ifdef USE_FAST_EIGENVECTOR_SOLVER
    Eigen::EigenSolver<Matrix40> eigensolver(action_matrix, false);
    const Eigen::Matrix<std::complex<double>, 40, 1> eigenvalues = eigensolver.eigenvalues();
    double eigv[40];
    for (int k = 0; k < 40; ++k) {
        if (std::abs(eigenvalues(k).imag()) < kImagTol) {
            eigv[n_roots++] = eigenvalues(k).real();
        }
    }
    fast_eigenvector_solver_42(eigv, n_roots, action_matrix, sols);
#else
    Eigen::EigenSolver<Matrix40> eigensolver(action_matrix);
    const Eigen::ArrayXcd eigenvalues = eigensolver.eigenvalues();
    const Eigen::ArrayXXcd eigenvectors = eigensolver.eigenvectors();
    for (int k = 0; k < 40; ++k) {
        if (std::abs(eigenvalues(k).imag()) >= kImagTol) {
            continue;
        }
        const std::complex<double> scale = eigenvectors(0, k);
        if (std::abs(scale) < 1e-12) {
            continue;
        }

        sols(0, n_roots) = (eigenvectors(1, k) / scale).real();
        sols(1, n_roots) = (eigenvectors(23, k) / scale).real();
        sols(2, n_roots) = eigenvalues(k).real();
        ++n_roots;
    }
#endif

    std::vector<Eigen::Vector3d> p1(kNumObs, Eigen::Vector3d::Zero());
    std::vector<Eigen::Vector3d> x1;
    std::vector<Eigen::Vector3d> p2;
    std::vector<Eigen::Vector3d> x2;
    x1.reserve(kNumObs);
    p2.reserve(kNumObs);
    x2.reserve(kNumObs);
    for (int i = 0; i < kNumRefObs; ++i) {
        x1.push_back(x1_ref[i]);
        x2.push_back(x2_ref[i]);
        p2.emplace_back(Eigen::Vector3d::Zero());
    }
    for (int i = 0; i < kNumOffObs; ++i) {
        x1.push_back(x1_off[i]);
        x2.push_back(x2_off[i]);
        p2.push_back(p2_off);
    }

    output->reserve(n_roots);
    for (int sol_k = 0; sol_k < n_roots; ++sol_k) {
        CameraPose pose;
        pose.q << 1.0, sols(0, sol_k), sols(1, sol_k), sols(2, sol_k);
        pose.q.normalize();

        const Eigen::Matrix3d R = quat_to_rotmat(pose.q);

        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        for (int i = 0; i < kNumObs; ++i) {
            const Eigen::Vector3d u = (R * x1[i]).cross(x2[i]);
            const Eigen::Vector3d v = p2[i] - R * p1[i];
            A += u * u.transpose();
            b += u * (u.dot(v));
        }
        pose.t = A.llt().solve(b);

        bool cheiral_ok = true;
        for (int pt_k = 0; pt_k < kNumObs; ++pt_k) {
            if (!check_cheirality(pose, p1[pt_k], x1[pt_k], p2[pt_k], x2[pt_k])) {
                cheiral_ok = false;
                break;
            }
        }
        if (!cheiral_ok) {
            continue;
        }

        output->push_back(pose);
    }

    root_refinement(p1, x1, p2, x2, output);
    return static_cast<int>(output->size());
}

} // namespace poselib
