//
// Created by kocur on 15-May-24.
//

#include "PoseLib/solvers/relpose_k2Fk1_10pt.h"

namespace poselib {
template <typename Derived> inline void colEchelonForm(Eigen::MatrixBase<Derived> &M, double pivtol = 1e-12) {
    typedef typename Derived::Scalar Scalar;

    int n = M.rows();
    int m = M.cols();
    int i = 0, j = 0, k = 0;
    int col = 0;
    Scalar p, tp;

    while ((i < m) && (j < n)) {
        p = std::numeric_limits<Scalar>::min();
        col = i;

        for (k = i; k < m; k++) {
            tp = std::abs(M(j, k));
            if (tp > p) {
                p = tp;
                col = k;
            }
        }

        if (p < Scalar(pivtol)) {
            M.block(j, i, 1, m - i).setZero();
            j++;
        } else {
            if (col != i)
                M.block(j, i, n - j, 1).swap(M.block(j, col, n - j, 1));

            M.block(j + 1, i, n - j - 1, 1) /= M(j, i);
            M(j, i) = 1.0;

            for (k = 0; k < m; k++) {
                if (k == i)
                    continue;

                M.block(j, k, n - j, 1) -= M(j, k) * M.block(j, i, n - j, 1);
            }

            i++;
            j++;
        }
    }
}

template <typename Scalar, typename Derived>
int relpose_k2Fk1_10pt_gb(Eigen::Matrix<Scalar, 29, 1> &pr, Eigen::MatrixBase<Derived> &sols, double pivtol = 1e-16) {
    Eigen::Matrix<Scalar, 36, 1> c;

    c(0) = pr(0) * pr(14) - pr(4) * pr(10);
    c(1) = pr(0) * pr(16) + pr(2) * pr(14) - pr(4) * pr(12) - pr(6) * pr(10);
    c(2) = pr(2) * pr(18) - pr(8) * pr(12);
    c(3) = pr(1) * pr(15) - pr(5) * pr(11);
    c(4) = pr(1) * pr(17) + pr(3) * pr(15) - pr(5) * pr(13) - pr(7) * pr(11);
    c(5) = pr(0) * pr(15) + pr(1) * pr(14) - pr(4) * pr(11) - pr(5) * pr(10);
    c(6) = pr(0) * pr(17) + pr(1) * pr(16) + pr(2) * pr(15) + pr(3) * pr(14) - pr(4) * pr(13) - pr(5) * pr(12) -
           pr(6) * pr(11) - pr(7) * pr(10);
    c(7) = pr(0) * pr(18) + pr(2) * pr(16) - pr(6) * pr(12) - pr(8) * pr(10);
    c(8) = pr(0) * pr(19) + pr(1) * pr(18) + pr(2) * pr(17) + pr(3) * pr(16) - pr(6) * pr(13) - pr(7) * pr(12) -
           pr(8) * pr(11) - pr(9) * pr(10);
    c(9) = pr(2) * pr(19) + pr(3) * pr(18) - pr(8) * pr(13) - pr(9) * pr(12);
    c(10) = pr(1) * pr(19) + pr(3) * pr(17) - pr(7) * pr(13) - pr(9) * pr(11);
    c(11) = pr(3) * pr(19) - pr(9) * pr(13);
    c(12) = pr(0) * pr(23) - pr(4) * pr(20);
    c(13) = pr(1) * pr(23) + pr(0) * pr(25) - pr(4) * pr(21) - pr(5) * pr(20);
    c(14) = pr(2) * pr(24) - pr(8) * pr(20);
    c(15) = pr(3) * pr(24) + pr(2) * pr(27) - pr(8) * pr(21) - pr(9) * pr(20);
    c(16) = pr(1) * pr(26) - pr(5) * pr(22);
    c(17) = pr(0) * pr(24) + pr(2) * pr(23) - pr(6) * pr(20);
    c(18) = pr(0) * pr(26) + pr(1) * pr(25) - pr(4) * pr(22) - pr(5) * pr(21);
    c(19) = pr(1) * pr(24) + pr(3) * pr(23) + pr(0) * pr(27) + pr(2) * pr(25) - pr(6) * pr(21) - pr(7) * pr(20);
    c(20) = pr(0) * pr(28) + pr(1) * pr(27) + pr(2) * pr(26) + pr(3) * pr(25) - pr(6) * pr(22) - pr(7) * pr(21);
    c(21) = pr(2) * pr(28) + pr(3) * pr(27) - pr(8) * pr(22) - pr(9) * pr(21);
    c(22) = pr(1) * pr(28) + pr(3) * pr(26) - pr(7) * pr(22);
    c(23) = pr(3) * pr(28) - pr(9) * pr(22);
    c(24) = pr(10) * pr(23) - pr(14) * pr(20);
    c(25) = pr(11) * pr(23) + pr(10) * pr(25) - pr(14) * pr(21) - pr(15) * pr(20);
    c(26) = pr(12) * pr(24) - pr(18) * pr(20);
    c(27) = pr(13) * pr(24) + pr(12) * pr(27) - pr(18) * pr(21) - pr(19) * pr(20);
    c(28) = pr(11) * pr(26) - pr(15) * pr(22);
    c(29) = pr(10) * pr(24) + pr(12) * pr(23) - pr(16) * pr(20);
    c(30) = pr(10) * pr(26) + pr(11) * pr(25) - pr(14) * pr(22) - pr(15) * pr(21);
    c(31) = pr(11) * pr(24) + pr(13) * pr(23) + pr(10) * pr(27) + pr(12) * pr(25) - pr(16) * pr(21) - pr(17) * pr(20);
    c(32) = pr(10) * pr(28) + pr(11) * pr(27) + pr(12) * pr(26) + pr(13) * pr(25) - pr(16) * pr(22) - pr(17) * pr(21);
    c(33) = pr(12) * pr(28) + pr(13) * pr(27) - pr(18) * pr(22) - pr(19) * pr(21);
    c(34) = pr(11) * pr(28) + pr(13) * pr(26) - pr(17) * pr(22);
    c(35) = pr(13) * pr(28) - pr(19) * pr(22);

    Eigen::Matrix<Scalar, 20, 10> M;
    M.setZero();

    M(0) = c(0);
    M(61) = c(0);
    M(82) = c(0);
    M(144) = c(0);
    M(2) = c(1);
    M(64) = c(1);
    M(85) = c(1);
    M(148) = c(1);
    M(9) = c(2);
    M(72) = c(2);
    M(93) = c(2);
    M(156) = c(2);
    M(3) = c(3);
    M(66) = c(3);
    M(87) = c(3);
    M(150) = c(3);
    M(7) = c(4);
    M(70) = c(4);
    M(91) = c(4);
    M(154) = c(4);
    M(1) = c(5);
    M(63) = c(5);
    M(84) = c(5);
    M(147) = c(5);
    M(4) = c(6);
    M(67) = c(6);
    M(88) = c(6);
    M(151) = c(6);
    M(5) = c(7);
    M(68) = c(7);
    M(89) = c(7);
    M(152) = c(7);
    M(8) = c(8);
    M(71) = c(8);
    M(92) = c(8);
    M(155) = c(8);
    M(12) = c(9);
    M(75) = c(9);
    M(96) = c(9);
    M(158) = c(9);
    M(11) = c(10);
    M(74) = c(10);
    M(95) = c(10);
    M(157) = c(10);
    M(15) = c(11);
    M(77) = c(11);
    M(98) = c(11);
    M(159) = c(11);
    M(20) = c(12);
    M(102) = c(12);
    M(165) = c(12);
    M(21) = c(13);
    M(104) = c(13);
    M(168) = c(13);
    M(25) = c(14);
    M(109) = c(14);
    M(173) = c(14);
    M(28) = c(15);
    M(112) = c(15);
    M(176) = c(15);
    M(26) = c(16);
    M(110) = c(16);
    M(174) = c(16);
    M(22) = c(17);
    M(105) = c(17);
    M(169) = c(17);
    M(23) = c(18);
    M(107) = c(18);
    M(171) = c(18);
    M(24) = c(19);
    M(108) = c(19);
    M(172) = c(19);
    M(27) = c(20);
    M(111) = c(20);
    M(175) = c(20);
    M(31) = c(21);
    M(115) = c(21);
    M(178) = c(21);
    M(30) = c(22);
    M(114) = c(22);
    M(177) = c(22);
    M(34) = c(23);
    M(117) = c(23);
    M(179) = c(23);
    M(40) = c(24);
    M(122) = c(24);
    M(185) = c(24);
    M(41) = c(25);
    M(124) = c(25);
    M(188) = c(25);
    M(45) = c(26);
    M(129) = c(26);
    M(193) = c(26);
    M(48) = c(27);
    M(132) = c(27);
    M(196) = c(27);
    M(46) = c(28);
    M(130) = c(28);
    M(194) = c(28);
    M(42) = c(29);
    M(125) = c(29);
    M(189) = c(29);
    M(43) = c(30);
    M(127) = c(30);
    M(191) = c(30);
    M(44) = c(31);
    M(128) = c(31);
    M(192) = c(31);
    M(47) = c(32);
    M(131) = c(32);
    M(195) = c(32);
    M(51) = c(33);
    M(135) = c(33);
    M(198) = c(33);
    M(50) = c(34);
    M(134) = c(34);
    M(197) = c(34);
    M(54) = c(35);
    M(137) = c(35);
    M(199) = c(35);

    colEchelonForm(M, pivtol);

    Eigen::Matrix<Scalar, 10, 10> A;
    A.setZero();

    A(0, 2) = 1.000000;
    A(1, 4) = 1.000000;
    A(2, 5) = 1.000000;
    A(3, 7) = 1.000000;
    A(4, 8) = 1.000000;
    A(5, 9) = 1.000000;
    A(6, 0) = -M(19, 9);
    A(6, 1) = -M(18, 9);
    A(6, 2) = -M(17, 9);
    A(6, 3) = -M(16, 9);
    A(6, 4) = -M(15, 9);
    A(6, 5) = -M(14, 9);
    A(6, 6) = -M(13, 9);
    A(6, 7) = -M(12, 9);
    A(6, 8) = -M(11, 9);
    A(6, 9) = -M(10, 9);
    A(7, 0) = -M(19, 8);
    A(7, 1) = -M(18, 8);
    A(7, 2) = -M(17, 8);
    A(7, 3) = -M(16, 8);
    A(7, 4) = -M(15, 8);
    A(7, 5) = -M(14, 8);
    A(7, 6) = -M(13, 8);
    A(7, 7) = -M(12, 8);
    A(7, 8) = -M(11, 8);
    A(7, 9) = -M(10, 8);
    A(8, 0) = -M(19, 7);
    A(8, 1) = -M(18, 7);
    A(8, 2) = -M(17, 7);
    A(8, 3) = -M(16, 7);
    A(8, 4) = -M(15, 7);
    A(8, 5) = -M(14, 7);
    A(8, 6) = -M(13, 7);
    A(8, 7) = -M(12, 7);
    A(8, 8) = -M(11, 7);
    A(8, 9) = -M(10, 7);
    A(9, 0) = -M(19, 6);
    A(9, 1) = -M(18, 6);
    A(9, 2) = -M(17, 6);
    A(9, 3) = -M(16, 6);
    A(9, 4) = -M(15, 6);
    A(9, 5) = -M(14, 6);
    A(9, 6) = -M(13, 6);
    A(9, 7) = -M(12, 6);
    A(9, 8) = -M(11, 6);
    A(9, 9) = -M(10, 6);

    Eigen::EigenSolver<Eigen::Matrix<Scalar, 10, 10>> eig(A);
    Eigen::Matrix<std::complex<Scalar>, 10, 2> esols;
    esols.col(0).array() = eig.eigenvectors().row(2).array() / eig.eigenvectors().row(0).array();
    esols.col(1).array() = eig.eigenvectors().row(1).array() / eig.eigenvectors().row(0).array();

    int nsols = 0;
    for (int i = 0; i < 10; i++) {
        if (esols.row(i).imag().isZero(100.0 * std::numeric_limits<Scalar>::epsilon()))
            sols.col(nsols++) = esols.row(i).real();
    }

    return nsols;
}

template <typename Derived1, typename Derived2, typename Derived3>
inline int relpose_k2Fk1_10pt_solver(const Eigen::MatrixBase<Derived1> &X, const Eigen::MatrixBase<Derived1> &U,
                                     Eigen::MatrixBase<Derived2> &Fs, Eigen::MatrixBase<Derived3> &Ls,
                                     double pivtol = 1e-16) {
    typedef typename Derived1::Scalar Scalar;

    eigen_assert((X.rows() == 10 && X.cols() == 2) && "The first parameter (x) must be a 10x2 matrix");
    eigen_assert((U.rows() == 10 && U.cols() == 2) && "The second parameter (u) must be a 10x2 matrix");
    eigen_assert((Fs.rows() == 9 && Fs.cols() == 10) && "The third parameter (Fs) must be a 9x10 matrix");
    eigen_assert((Ls.rows() == 2 && Ls.cols() == 10) && "The forth parameter (Ls) must be a 2x10 matrix");

    Eigen::Matrix<Scalar, 10, 1> Z1;
    Eigen::Matrix<Scalar, 10, 1> Z2;
    Eigen::Matrix<Scalar, 10, 16> A;

    Z1.array() = X.col(0).array() * X.col(0).array() + X.col(1).array() * X.col(1).array();
    Z2.array() = U.col(0).array() * U.col(0).array() + U.col(1).array() * U.col(1).array();

    A.col(0).array() = X.col(0).array() * U.col(0).array();
    A.col(1).array() = X.col(0).array() * U.col(1).array();
    A.col(2).array() = X.col(1).array() * U.col(0).array();
    A.col(3).array() = X.col(1).array() * U.col(1).array();
    A.col(4).array() = U.col(0).array() * Z1.array();
    A.col(5).array() = U.col(0).array();
    A.col(6).array() = U.col(1).array() * Z1.array();
    A.col(7).array() = U.col(1).array();
    A.col(8).array() = X.col(0).array() * Z2.array();
    A.col(9).array() = X.col(0).array();
    A.col(10).array() = X.col(1).array() * Z2.array();
    A.col(11).array() = X.col(1).array();
    A.col(12).array() = Z1.array() * Z2.array();
    A.col(13).array() = Z1.array();
    A.col(14).array() = Z2.array();
    A.col(15).fill(1.0);

    Eigen::Matrix<Scalar, 10, 6> Mr = A.template block<10, 10>(0, 0).lu().solve(A.template block<10, 6>(0, 10));

    Eigen::Matrix<Scalar, 29, 1> params;

    params << Mr(5, 0), Mr(5, 1), -Mr(4, 0), -Mr(4, 1), Mr(5, 2), Mr(5, 3), Mr(5, 4) - Mr(4, 2), Mr(5, 5) - Mr(4, 3),
        -Mr(4, 4), -Mr(4, 5), Mr(7, 0), Mr(7, 1), -Mr(6, 0), -Mr(6, 1), Mr(7, 2), Mr(7, 3), Mr(7, 4) - Mr(6, 2),
        Mr(7, 5) - Mr(6, 3), -Mr(6, 4), -Mr(6, 5), Mr(9, 0), Mr(9, 1) - Mr(8, 0), -Mr(8, 1), Mr(9, 2), Mr(9, 4),
        Mr(9, 3) - Mr(8, 2), -Mr(8, 3), Mr(9, 5) - Mr(8, 4), -Mr(8, 5);

    int nsols = relpose_k2Fk1_10pt_gb(params, Ls);

    if (nsols > 0) {
        Eigen::Matrix<Scalar, 4, 1> m1;
        Eigen::Matrix<Scalar, 6, 1> m2;
        Eigen::Matrix<Scalar, 6, 1> m3;
        Eigen::Matrix<Scalar, 10, 1> b;

        b << Mr(5, 0), Mr(5, 1), -Mr(4, 0), -Mr(4, 1), Mr(5, 2), Mr(5, 3), Mr(5, 4) - Mr(4, 2), Mr(5, 5) - Mr(4, 3),
            -Mr(4, 4), -Mr(4, 5);

        for (int i = 0; i < nsols; i++) {
            Scalar l1 = Ls(0, i);
            Scalar l2 = Ls(1, i);
            Scalar l1l1 = l1 * l1;
            Scalar l1l2 = l1 * l2;
            Scalar f23;

            m1 << l1l2, l1, l2, 1;
            m2 << l1l2 * l1, l1l1, l1l2, l1, l2, 1;
            f23 = -b.template block<6, 1>(4, 0).dot(m2) / b.template block<4, 1>(0, 0).dot(m1);
            m3 << l2 * f23, f23, l1l2, l1, l2, 1;

            Fs(0, i) = m3.dot(-Mr.row(0));
            Fs(1, i) = m3.dot(-Mr.row(2));
            Fs(2, i) = m3.dot(-Mr.row(5));
            Fs(3, i) = m3.dot(-Mr.row(1));
            Fs(4, i) = m3.dot(-Mr.row(3));
            Fs(5, i) = m3.dot(-Mr.row(7));
            Fs(6, i) = m3.dot(-Mr.row(9));
            Fs(7, i) = f23;
            Fs(8, i) = 1;
        }
    }

    return nsols;
}

int relpose_k2Fk1_10pt(const std::vector<Eigen::Vector3d> &x1, const std::vector<Eigen::Vector3d> &x2,
                       std::vector<ProjectiveImagePairWithDivisionCamera> *cam_pairs) {
    Eigen::MatrixXd X(10, 2), U(10, 2), Fs(9, 10), Ls(2, 10);

    for (int i = 0; i < 10; ++i) {
        X(i, 0) = x1[i](0);
        X(i, 1) = x1[i](1);
        U(i, 0) = x2[i](0);
        U(i, 1) = x2[i](1);
    }

    int n_sols = relpose_k2Fk1_10pt_solver(X, U, Fs, Ls);

    cam_pairs->clear();
    cam_pairs->reserve(n_sols);

    for (int i = 0; i < n_sols; ++i) {
        Eigen::Matrix3d F;
        F << Fs.col(i)[0], Fs.col(i)[1], Fs.col(i)[2], Fs.col(i)[3], Fs.col(i)[4], Fs.col(i)[5], Fs.col(i)[6],
            Fs.col(i)[7], Fs.col(i)[8];

        const double k1 = Ls(0, i);
        const double k2 = Ls(1, i);

        cam_pairs->emplace_back(F, k1, k2);
    }

    return n_sols;
}

} // namespace poselib
