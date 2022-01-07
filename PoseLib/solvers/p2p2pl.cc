// Copyright (c) 2020, Viktor Larsson
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

#include "p2p2pl.h"

void p2p2l_fast_eigenvector_solver(double *eigv, int neig, Eigen::Matrix<double, 16, 16> &AM,
                                   Eigen::Matrix<double, 2, 16> &sols) {
    static const int ind[] = {4, 5, 7, 10, 15};
    // Truncated action matrix containing non-trivial rows
    Eigen::Matrix<double, 5, 16> AMs;
    double zi[6];

    for (int i = 0; i < 5; i++) {
        AMs.row(i) = AM.row(ind[i]);
    }
    for (int i = 0; i < neig; i++) {
        zi[0] = eigv[i];
        for (int j = 1; j < 6; j++) {
            zi[j] = zi[j - 1] * eigv[i];
        }
        Eigen::Matrix<double, 5, 5> AA;
        AA.col(0) = AMs.col(4);
        AA.col(1) = AMs.col(3) + zi[0] * AMs.col(5);
        AA.col(2) = AMs.col(2) + zi[0] * AMs.col(6) + zi[1] * AMs.col(7);
        AA.col(3) = AMs.col(1) + zi[0] * AMs.col(8) + zi[1] * AMs.col(9) + zi[2] * AMs.col(10);
        AA.col(4) = AMs.col(0) + zi[0] * AMs.col(11) + zi[1] * AMs.col(12) + zi[2] * AMs.col(13) + zi[3] * AMs.col(14) +
                    zi[4] * AMs.col(15);
        AA(0, 0) = AA(0, 0) - zi[0];
        AA(1, 1) = AA(1, 1) - zi[1];
        AA(2, 2) = AA(2, 2) - zi[2];
        AA(3, 3) = AA(3, 3) - zi[3];
        AA(4, 4) = AA(4, 4) - zi[5];

        Eigen::Matrix<double, 4, 1> s = AA.leftCols(4).colPivHouseholderQr().solve(-AA.col(4));
        sols(0, i) = s(3);
        sols(1, i) = zi[0];
    }
}

int poselib::p2p2pl(const std::vector<Eigen::Vector3d> &xp0, const std::vector<Eigen::Vector3d> &Xp0,
                    const std::vector<Eigen::Vector3d> &x0, const std::vector<Eigen::Vector3d> &X0,
                    const std::vector<Eigen::Vector3d> &V0, CameraPoseVector *output) {

    // Change world coordinate system
    Eigen::Vector3d t0 = Xp0[0];
    Eigen::Matrix<double, 3, 2> X;
    X << X0[0] - t0, X0[1] - t0;
    Eigen::Matrix<double, 3, 2> Xp;
    Xp << Xp0[0] - t0, Xp0[1] - t0;
    Eigen::Matrix<double, 3, 2> V;
    V << V0[0], V0[1];

    double s0 = Xp.col(1).norm();
    Xp /= s0;
    X /= s0;

    Eigen::Matrix3d R0 =
        Eigen::Quaternion<double>::FromTwoVectors(Xp.col(1), Eigen::Vector3d(1.0, 0.0, 0.0)).toRotationMatrix();
    Xp = R0 * Xp;
    X = R0 * X;
    V = R0 * V;

    // Change image coordinate system
    Eigen::Matrix<double, 3, 2> x;
    x << x0[0].normalized(), x0[1].normalized();
    Eigen::Matrix<double, 3, 2> xp;
    xp << xp0[0].normalized(), xp0[1].normalized();

    Eigen::Matrix3d R1 =
        Eigen::Quaternion<double>::FromTwoVectors(xp.col(0), Eigen::Vector3d(0.0, 0.0, 1.0)).toRotationMatrix();
    xp = R1 * xp;
    x = R1 * x;

    Eigen::Matrix3d R2;
    R2.setIdentity();
    Eigen::Vector2d a = xp.block<2, 1>(0, 1).normalized();
    R2(0, 0) = a(0);
    R2(0, 1) = a(1);
    R2(1, 0) = -a(1);
    R2(1, 1) = a(0);

    xp = R2 * xp;
    x = R2 * x;
    double u = xp(2, 1) / xp(0, 1);

    double coeffs[30];
    coeffs[0] = V(1, 0) * X(0, 0) * x(2, 0) - V(0, 0) * X(2, 0) * x(1, 0) - V(0, 0) * X(1, 0) * x(2, 0) +
                V(1, 0) * X(2, 0) * x(0, 0) + V(2, 0) * X(0, 0) * x(1, 0) - V(2, 0) * X(1, 0) * x(0, 0) +
                V(0, 0) * u * x(1, 0) - V(1, 0) * u * x(0, 0);
    coeffs[1] = 2 * V(1, 0) * x(0, 0) - 2 * V(0, 0) * x(1, 0) + 2 * V(0, 0) * X(1, 0) * x(0, 0) -
                2 * V(1, 0) * X(0, 0) * x(0, 0) + 2 * V(1, 0) * X(2, 0) * x(2, 0) - 2 * V(2, 0) * X(1, 0) * x(2, 0) +
                2 * V(2, 0) * u * x(1, 0);
    coeffs[2] = 2 * V(0, 0) * X(1, 0) * x(1, 0) - 2 * V(1, 0) * X(0, 0) * x(1, 0) - 2 * V(0, 0) * X(2, 0) * x(2, 0) +
                2 * V(2, 0) * X(0, 0) * x(2, 0) - 2 * V(2, 0) * u * x(0, 0);
    coeffs[3] = 2 * V(2, 0) * X(0, 0) * x(1, 0) - 2 * V(0, 0) * X(2, 0) * x(1, 0) - 4 * V(2, 0) * x(1, 0) -
                2 * V(0, 0) * u * x(1, 0);
    coeffs[4] = 4 * V(2, 0) * x(0, 0) + 4 * V(0, 0) * X(2, 0) * x(0, 0) - 4 * V(2, 0) * X(0, 0) * x(0, 0) -
                4 * V(1, 0) * u * x(1, 0);
    coeffs[5] = V(0, 0) * X(1, 0) * x(2, 0) + V(0, 0) * X(2, 0) * x(1, 0) - V(1, 0) * X(0, 0) * x(2, 0) +
                V(1, 0) * X(2, 0) * x(0, 0) - V(2, 0) * X(0, 0) * x(1, 0) - V(2, 0) * X(1, 0) * x(0, 0) +
                V(0, 0) * u * x(1, 0) + V(1, 0) * u * x(0, 0);
    coeffs[6] = 2 * V(0, 0) * x(1, 0) + 2 * V(1, 0) * x(0, 0) + 2 * V(0, 0) * X(1, 0) * x(0, 0) -
                2 * V(1, 0) * X(0, 0) * x(0, 0) + 2 * V(1, 0) * X(2, 0) * x(2, 0) - 2 * V(2, 0) * X(1, 0) * x(2, 0) -
                2 * V(2, 0) * u * x(1, 0);
    coeffs[7] = 8 * V(1, 0) * x(1, 0) + 4 * V(0, 0) * X(1, 0) * x(1, 0) - 4 * V(1, 0) * X(0, 0) * x(1, 0);
    coeffs[8] = 2 * V(1, 0) * X(0, 0) * x(0, 0) - 2 * V(1, 0) * x(0, 0) - 2 * V(0, 0) * X(1, 0) * x(0, 0) -
                2 * V(0, 0) * x(1, 0) + 2 * V(1, 0) * X(2, 0) * x(2, 0) - 2 * V(2, 0) * X(1, 0) * x(2, 0) -
                2 * V(2, 0) * u * x(1, 0);
    coeffs[9] = V(0, 0) * X(1, 0) * x(2, 0) - V(0, 0) * X(2, 0) * x(1, 0) - V(1, 0) * X(0, 0) * x(2, 0) -
                V(1, 0) * X(2, 0) * x(0, 0) + V(2, 0) * X(0, 0) * x(1, 0) + V(2, 0) * X(1, 0) * x(0, 0) +
                V(0, 0) * u * x(1, 0) + V(1, 0) * u * x(0, 0);
    coeffs[10] = 4 * V(2, 0) * x(0, 0) + 4 * V(0, 0) * X(2, 0) * x(0, 0) - 4 * V(2, 0) * X(0, 0) * x(0, 0) +
                 4 * V(1, 0) * u * x(1, 0);
    coeffs[11] = 4 * V(2, 0) * x(1, 0) + 2 * V(0, 0) * X(2, 0) * x(1, 0) - 2 * V(2, 0) * X(0, 0) * x(1, 0) -
                 2 * V(0, 0) * u * x(1, 0);
    coeffs[12] = 2 * V(0, 0) * X(1, 0) * x(1, 0) - 2 * V(1, 0) * X(0, 0) * x(1, 0) + 2 * V(0, 0) * X(2, 0) * x(2, 0) -
                 2 * V(2, 0) * X(0, 0) * x(2, 0) + 2 * V(2, 0) * u * x(0, 0);
    coeffs[13] = 2 * V(0, 0) * x(1, 0) - 2 * V(1, 0) * x(0, 0) - 2 * V(0, 0) * X(1, 0) * x(0, 0) +
                 2 * V(1, 0) * X(0, 0) * x(0, 0) + 2 * V(1, 0) * X(2, 0) * x(2, 0) - 2 * V(2, 0) * X(1, 0) * x(2, 0) +
                 2 * V(2, 0) * u * x(1, 0);
    coeffs[14] = V(0, 0) * X(2, 0) * x(1, 0) - V(0, 0) * X(1, 0) * x(2, 0) + V(1, 0) * X(0, 0) * x(2, 0) -
                 V(1, 0) * X(2, 0) * x(0, 0) - V(2, 0) * X(0, 0) * x(1, 0) + V(2, 0) * X(1, 0) * x(0, 0) +
                 V(0, 0) * u * x(1, 0) - V(1, 0) * u * x(0, 0);
    coeffs[15] = V(1, 1) * X(0, 1) * x(2, 1) - V(0, 1) * X(2, 1) * x(1, 1) - V(0, 1) * X(1, 1) * x(2, 1) +
                 V(1, 1) * X(2, 1) * x(0, 1) + V(2, 1) * X(0, 1) * x(1, 1) - V(2, 1) * X(1, 1) * x(0, 1) +
                 V(0, 1) * u * x(1, 1) - V(1, 1) * u * x(0, 1);
    coeffs[16] = 2 * V(1, 1) * x(0, 1) - 2 * V(0, 1) * x(1, 1) + 2 * V(0, 1) * X(1, 1) * x(0, 1) -
                 2 * V(1, 1) * X(0, 1) * x(0, 1) + 2 * V(1, 1) * X(2, 1) * x(2, 1) - 2 * V(2, 1) * X(1, 1) * x(2, 1) +
                 2 * V(2, 1) * u * x(1, 1);
    coeffs[17] = 2 * V(0, 1) * X(1, 1) * x(1, 1) - 2 * V(1, 1) * X(0, 1) * x(1, 1) - 2 * V(0, 1) * X(2, 1) * x(2, 1) +
                 2 * V(2, 1) * X(0, 1) * x(2, 1) - 2 * V(2, 1) * u * x(0, 1);
    coeffs[18] = 2 * V(2, 1) * X(0, 1) * x(1, 1) - 2 * V(0, 1) * X(2, 1) * x(1, 1) - 4 * V(2, 1) * x(1, 1) -
                 2 * V(0, 1) * u * x(1, 1);
    coeffs[19] = 4 * V(2, 1) * x(0, 1) + 4 * V(0, 1) * X(2, 1) * x(0, 1) - 4 * V(2, 1) * X(0, 1) * x(0, 1) -
                 4 * V(1, 1) * u * x(1, 1);
    coeffs[20] = V(0, 1) * X(1, 1) * x(2, 1) + V(0, 1) * X(2, 1) * x(1, 1) - V(1, 1) * X(0, 1) * x(2, 1) +
                 V(1, 1) * X(2, 1) * x(0, 1) - V(2, 1) * X(0, 1) * x(1, 1) - V(2, 1) * X(1, 1) * x(0, 1) +
                 V(0, 1) * u * x(1, 1) + V(1, 1) * u * x(0, 1);
    coeffs[21] = 2 * V(0, 1) * x(1, 1) + 2 * V(1, 1) * x(0, 1) + 2 * V(0, 1) * X(1, 1) * x(0, 1) -
                 2 * V(1, 1) * X(0, 1) * x(0, 1) + 2 * V(1, 1) * X(2, 1) * x(2, 1) - 2 * V(2, 1) * X(1, 1) * x(2, 1) -
                 2 * V(2, 1) * u * x(1, 1);
    coeffs[22] = 8 * V(1, 1) * x(1, 1) + 4 * V(0, 1) * X(1, 1) * x(1, 1) - 4 * V(1, 1) * X(0, 1) * x(1, 1);
    coeffs[23] = 2 * V(1, 1) * X(0, 1) * x(0, 1) - 2 * V(1, 1) * x(0, 1) - 2 * V(0, 1) * X(1, 1) * x(0, 1) -
                 2 * V(0, 1) * x(1, 1) + 2 * V(1, 1) * X(2, 1) * x(2, 1) - 2 * V(2, 1) * X(1, 1) * x(2, 1) -
                 2 * V(2, 1) * u * x(1, 1);
    coeffs[24] = V(0, 1) * X(1, 1) * x(2, 1) - V(0, 1) * X(2, 1) * x(1, 1) - V(1, 1) * X(0, 1) * x(2, 1) -
                 V(1, 1) * X(2, 1) * x(0, 1) + V(2, 1) * X(0, 1) * x(1, 1) + V(2, 1) * X(1, 1) * x(0, 1) +
                 V(0, 1) * u * x(1, 1) + V(1, 1) * u * x(0, 1);
    coeffs[25] = 4 * V(2, 1) * x(0, 1) + 4 * V(0, 1) * X(2, 1) * x(0, 1) - 4 * V(2, 1) * X(0, 1) * x(0, 1) +
                 4 * V(1, 1) * u * x(1, 1);
    coeffs[26] = 4 * V(2, 1) * x(1, 1) + 2 * V(0, 1) * X(2, 1) * x(1, 1) - 2 * V(2, 1) * X(0, 1) * x(1, 1) -
                 2 * V(0, 1) * u * x(1, 1);
    coeffs[27] = 2 * V(0, 1) * X(1, 1) * x(1, 1) - 2 * V(1, 1) * X(0, 1) * x(1, 1) + 2 * V(0, 1) * X(2, 1) * x(2, 1) -
                 2 * V(2, 1) * X(0, 1) * x(2, 1) + 2 * V(2, 1) * u * x(0, 1);
    coeffs[28] = 2 * V(0, 1) * x(1, 1) - 2 * V(1, 1) * x(0, 1) - 2 * V(0, 1) * X(1, 1) * x(0, 1) +
                 2 * V(1, 1) * X(0, 1) * x(0, 1) + 2 * V(1, 1) * X(2, 1) * x(2, 1) - 2 * V(2, 1) * X(1, 1) * x(2, 1) +
                 2 * V(2, 1) * u * x(1, 1);
    coeffs[29] = V(0, 1) * X(2, 1) * x(1, 1) - V(0, 1) * X(1, 1) * x(2, 1) + V(1, 1) * X(0, 1) * x(2, 1) -
                 V(1, 1) * X(2, 1) * x(0, 1) - V(2, 1) * X(0, 1) * x(1, 1) + V(2, 1) * X(1, 1) * x(0, 1) +
                 V(0, 1) * u * x(1, 1) - V(1, 1) * u * x(0, 1);

    // Setup elimination template
    static const int coeffs0_ind[] = {
        0,  15, 1,  0,  15, 16, 2,  0,  15, 17, 3,  1,  0,  15, 16, 18, 4,  2,  1,  0,  16, 15, 17, 19, 5,  2,  0,
        15, 17, 20, 6,  3,  1,  0,  15, 16, 18, 21, 7,  4,  3,  2,  1,  0,  18, 17, 15, 16, 19, 22, 8,  5,  4,  2,
        1,  0,  16, 15, 19, 17, 20, 23, 5,  2,  17, 20, 9,  6,  3,  1,  16, 18, 21, 24, 10, 7,  6,  4,  3,  2,  1,
        17, 0,  21, 19, 15, 16, 18, 22, 25, 11, 8,  7,  5,  4,  3,  2,  1,  18, 15, 16, 22, 20, 17, 19, 23, 0,  26,
        8,  5,  4,  2,  19, 17, 23, 20, 5,  20, 9,  6,  3,  18, 21, 24, 12, 10, 9,  7,  6,  4,  3,  19, 1,  24, 22,
        16, 18, 21, 25, 27, 13, 11, 10, 8,  7,  6,  5,  4,  3,  21, 20, 2,  16, 18, 25, 23, 17, 19, 22, 26, 1,  0,
        15, 28, 11, 8,  7,  5,  4,  22, 17, 19, 26, 20, 23, 2,  9,  6,  21, 24, 12, 10, 9,  7,  6,  22, 3,  25, 18,
        21, 24, 27, 14, 13, 12, 11, 10, 9,  8,  7,  6,  24, 23, 4,  18, 21, 27, 26, 19, 22, 25, 28, 3,  1,  16, 29,
        13, 11, 10, 8,  7,  25, 5,  19, 22, 28, 20, 23, 26, 4,  2,  17, 8,  5,  23, 20};
    static const int coeffs1_ind[] = {
        14, 29, 14, 29, 12, 27, 14, 29, 12, 27, 9,  24, 12, 27, 9,  24, 9,  24, 12, 10, 9,  25, 6,  27, 21,
        24, 14, 13, 12, 28, 10, 24, 29, 25, 27, 9,  6,  21, 14, 13, 12, 11, 10, 9,  26, 7,  21, 24, 28, 22,
        25, 27, 29, 6,  3,  18, 14, 13, 27, 28, 29, 12, 10, 25, 14, 13, 12, 11, 25, 27, 26, 28, 29, 10, 7,
        22, 14, 13, 12, 11, 10, 27, 8,  22, 25, 29, 23, 26, 28, 7,  4,  19, 29, 14, 13, 28, 14, 28, 29, 13,
        11, 26, 14, 13, 29, 26, 28, 11, 8,  23, 13, 11, 28, 23, 26, 8,  5,  20, 11, 8,  26, 20, 23, 5};
    static const int C0_ind[] = {
        0,   23,  24,  25,  43,  47,  48,  50,  62,  71,  72,  73,  75,  87,  91,  95,  96,  97,  98,  100, 110, 114,
        115, 119, 120, 122, 125, 129, 134, 143, 144, 145, 147, 150, 154, 159, 163, 167, 168, 169, 170, 171, 172, 175,
        182, 183, 185, 186, 187, 191, 192, 193, 194, 196, 197, 200, 201, 205, 206, 210, 211, 215, 218, 221, 225, 230,
        240, 241, 243, 246, 250, 255, 259, 263, 264, 265, 266, 267, 268, 270, 271, 274, 275, 278, 279, 280, 281, 282,
        283, 287, 288, 289, 290, 291, 292, 293, 295, 296, 297, 300, 301, 302, 303, 305, 306, 307, 308, 311, 314, 316,
        317, 320, 321, 325, 326, 330, 341, 345, 361, 363, 366, 370, 375, 379, 384, 385, 386, 387, 388, 390, 391, 394,
        395, 398, 399, 400, 401, 402, 403, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421,
        422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 434, 436, 437, 439, 440, 441, 444, 445, 446, 449, 450, 452,
        459, 462, 466, 471, 481, 483, 484, 486, 487, 490, 491, 495, 496, 497, 498, 499, 504, 505, 506, 507, 508, 509,
        510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 530, 532, 533, 535,
        536, 537, 539, 540, 541, 542, 544, 545, 546, 548, 549, 550, 557, 560, 561, 565};
    static const int C1_ind[] = {
        21,  22,  35,  40,  45,  46,  54,  58,  59,  64,  69,  70,  78,  82,  83,  88,  102, 106, 123, 126, 127,
        130, 131, 135, 136, 137, 147, 150, 151, 154, 155, 156, 159, 160, 161, 164, 165, 166, 169, 171, 172, 174,
        175, 176, 178, 179, 180, 181, 183, 184, 185, 186, 187, 188, 189, 190, 199, 203, 204, 208, 209, 212, 213,
        214, 220, 223, 224, 227, 228, 229, 232, 233, 234, 236, 237, 238, 242, 244, 245, 247, 248, 249, 251, 252,
        253, 254, 256, 257, 258, 260, 261, 262, 276, 284, 285, 286, 296, 300, 301, 308, 309, 310, 317, 320, 321,
        324, 325, 332, 333, 334, 341, 344, 345, 348, 349, 356, 357, 358, 365, 368, 369, 372, 373, 380};

    Eigen::Matrix<double, 24, 24> C0;
    C0.setZero();
    Eigen::Matrix<double, 24, 16> C1;
    C1.setZero();
    for (int i = 0; i < 236; i++) {
        C0(C0_ind[i]) = coeffs[coeffs0_ind[i]];
    }
    for (int i = 0; i < 124; i++) {
        C1(C1_ind[i]) = coeffs[coeffs1_ind[i]];
    }

    Eigen::Matrix<double, 24, 16> C12 = C0.partialPivLu().solve(C1);

    // Setup action matrix
    Eigen::Matrix<double, 16, 16> AM;
    AM.setZero();
    AM(0, 11) = 1.0;
    AM(1, 8) = 1.0;
    AM(2, 6) = 1.0;
    AM(3, 5) = 1.0;
    AM.row(4) = -C12.row(19);
    AM.row(5) = -C12.row(20);
    AM(6, 7) = 1.0;
    AM.row(7) = -C12.row(21);
    AM(8, 9) = 1.0;
    AM(9, 10) = 1.0;
    AM.row(10) = -C12.row(22);
    AM(11, 12) = 1.0;
    AM(12, 13) = 1.0;
    AM(13, 14) = 1.0;
    AM(14, 15) = 1.0;
    AM.row(15) = -C12.row(23);

    // Solve for eigenvalues
    Eigen::EigenSolver<Eigen::Matrix<double, 16, 16>> es(AM, false);
    Eigen::Array<std::complex<double>, 16, 1> D = es.eigenvalues();

    int nroots = 0;
    double eigv[16];
    for (int i = 0; i < 16; i++) {
        if (std::abs(D(i).imag()) < 1e-6)
            eigv[nroots++] = D(i).real();
    }

    // Solve for the eigenvectors (exploiting their structure)
    Eigen::Matrix<double, 2, 16> sols;
    p2p2l_fast_eigenvector_solver(eigv, nroots, AM, sols);

    output->clear();
    for (int i = 0; i < nroots; ++i) {
        double a = 1.0;
        double b = sols(0, i);
        double c = sols(1, i);
        double d = -b * c;

        Eigen::Quaternion<double> q(a, b, c, d);

        CameraPose pose;
        pose.q << a, b, c, d;

        pose.t << 0.0, 0.0, u * (1.0 + b * b - c * c - d * d) - 2 * b * d + 2 * a * c;
        pose.t(2) /= pose.q.squaredNorm();

        pose.q.normalize();

        // Revert change of variable
        pose.q = rotmat_to_quat(R1.transpose() * R2.transpose() * pose.R() * R0);
        pose.t = R1.transpose() * R2.transpose() * pose.t;
        pose.t = pose.t * s0 - pose.R() * t0;

        output->push_back(pose);
    }

    return nroots;
}
