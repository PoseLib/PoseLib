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

#include "r6p.h"
#include "prepare_1lin.h"
#include "PoseLib/misc/danilevsky.h"
#include "PoseLib/misc/sturm_hartley.h"
#include "PoseLib/misc/utils.h"

namespace poselib {


std::vector<Eigen::Vector3d> solve_xy_from_AM(double * z, int nroots, Eigen::MatrixXd AM) {
	int ind[] = { 5, 6, 7, 9, 10, 12, 15, 16, 18, 22, 26, 27, 29, 33, 38, 43, 44, 47, 51, 56, 63 };
	Eigen::MatrixXd AMs(21, 64);
	double zi[8];
	std::vector<Eigen::Vector3d> sols;
	for (int i = 0; i < 21; i++)
	{
		AMs.row(i) = AM.row(ind[i]);
	}
	for (int i = 0; i < nroots; i++)
	{
		zi[0] = z[i];
		for (int j = 1; j < 8; j++)
		{
			zi[j] = zi[j - 1] * z[i];
		}
		Eigen::MatrixXd AA = Eigen::MatrixXd::Zero(21, 21);
		AA.col(0) = AMs.col(5);
		AA.col(1) = AMs.col(6);
		AA.col(2) = AMs.col(9);
		AA.col(3) = AMs.col(15);
		AA.col(4) = AMs.col(26);
		AA.col(5) = AMs.col(43);
		AA.col(6) = AMs.col(4) + zi[0] * AMs.col(7);
		AA.col(7) = AMs.col(8) + zi[0] * AMs.col(10);
		AA.col(8) = AMs.col(14) + zi[0] * AMs.col(16);
		AA.col(9) = AMs.col(25) + zi[0] * AMs.col(27);
		AA.col(10) = AMs.col(42) + zi[0] * AMs.col(44);
		AA.col(11) = AMs.col(3) + zi[0] * AMs.col(11) + zi[1] * AMs.col(12);
		AA.col(12) = AMs.col(13) + zi[0] * AMs.col(17) + zi[1] * AMs.col(18);
		AA.col(13) = AMs.col(24) + zi[0] * AMs.col(28) + zi[1] * AMs.col(29);
		AA.col(14) = AMs.col(41) + zi[0] * AMs.col(45) + zi[1] * AMs.col(46) + zi[2] * AMs.col(47);
		AA.col(15) = AMs.col(2) + zi[0] * AMs.col(19) + zi[1] * AMs.col(20) + zi[2] * AMs.col(21) + zi[3] * AMs.col(22);
		AA.col(16) = AMs.col(23) + zi[0] * AMs.col(30) + zi[1] * AMs.col(31) + zi[2] * AMs.col(32) + zi[3] * AMs.col(33);
		AA.col(17) = AMs.col(40) + zi[0] * AMs.col(48) + zi[1] * AMs.col(49) + zi[2] * AMs.col(50) + zi[3] * AMs.col(51);
		AA.col(18) = AMs.col(1) + zi[0] * AMs.col(34) + zi[1] * AMs.col(35) + zi[2] * AMs.col(36) + zi[3] * AMs.col(37) + zi[4] * AMs.col(38);
		AA.col(19) = AMs.col(39) + zi[0] * AMs.col(52) + zi[1] * AMs.col(53) + zi[2] * AMs.col(54) + zi[3] * AMs.col(55) + zi[4] * AMs.col(56);
		AA.col(20) = AMs.col(0) + zi[0] * AMs.col(57) + zi[1] * AMs.col(58) + zi[2] * AMs.col(59) + zi[3] * AMs.col(60) + zi[4] * AMs.col(61) + zi[5] * AMs.col(62) + zi[6] * AMs.col(63);
		AA(0, 0) = AA(0, 0) - zi[0];
		AA(1, 1) = AA(1, 1) - zi[0];
		AA(3, 2) = AA(3, 2) - zi[0];
		AA(6, 3) = AA(6, 3) - zi[0];
		AA(10, 4) = AA(10, 4) - zi[0];
		AA(15, 5) = AA(15, 5) - zi[0];
		AA(2, 6) = AA(2, 6) - zi[1];
		AA(4, 7) = AA(4, 7) - zi[1];
		AA(7, 8) = AA(7, 8) - zi[1];
		AA(11, 9) = AA(11, 9) - zi[1];
		AA(16, 10) = AA(16, 10) - zi[1];
		AA(5, 11) = AA(5, 11) - zi[2];
		AA(8, 12) = AA(8, 12) - zi[2];
		AA(12, 13) = AA(12, 13) - zi[2];
		AA(17, 14) = AA(17, 14) - zi[3];
		AA(9, 15) = AA(9, 15) - zi[4];
		AA(13, 16) = AA(13, 16) - zi[4];
		AA(18, 17) = AA(18, 17) - zi[4];
		AA(14, 18) = AA(14, 18) - zi[5];
		AA(19, 19) = AA(19, 19) - zi[5];
		AA(20, 20) = AA(20, 20) - zi[7];
		Eigen::VectorXd s = AA.leftCols(20).colPivHouseholderQr().solve(-AA.col(20));
		sols.push_back(Eigen::Vector3d(s(18), s(19), zi[0]));
	}

	return sols;
}

int r6pSingleLin(const  Eigen::MatrixXd & X, const  Eigen::MatrixXd & u, const Eigen::MatrixXd & ud, int direction, double r0, int maxpow, RSCameraPoseVector * results) {
	Eigen::VectorXd coeffs(1260);
	Eigen::MatrixXd CC = Eigen::MatrixXd::Zero(99, 163);

	double solver_input[2475];

	std::vector<Eigen::Vector3d> v;

	if (direction == 0) {
		R6P_cayley_linearized_prepare_data_C(X.data(), u.data(), ud.data(), r0, solver_input);
	}
	else {
		R6P_cayley_linearized_prepare_data_y_C(X.data(), u.data(), ud.data(), r0, solver_input);
	}

	Eigen::Map<Eigen::MatrixXd> data(solver_input, 15, 165);

	for (int j = 0; j < 15; j++)
	{
		data.row(j) /= data.row(j).cwiseAbs().mean();
	}

	computeCoeffs(data, coeffs);
	setupEliminationTemplate(coeffs, CC);

	Eigen::MatrixXd C0 = CC.leftCols(99);
	Eigen::MatrixXd C1 = CC.rightCols(64);
	

	Eigen::MatrixXd alpha = C0.partialPivLu().solve(C1);

	Eigen::MatrixXd RR(85, 64);
	for (int j = 0; j < 21; j++)
	{
		RR.row(j) = -alpha.row(78 + j);
	}


	RR.bottomRows(64) = Eigen::MatrixXd::Identity(64, 64);

	int AM_ind[] = { 78, 55, 40, 32, 28, 0, 1, 2, 31, 3, 4, 33, 5, 38, 37, 6, 7, 39, 8, 41, 42, 43, 9, 51, 49, 48, 10, 11, 50, 12, 52, 53, 54, 13, 56, 57, 58, 59, 14, 73, 69, 66, 65, 15, 16, 67, 68, 17, 70, 71, 72, 18, 74, 75, 76, 77, 19, 79, 80, 81, 82, 83, 84, 20 };
	Eigen::MatrixXd AM(64, 64);
	for (int j = 0; j < 64; j++)
	{
		AM.row(j) = RR.row(AM_ind[j]);
	}

	double p[65];
	charpoly_danilevsky(AM, p);


	int order = 64;
	double proots[65];
	double roots[64];
	int nroots;
	for (int j = 0; j <= order; j++) {
		proots[order - j] = p[j];
	}
	find_real_roots_sturm(proots, order, roots, &nroots, maxpow, 0);


	Eigen::MatrixXcd sols(3, nroots);

	v = solve_xy_from_AM(roots, nroots, AM);


	for (int j = 0; j < nroots; j++)
	{
		Eigen::Matrix3d R;
		Eigen::Vector4d q;
		q(0) = 1;
		q.segment(1, 3) = v[j];
		q.normalize();
		quat2R(q, R);

		Eigen::Quaternion quat = Eigen::Quaternion<double>(R);
        Eigen::Vector4d quat_res;
        quat_res[0] = quat.w();
        quat_res[1] = quat.x();
        quat_res[2] = quat.y();
        quat_res[3] = quat.z();

		//std::cout << R;
		Eigen::MatrixXd A(18, 9);
		Eigen::VectorXd b(18);
		for (int k = 0; k < 6; k++)
		{
			double dr = u(k * 2 + direction) - r0;
			Eigen::Matrix<double, 3, 9> temp;
			temp << -dr*X_(R*Eigen::Vector3d(X(k * 3), X(k * 3 + 1), X(k * 3 + 2))), Eigen::Matrix3d::Identity(), dr*Eigen::Matrix3d::Identity();

			A.block(k * 3, 0, 3, 9) = X_(Eigen::Vector3d(u(k * 2), u(k * 2 + 1), 1))*temp;

			b.segment(k * 3, 3) = -X_(Eigen::Vector3d(u(k * 2), u(k * 2 + 1), 1))*R*Eigen::Vector3d(X(k * 3), X(k * 3 + 1), X(k * 3 + 2));

		}

		Eigen::VectorXd wCt = A.householderQr().solve(b);


		results->push_back({quat_res, wCt.segment(3,3), wCt.segment(0,3), wCt.segment(6,3)});

	}


	return 0;

}

int R6P1Lin(const Eigen::MatrixXd &X, const Eigen::MatrixXd &u, int direction, double r0, int maxpow, RSCameraPoseVector * results) {
	return r6pSingleLin(X, u, u, direction, r0, maxpow, results);
}

// 
int r6p(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, RSCameraPoseVector *output) {
    if (x.size() != 6 || X.size() != 6){
        std::cout << "Wrong number of input correspondences";
        return 0;
    }
    
    Eigen::MatrixXd xin(2,6);
    Eigen::MatrixXd Xin(3,6);
    for(size_t i = 0; i < x.size(); i++) {
        xin.col(i) = x[i];
    }

    for(size_t i = 0; i < X.size(); i++) {
        Xin.col(i) = X[i];
    }

    R6P1Lin(Xin, xin, 1, 0, 2, output);

    return output->size();
}

} // namespace poselib
