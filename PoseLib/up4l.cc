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

#include "up4l.h"
#include <iostream>

int pose_lib::up4l(const std::vector<Eigen::Vector3d> &x, const std::vector<Eigen::Vector3d> &X,
		   	 	   const std::vector<Eigen::Vector3d> &V, CameraPoseVector* output)
{

	Eigen::Matrix<double, 4, 4> M, C, K;
	M(0, 0) = V[0](1) * x[0](2) + V[0](2) * x[0](1);
	M(0, 1) = -V[0](0) * x[0](2) - V[0](2) * x[0](0);
	M(0, 2) = V[0](0) * x[0](1) - V[0](1) * x[0](0);
	M(0, 3) = V[0](0) * X[0](1) * x[0](2) + V[0](0) * X[0](2) * x[0](1) - V[0](1) * X[0](0) * x[0](2) - V[0](1) * X[0](2) * x[0](0) - V[0](2) * X[0](0) * x[0](1) + V[0](2) * X[0](1) * x[0](0);
	M(1, 0) = V[1](1) * x[1](2) + V[1](2) * x[1](1);
	M(1, 1) = -V[1](0) * x[1](2) - V[1](2) * x[1](0);
	M(1, 2) = V[1](0) * x[1](1) - V[1](1) * x[1](0);
	M(1, 3) = V[1](0) * X[1](1) * x[1](2) + V[1](0) * X[1](2) * x[1](1) - V[1](1) * X[1](0) * x[1](2) - V[1](1) * X[1](2) * x[1](0) - V[1](2) * X[1](0) * x[1](1) + V[1](2) * X[1](1) * x[1](0);
	M(2, 0) = V[2](1) * x[2](2) + V[2](2) * x[2](1);
	M(2, 1) = -V[2](0) * x[2](2) - V[2](2) * x[2](0);
	M(2, 2) = V[2](0) * x[2](1) - V[2](1) * x[2](0);
	M(2, 3) = V[2](0) * X[2](1) * x[2](2) + V[2](0) * X[2](2) * x[2](1) - V[2](1) * X[2](0) * x[2](2) - V[2](1) * X[2](2) * x[2](0) - V[2](2) * X[2](0) * x[2](1) + V[2](2) * X[2](1) * x[2](0);
	M(3, 0) = V[3](1) * x[3](2) + V[3](2) * x[3](1);
	M(3, 1) = -V[3](0) * x[3](2) - V[3](2) * x[3](0);
	M(3, 2) = V[3](0) * x[3](1) - V[3](1) * x[3](0);
	M(3, 3) = V[3](0) * X[3](1) * x[3](2) + V[3](0) * X[3](2) * x[3](1) - V[3](1) * X[3](0) * x[3](2) - V[3](1) * X[3](2) * x[3](0) - V[3](2) * X[3](0) * x[3](1) + V[3](2) * X[3](1) * x[3](0);

	C(0, 0) = -2 * V[0](0) * x[0](2);
	C(0, 1) = -2 * V[0](1) * x[0](2);
	C(0, 2) = 2 * V[0](0) * x[0](0) + 2 * V[0](1) * x[0](1);
	C(0, 3) = 2 * V[0](0) * X[0](2) * x[0](0) - 2 * V[0](2) * X[0](0) * x[0](0) + 2 * V[0](1) * X[0](2) * x[0](1) - 2 * V[0](2) * X[0](1) * x[0](1);
	C(1, 0) = -2 * V[1](0) * x[1](2);
	C(1, 1) = -2 * V[1](1) * x[1](2);
	C(1, 2) = 2 * V[1](0) * x[1](0) + 2 * V[1](1) * x[1](1);
	C(1, 3) = 2 * V[1](0) * X[1](2) * x[1](0) - 2 * V[1](2) * X[1](0) * x[1](0) + 2 * V[1](1) * X[1](2) * x[1](1) - 2 * V[1](2) * X[1](1) * x[1](1);
	C(2, 0) = -2 * V[2](0) * x[2](2);
	C(2, 1) = -2 * V[2](1) * x[2](2);
	C(2, 2) = 2 * V[2](0) * x[2](0) + 2 * V[2](1) * x[2](1);
	C(2, 3) = 2 * V[2](0) * X[2](2) * x[2](0) - 2 * V[2](2) * X[2](0) * x[2](0) + 2 * V[2](1) * X[2](2) * x[2](1) - 2 * V[2](2) * X[2](1) * x[2](1);
	C(3, 0) = -2 * V[3](0) * x[3](2);
	C(3, 1) = -2 * V[3](1) * x[3](2);
	C(3, 2) = 2 * V[3](0) * x[3](0) + 2 * V[3](1) * x[3](1);
	C(3, 3) = 2 * V[3](0) * X[3](2) * x[3](0) - 2 * V[3](2) * X[3](0) * x[3](0) + 2 * V[3](1) * X[3](2) * x[3](1) - 2 * V[3](2) * X[3](1) * x[3](1);

	K(0, 0) = V[0](2) * x[0](1) - V[0](1) * x[0](2);
	K(0, 1) = V[0](0) * x[0](2) - V[0](2) * x[0](0);
	K(0, 2) = V[0](1) * x[0](0) - V[0](0) * x[0](1);
	K(0, 3) = V[0](0) * X[0](1) * x[0](2) - V[0](0) * X[0](2) * x[0](1) - V[0](1) * X[0](0) * x[0](2) + V[0](1) * X[0](2) * x[0](0) + V[0](2) * X[0](0) * x[0](1) - V[0](2) * X[0](1) * x[0](0);
	K(1, 0) = V[1](2) * x[1](1) - V[1](1) * x[1](2);
	K(1, 1) = V[1](0) * x[1](2) - V[1](2) * x[1](0);
	K(1, 2) = V[1](1) * x[1](0) - V[1](0) * x[1](1);
	K(1, 3) = V[1](0) * X[1](1) * x[1](2) - V[1](0) * X[1](2) * x[1](1) - V[1](1) * X[1](0) * x[1](2) + V[1](1) * X[1](2) * x[1](0) + V[1](2) * X[1](0) * x[1](1) - V[1](2) * X[1](1) * x[1](0);
	K(2, 0) = V[2](2) * x[2](1) - V[2](1) * x[2](2);
	K(2, 1) = V[2](0) * x[2](2) - V[2](2) * x[2](0);
	K(2, 2) = V[2](1) * x[2](0) - V[2](0) * x[2](1);
	K(2, 3) = V[2](0) * X[2](1) * x[2](2) - V[2](0) * X[2](2) * x[2](1) - V[2](1) * X[2](0) * x[2](2) + V[2](1) * X[2](2) * x[2](0) + V[2](2) * X[2](0) * x[2](1) - V[2](2) * X[2](1) * x[2](0);
	K(3, 0) = V[3](2) * x[3](1) - V[3](1) * x[3](2);
	K(3, 1) = V[3](0) * x[3](2) - V[3](2) * x[3](0);
	K(3, 2) = V[3](1) * x[3](0) - V[3](0) * x[3](1);
	K(3, 3) = V[3](0) * X[3](1) * x[3](2) - V[3](0) * X[3](2) * x[3](1) - V[3](1) * X[3](0) * x[3](2) + V[3](1) * X[3](2) * x[3](0) + V[3](2) * X[3](0) * x[3](1) - V[3](2) * X[3](1) * x[3](0);


	Eigen::Matrix<double, 8, 8> A;
	A.block<4, 4>(0, 0) = C;
	A.block<4, 4>(0, 4) = K;
	A.block<4, 4>(4, 0).setIdentity();
	A.block<4, 4>(4, 4).setZero();
	A.block<4, 8>(0, 0) = -M.partialPivLu().solve(A.block<4, 8>(0, 0));
	Eigen::EigenSolver<Eigen::Matrix<double, 8, 8>> es(A, true);	

	Eigen::Array<std::complex<double>, 8, 1> s = es.eigenvalues();
	Eigen::Array<std::complex<double>, 8, 8> v = es.eigenvectors();

	for (int i = 0; i < 8; ++i) {
		if (std::abs(s(i).imag()) > 1e-8)
			continue;

		pose_lib::CameraPose pose;
		double q = s(i).real();				
		double q2 = q * q;
		double cq = (1 - q2) / (1 + q2);
		double sq = 2 * q / (1 + q2);

		pose.R.setIdentity();
		pose.R(0, 0) = cq; pose.R(0, 1) = -sq;
		pose.R(1, 0) = sq; pose.R(1, 1) = cq;
		pose.t = v.block<3, 1>(4, i).real() / v(7, i).real();

		output->push_back(pose);
	}
	return output->size();
}
