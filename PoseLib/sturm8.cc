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

#include "sturm8.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>
#ifdef _MSC_VER
#  include <intrin.h>
#  define __builtin_popcount __popcnt
#endif

namespace pose_lib {
	namespace sturm8 {

		// Constructs the quotients needed for evaluating the sturm sequence.
		void build_sturm_seq(const double* fvec, double* svec) {

			double f[24];
			double* f1 = f;
			double* f2 = f1 + 9;
			double* f3 = f2 + 8;

			std::copy(fvec, fvec + 17, f);

			for (int i = 0; i < 7; ++i) {
				const double q1 = f1[8 - i] * f2[7 - i];
				const double q0 = f1[7 - i] * f2[7 - i] - f1[8 - i] * f2[6 - i];

				f3[0] = f1[0] - q0 * f2[0];
				for (int j = 1; j < 7 - i; ++j) {
					f3[j] = f1[j] - q1 * f2[j - 1] - q0 * f2[j];
				}
				const double c = -std::abs(f3[6 - i]);
				const double ci = 1.0 / c;
				for (int j = 0; j < 7 - i; ++j) {
					f3[j] = f3[j] * ci;
				}

				// juggle pointers (f1,f2,f3) -> (f2,f3,f1)
				double* tmp = f1;
				f1 = f2;
				f2 = f3;
				f3 = tmp;

				svec[3 * i] = q0;
				svec[3 * i + 1] = q1;
				svec[3 * i + 2] = c;
			}

			svec[21] = f1[0];
			svec[22] = f1[1];
			svec[23] = f2[0];
		}

		// Evaluates polynomial using Horner's method.
		// Assumes that f[DEG] = 1.0
		template <int DEG>
		inline double polyval(const double* f, double x) {
			double fx = x + f[DEG - 1];
			for (int i = DEG - 2; i >= 0; --i) {
				fx = x * fx + f[i];
			}
			return fx;
		}

		// Evaluates the sturm sequence and counts the number of sign changes
		inline int signchanges(const double svec[24], double x) {
			double f[9];

			f[8] = svec[23];
			f[7] = svec[21] + x * svec[22];

			for (int i = 6; i >= 0; --i) {
				f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
			}

			// In testing this turned out to be slightly faster compared to a naive loop
			unsigned int S = ((f[8] < 0) << 8) | ((f[7] < 0) << 7) | ((f[6] < 0) << 6) | ((f[5] < 0) << 5) | ((f[4] < 0) << 4) | ((f[3] < 0) << 3) | ((f[2] < 0) << 2) | ((f[1] < 0) << 1) | (f[0] < 0);
			return __builtin_popcount((S ^ (S >> 1)) & 0xFF);
		}

		// Computes the Cauchy bound on the real roots.
		// Experiments with more complicated (expensive) bounds did not seem to have a good trade-off.
		inline double get_bounds(const double fvec[45]) {
			double f1 = std::max(std::abs(fvec[0]), std::abs(fvec[1]));
			double f2 = std::max(std::abs(fvec[2]), std::abs(fvec[3]));
			double f3 = std::max(std::abs(fvec[4]), std::abs(fvec[5]));
			double f4 = std::max(std::abs(fvec[6]), std::abs(fvec[7]));
			return 1.0 + std::max(std::max(f1, f2), std::max(f3, f4));
		}

		// Applies Ridder's bracketing method until we get close to root, followed by newton iterations
		void ridders_method_newton(const double fvec[17], double a, double b, double roots[8], int& n_roots, double tol) {
			double fa = polyval<8>(fvec, a);
			double fb = polyval<8>(fvec, b);

			if (!((fa < 0) ^ (fb < 0)))
				return;

			const double tol_newton = 1e-3;

			for (int iter = 0; iter < 30; ++iter) {
				if (std::abs(a - b) < tol_newton) {
					break;
				}
				const double c = (a + b) * 0.5;
				const double fc = polyval<8>(fvec, c);
				const double s = std::sqrt(fc * fc - fa * fb);
				if (!s)
					break;
				const double d = (fa < fb) ? c + (a - c) * fc / s : c + (c - a) * fc / s;
				const double fd = polyval<8>(fvec, d);

				if (fd >= 0 ? (fc < 0) : (fc > 0)) {
					a = c; fa = fc;
					b = d; fb = fd;
				}
				else if (fd >= 0 ? (fa < 0) : (fa > 0)) {
					b = d; fb = fd;
				}
				else {
					a = d; fa = fd;
				}
			}

			// We switch to Newton's method once we are close to the root
			double x = (a + b) * 0.5;

			double fx, fpx, dx;
			const double* fpvec = fvec + 9;
			for (int iter = 0; iter < 10; ++iter) {
				fx = polyval<8>(fvec, x);
				if (std::abs(fx) < tol) {
					break;
				}
				fpx = 8.0 * polyval<7>(fpvec, x);
				dx = fx / fpx;
				x = x - dx;
				if (std::abs(dx) < tol) {
					break;
				}
			}

			roots[n_roots++] = x;

		}



		void isolate_roots(const double fvec[17], const double svec[24], double a, double b, int sa, int sb, double roots[8], int& n_roots, double tol, int depth) {
			if (depth > 30)
				return;

			int n_rts = sa - sb;

			double sz = b - a;

			if (n_rts > 1) {
				if (sz < 1e-8) {
					return; // Roots might be too close to isolate (potentially double root)
				}
				else {
					double c = (a + b) * 0.5;
					int sc = signchanges(svec, c);
					isolate_roots(fvec, svec, a, c, sa, sc, roots, n_roots, tol, depth + 1);
					isolate_roots(fvec, svec, c, b, sc, sb, roots, n_roots, tol, depth + 1);
				}
			} else if (n_rts == 1) {
				ridders_method_newton(fvec, a, b, roots, n_roots, tol);
			}
		}


		int bisect_sturm(double c0, double c1, double c2, double c3, double c4, double c5, double c6, double c7, double c8, double roots[8], double tol)
		{
			if (c8 == 0.0)
				return 0;

			double fvec[17];
			double svec[24];

			// fvec is the polynomial and its first derivative.
			double c8i = 1.0 / c8;
			fvec[0] = c0 * c8i;
			fvec[1] = c1 * c8i;
			fvec[2] = c2 * c8i;
			fvec[3] = c3 * c8i;
			fvec[4] = c4 * c8i;
			fvec[5] = c5 * c8i;
			fvec[6] = c6 * c8i;
			fvec[7] = c7 * c8i;
			fvec[8] = 1.0;			

			// Compute the derivative with normalized coefficients
			fvec[9] = fvec[1] * 0.125;
			fvec[10] = fvec[2] * 0.25;
			fvec[11] = fvec[3] * 0.375;
			fvec[12] = fvec[4] * 0.5;
			fvec[13] = fvec[5] * 0.625;
			fvec[14] = fvec[6] * 0.75;
			fvec[15] = fvec[7] * 0.875;
			fvec[16] = 1.0;

			// Compute sturm sequences
			build_sturm_seq(fvec, svec);

			// All real roots are in the interval [-r0, r0]
			double r0 = get_bounds(fvec);
			double a = -r0;
			double b = r0;

			int sa = signchanges(svec, a);
			int sb = signchanges(svec, b);

			int n_roots = sa - sb;
			if (n_roots == 0)
				return 0;

			n_roots = 0;
			isolate_roots(fvec, svec, a, b, sa, sb, roots, n_roots, tol, 0);

			return n_roots;
		}

	}
}