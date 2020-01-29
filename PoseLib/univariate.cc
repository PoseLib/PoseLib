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

#include <Eigen/Eigen>
#include <complex>
#include "univariate.h"

namespace pose_lib {
	namespace univariate {
		/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
		void solve_quadratic(double a, double b, double c, std::complex<double> roots[2]) {

			std::complex<double> b2m4ac = b * b - 4 * a * c;
			std::complex<double> sq = std::sqrt(b2m4ac);

			// Choose sign to avoid cancellations
			roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
			roots[1] = c / (a * roots[0]);
		}

		/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
		int solve_quadratic_real(double a, double b, double c, double roots[2]) {

			double b2m4ac = b * b - 4 * a * c;
			if(b2m4ac < 0)
				return 0;

			double sq = std::sqrt(b2m4ac);

			// Choose sign to avoid cancellations
			roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
			roots[1] = c / (a * roots[0]);

			return 2;
		}

		


		/* Sign of component with largest magnitude */
		inline double sign2(const std::complex<double> z) {
			if (std::abs(z.real()) > std::abs(z.imag()))
				return z.real() < 0 ? -1.0 : 1.0;
			else
				return z.imag() < 0 ? -1.0 : 1.0;
		}

		/* Sign of component with largest magnitude */
		inline double sign(const double z) {			
			return z < 0 ? -1.0 : 1.0;
		}

		void solve_cubic_single_real(double c2, double c1, double c0, double &root) {
			double a = c1 - c2*c2/3.0;
			double b = (2.0*c2*c2*c2 - 9.0*c2*c1)/27.0 + c0;
			double c = b*b/4.0 + a*a*a/27.0;
			if(c > 0) {
				c = std::sqrt(c);
				b *= -0.5;
				root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
			} else {
				c = 3.0*b/(2.0*a) * std::sqrt(-3.0/a);
				root = 2.0 * std::sqrt(-a/3.0) * std::cos(std::acos(c)/3.0) - c2 / 3.0;
			}
		}

		/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0 */
		void solve_quartic(double b, double c, double d, double e, std::complex<double> roots[4]) {


			// Find depressed quartic
			std::complex<double> p = c - 3.0 * b * b / 8.0;
			std::complex<double> q = b * b * b / 8.0 - 0.5 * b * c + d;
			std::complex<double> r = (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) / 256.0;

			// Resolvent cubic is now
			// U^3 + 2*p U^2 + (p^2 - 4*r) * U - q^2
			std::complex<double> bb = 2.0 * p;
			std::complex<double> cc = p * p - 4.0 * r;
			std::complex<double> dd = -q * q;

			
			// Solve resolvent cubic
			std::complex<double> d0 = bb * bb - 3.0 * cc;
			std::complex<double> d1 = 2.0 * bb * bb * bb - 9.0 * bb * cc + 27.0 * dd;

			std::complex<double> C3 = (d1.real() < 0) ? (d1 - sqrt(d1 * d1 - 4.0 * d0 * d0 * d0)) / 2.0 : (d1 + sqrt(d1 * d1 - 4.0 * d0 * d0 * d0)) / 2.0;

			std::complex<double> C;
			if (C3.real() < 0)
				C = -std::pow(-C3, 1.0 / 3);
			else
				C = std::pow(C3, 1.0 / 3);

			std::complex<double> u2 = (bb + C + d0 / C) / -3.0;

			

			//std::complex<double> db = u2 * u2 * u2 + bb * u2 * u2 + cc * u2 + dd;

			std::complex<double> u = sqrt(u2);

			std::complex<double> s = -u;
			std::complex<double> t = (p + u * u + q / u) / 2.0;
			std::complex<double> v = (p + u * u - q / u) / 2.0;

			roots[0] = (-u - sign2(u) * sqrt(u * u - 4.0 * v)) / 2.0;
			roots[1] = v / roots[0];
			roots[2] = (-s - sign2(s) * sqrt(s * s - 4.0 * t)) / 2.0;
			roots[3] = t / roots[2];

			for (int i = 0; i < 4; i++) {
				roots[i] = roots[i] - b / 4.0;

				// do one step of newton refinement
				std::complex<double> x = roots[i];
				std::complex<double> x2 = x * x;
				std::complex<double> x3 = x * x2;
				std::complex<double> dx = -(x2 * x2 + b * x3 + c * x2 + d * x + e) / (4.0 * x3 + 3.0 * b * x2 + 2.0 * c * x + d);
				roots[i] = x + dx;
			}
		}

		/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0 */
		int solve_quartic_real(double b, double c, double d, double e, double roots[4]) {


			// Find depressed quartic
			double p = c - 3.0 * b * b / 8.0;
			double q = b * b * b / 8.0 - 0.5 * b * c + d;
			double r = (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) / 256.0;

			// Resolvent cubic is now
			// U^3 + 2*p U^2 + (p^2 - 4*r) * U - q^2
			double bb = 2.0 * p;
			double cc = p * p - 4.0 * r;
			double dd = -q * q;

			// Solve resolvent cubic
			double u2;
			solve_cubic_single_real(bb,cc,dd,u2);
			
			if(u2 < 0)
				return 0;

			double u = sqrt(u2);

			double s = -u;
			double t = (p + u * u + q / u) / 2.0;
			double v = (p + u * u - q / u) / 2.0;

			int sols = 0;
			double disc = u * u - 4.0 * v;
			if(disc > 0) {
				roots[0] = (-u - sign(u) * std::sqrt(disc)) / 2.0;
				roots[1] = v / roots[0];
				sols += 2;
			}
			disc = s * s - 4.0 * t;
			if(disc > 0) {
				roots[sols] = (-s - sign(s) * std::sqrt(disc)) / 2.0;
				roots[sols+1] = t / roots[sols];
				sols += 2;
			}

			for (int i = 0; i < sols; i++) {
				roots[i] = roots[i] - b / 4.0;

				// do one step of newton refinement
				double x = roots[i];
				double x2 = x * x;
				double x3 = x * x2;
				double dx = -(x2 * x2 + b * x3 + c * x2 + d * x + e) / (4.0 * x3 + 3.0 * b * x2 + 2.0 * c * x + d);
				roots[i] = x + dx;
			}
			return sols;
		}

	};
};
