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

#ifndef POSELIB_MISC_UNIVARIATE_H_
#define POSELIB_MISC_UNIVARIATE_H_

#include <Eigen/Eigen>
#include <complex>

namespace poselib {
namespace univariate {
/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
void solve_quadratic(double a, double b, double c, std::complex<double> roots[2]);

/* Solves the quadratic equation a*x^2 + b*x + c = 0. Only returns real roots */
int solve_quadratic_real(double a, double b, double c, double roots[2]);

/* Sign of component with largest magnitude */
double sign2(const std::complex<double> z);

/* Finds a single real root of x^3 + b*x^2 + c*x + d = 0 */
bool solve_cubic_single_real(double b, double c, double d, double &root);

/* Finds the real roots of x^3 + b*x^2 + c*x + d = 0 */
int solve_cubic_real(double b, double c, double d, double roots[3]);

/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0 */
void solve_quartic(double b, double c, double d, double e, std::complex<double> roots[4]);

/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0. Only returns real roots */
int solve_quartic_real(double b, double c, double d, double e, double roots[4]);

}; // namespace univariate
}; // namespace poselib

#endif