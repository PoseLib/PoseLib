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

#include "univariate.h"

#include <Eigen/Eigen>
#include <complex>

namespace poselib {
namespace univariate {
/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
void solve_quadratic(Real a, Real b, Real c, std::complex<Real> roots[2]) {

    std::complex<Real> b2m4ac = b * b - 4 * a * c;
    std::complex<Real> sq = std::sqrt(b2m4ac);

    // Choose sign to avoid cancellations
    roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
    roots[1] = c / (a * roots[0]);
}

/* Solves the quadratic equation a*x^2 + b*x + c = 0 */
int solve_quadratic_real(Real a, Real b, Real c, Real roots[2]) {

    Real b2m4ac = b * b - 4 * a * c;
    if (b2m4ac < 0)
        return 0;

    Real sq = std::sqrt(b2m4ac);

    // Choose sign to avoid cancellations
    roots[0] = (b > 0) ? (2 * c) / (-b - sq) : (2 * c) / (-b + sq);
    roots[1] = c / (a * roots[0]);

    return 2;
}

/* Sign of component with largest magnitude */
inline Real sign2(const std::complex<Real> z) {
    if (std::abs(z.real()) > std::abs(z.imag()))
        return z.real() < 0 ? -1.0 : 1.0;
    else
        return z.imag() < 0 ? -1.0 : 1.0;
}

/* Sign of component with largest magnitude */
inline Real sign(const Real z) { return z < 0 ? -1.0 : 1.0; }

bool solve_cubic_single_real(Real c2, Real c1, Real c0, Real &root) {
    Real a = c1 - c2 * c2 / 3.0;
    Real b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    Real c = b * b / 4.0 + a * a * a / 27.0;
    if (c != 0) {
        if (c > 0) {
            c = std::sqrt(c);
            b *= -0.5;
            root = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
            return true;
        } else {
            c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
            root = 2.0 * std::sqrt(-a / 3.0) * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
        }
    } else {
        root = -c2 / 3.0 + (a != 0 ? (3.0 * b / a) : 0);
    }
    return false;
}

int solve_cubic_real(Real c2, Real c1, Real c0, Real roots[3]) {
    Real a = c1 - c2 * c2 / 3.0;
    Real b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    Real c = b * b / 4.0 + a * a * a / 27.0;
    int n_roots;
    if (c > 0) {
        c = std::sqrt(c);
        b *= -0.5;
        roots[0] = std::cbrt(b + c) + std::cbrt(b - c) - c2 / 3.0;
        n_roots = 1;
    } else {
        c = 3.0 * b / (2.0 * a) * std::sqrt(-3.0 / a);
        Real d = 2.0 * std::sqrt(-a / 3.0);
        roots[0] = d * std::cos(std::acos(c) / 3.0) - c2 / 3.0;
        roots[1] = d * std::cos(std::acos(c) / 3.0 - 2.09439510239319526263557236234192) - c2 / 3.0; // 2*pi/3
        roots[2] = d * std::cos(std::acos(c) / 3.0 - 4.18879020478639052527114472468384) - c2 / 3.0; // 4*pi/3
        n_roots = 3;
    }

    // single newton iteration
    for (int i = 0; i < n_roots; ++i) {
        Real x = roots[i];
        Real x2 = x * x;
        Real x3 = x * x2;
        Real dx = -(x3 + c2 * x2 + c1 * x + c0) / (3 * x2 + 2 * c2 * x + c1);
        roots[i] += dx;
    }
    return n_roots;
}

/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0 */
void solve_quartic(Real b, Real c, Real d, Real e, std::complex<Real> roots[4]) {

    // Find depressed quartic
    std::complex<Real> p = c - 3.0 * b * b / 8.0;
    std::complex<Real> q = b * b * b / 8.0 - 0.5 * b * c + d;
    std::complex<Real> r = (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) / 256.0;

    // Resolvent cubic is now
    // U^3 + 2*p U^2 + (p^2 - 4*r) * U - q^2
    std::complex<Real> bb = (Real)2.0 * p;
    std::complex<Real> cc = p * p - (Real)4.0 * r;
    std::complex<Real> dd = -q * q;

    // Solve resolvent cubic
    std::complex<Real> d0 = bb * bb - (Real)3.0 * cc;
    std::complex<Real> d1 = (Real)2.0 * bb * bb * bb - (Real)9.0 * bb * cc + (Real)27.0 * dd;

    std::complex<Real> C3 = (d1.real() < 0) ? (d1 - sqrt(d1 * d1 - (Real)4.0 * d0 * d0 * d0))
                                            : (d1 + sqrt(d1 * d1 - (Real)4.0 * d0 * d0 * d0));
    C3 /= 2.0;

    std::complex<Real> C;
    if (C3.real() < 0)
        C = -std::pow(-C3, 1.0 / 3);
    else
        C = std::pow(C3, 1.0 / 3);

    std::complex<Real> u2 = (bb + C + d0 / C) / (Real)-3.0;

    // std::complex<Real> db = u2 * u2 * u2 + bb * u2 * u2 + cc * u2 + dd;

    std::complex<Real> u = sqrt(u2);

    std::complex<Real> s = -u;
    std::complex<Real> t = (p + u * u + q / u) / (Real)2.0;
    std::complex<Real> v = (p + u * u - q / u) / (Real)2.0;

    roots[0] = (-u - sign2(u) * sqrt(u * u - (Real)4.0 * v)) / (Real)2.0;
    roots[1] = v / roots[0];
    roots[2] = (-s - sign2(s) * sqrt(s * s - (Real)4.0 * t)) / (Real)2.0;
    roots[3] = t / roots[2];

    for (int i = 0; i < 4; i++) {
        roots[i] = roots[i] - b / 4.0f;

        // do one step of newton refinement
        std::complex<Real> x = roots[i];
        std::complex<Real> x2 = x * x;
        std::complex<Real> x3 = x * x2;
        std::complex<Real> dx =
            -(x2 * x2 + b * x3 + c * x2 + d * x + e) / ((Real)4.0 * x3 + (Real)3.0 * b * x2 + (Real)2.0 * c * x + d);
        roots[i] = x + dx;
    }
}

/* Solves the quartic equation x^4 + b*x^3 + c*x^2 + d*x + e = 0 */
int solve_quartic_real(Real b, Real c, Real d, Real e, Real roots[4]) {

    // Find depressed quartic
    Real p = c - 3.0 * b * b / 8.0;
    Real q = b * b * b / 8.0 - 0.5 * b * c + d;
    Real r = (-3.0 * b * b * b * b + 256.0 * e - 64.0 * b * d + 16.0 * b * b * c) / 256.0;

    // Resolvent cubic is now
    // U^3 + 2*p U^2 + (p^2 - 4*r) * U - q^2
    Real bb = 2.0 * p;
    Real cc = p * p - 4.0 * r;
    Real dd = -q * q;

    // Solve resolvent cubic
    Real u2;
    solve_cubic_single_real(bb, cc, dd, u2);

    if (u2 < 0)
        return 0;

    Real u = sqrt(u2);

    Real s = -u;
    Real t = (p + u * u + q / u) / 2.0;
    Real v = (p + u * u - q / u) / 2.0;

    int sols = 0;
    Real disc = u * u - 4.0 * v;
    if (disc > 0) {
        roots[0] = (-u - sign(u) * std::sqrt(disc)) / 2.0;
        roots[1] = v / roots[0];
        sols += 2;
    }
    disc = s * s - 4.0 * t;
    if (disc > 0) {
        roots[sols] = (-s - sign(s) * std::sqrt(disc)) / 2.0;
        roots[sols + 1] = t / roots[sols];
        sols += 2;
    }

    for (int i = 0; i < sols; i++) {
        roots[i] = roots[i] - b / 4.0;

        // do one step of newton refinement
        Real x = roots[i];
        Real x2 = x * x;
        Real x3 = x * x2;
        Real dx = -(x2 * x2 + b * x3 + c * x2 + d * x + e) / (4.0 * x3 + 3.0 * b * x2 + 2.0 * c * x + d);
        roots[i] = x + dx;
    }
    return sols;
}

}; // namespace univariate
}; // namespace poselib
