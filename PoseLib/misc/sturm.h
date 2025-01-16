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
#ifndef POSELIB_MISC_STURM_H_
#define POSELIB_MISC_STURM_H_

#include "PoseLib/real_matrix.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <vector>
#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

namespace poselib {
namespace sturm {

// Constructs the quotients needed for evaluating the sturm sequence.
template <int N> void build_sturm_seq(const real *fvec, real *svec) {

    real f[3 * N];
    real *f1 = f;
    real *f2 = f1 + N + 1;
    real *f3 = f2 + N;

    std::copy(fvec, fvec + (2 * N + 1), f);

    for (int i = 0; i < N - 1; ++i) {
        const real q1 = f1[N - i] * f2[N - 1 - i];
        const real q0 = f1[N - 1 - i] * f2[N - 1 - i] - f1[N - i] * f2[N - 2 - i];

        f3[0] = f1[0] - q0 * f2[0];
        for (int j = 1; j < N - 1 - i; ++j) {
            f3[j] = f1[j] - q1 * f2[j - 1] - q0 * f2[j];
        }
        const real c = -std::abs(f3[N - 2 - i]);
        const real ci = 1.0 / c;
        for (int j = 0; j < N - 1 - i; ++j) {
            f3[j] = f3[j] * ci;
        }

        // juggle pointers (f1,f2,f3) -> (f2,f3,f1)
        real *tmp = f1;
        f1 = f2;
        f2 = f3;
        f3 = tmp;

        svec[3 * i] = q0;
        svec[3 * i + 1] = q1;
        svec[3 * i + 2] = c;
    }

    svec[3 * N - 3] = f1[0];
    svec[3 * N - 2] = f1[1];
    svec[3 * N - 1] = f2[0];
}

// Evaluates polynomial using Horner's method.
// Assumes that f[N] = 1.0
template <int N> inline real polyval(const real *f, real x) {
    real fx = x + f[N - 1];
    for (int i = N - 2; i >= 0; --i) {
        fx = x * fx + f[i];
    }
    return fx;
}

// Daniel Thul is responsible for this template-trickery :)
template <int D> inline unsigned int flag_negative(const real *const f) {
    return ((f[D] < 0) << D) | flag_negative<D - 1>(f);
}
template <> inline unsigned int flag_negative<0>(const real *const f) { return f[0] < 0; }
// Evaluates the sturm sequence and counts the number of sign changes
template <int N, typename std::enable_if<(N < 32), void>::type * = nullptr>
inline int signchanges(const real *svec, real x) {

    real f[N + 1];
    f[N] = svec[3 * N - 1];
    f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];

    for (int i = N - 2; i >= 0; --i) {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    // In testing this turned out to be slightly faster compared to a naive loop
    unsigned int S = flag_negative<N>(f);

    return __builtin_popcount((S ^ (S >> 1)) & ~(0xFFFFFFFF << N));
}

template <int N, typename std::enable_if<(N >= 32), void>::type * = nullptr>
inline int signchanges(const real *svec, real x) {

    real f[N + 1];
    f[N] = svec[3 * N - 1];
    f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];

    for (int i = N - 2; i >= 0; --i) {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    int count = 0;
    bool neg1 = f[0] < 0;
    for (int i = 0; i < N; ++i) {
        bool neg2 = f[i + 1] < 0;
        if (neg1 ^ neg2) {
            ++count;
        }
        neg1 = neg2;
    }
    return count;
}

// Computes the Cauchy bound on the real roots.
// Experiments with more complicated (expensive) bounds did not seem to have a good trade-off.
template <int N> inline real get_bounds(const real *fvec) {
    real max = 0;
    for (int i = 0; i < N; ++i) {
        max = std::max(max, std::abs(fvec[i]));
    }
    return 1.0 + max;
}

// Applies Ridder's bracketing method until we get close to root, followed by newton iterations
template <int N> void ridders_method_newton(const real *fvec, real a, real b, real *roots, int &n_roots, real tol) {
    real fa = polyval<N>(fvec, a);
    real fb = polyval<N>(fvec, b);

    if (!((fa < 0) ^ (fb < 0)))
        return;

    const real tol_newton = 1e-3;

    for (int iter = 0; iter < 30; ++iter) {
        if (std::abs(a - b) < tol_newton) {
            break;
        }
        const real c = (a + b) * 0.5;
        const real fc = polyval<N>(fvec, c);
        const real s = std::sqrt(fc * fc - fa * fb);
        if (!s)
            break;
        const real d = (fa < fb) ? c + (a - c) * fc / s : c + (c - a) * fc / s;
        const real fd = polyval<N>(fvec, d);

        if (fd >= 0 ? (fc < 0) : (fc > 0)) {
            a = c;
            fa = fc;
            b = d;
            fb = fd;
        } else if (fd >= 0 ? (fa < 0) : (fa > 0)) {
            b = d;
            fb = fd;
        } else {
            a = d;
            fa = fd;
        }
    }

    // We switch to Newton's method once we are close to the root
    real x = (a + b) * 0.5;

    real fx, fpx, dx;
    const real *fpvec = fvec + N + 1;
    for (int iter = 0; iter < 10; ++iter) {
        fx = polyval<N>(fvec, x);
        if (std::abs(fx) < tol) {
            break;
        }
        fpx = static_cast<real>(N) * polyval<N - 1>(fpvec, x);
        dx = fx / fpx;
        x = x - dx;
        if (std::abs(dx) < tol) {
            break;
        }
    }

    roots[n_roots++] = x;
}

template <int N>
void isolate_roots(const real *fvec, const real *svec, real a, real b, int sa, int sb, real *roots, int &n_roots,
                   real tol, int depth) {
    if (depth > 300)
        return;

    int n_rts = sa - sb;

    if (n_rts > 1) {
        real c = (a + b) * 0.5;
        int sc = signchanges<N>(svec, c);
        isolate_roots<N>(fvec, svec, a, c, sa, sc, roots, n_roots, tol, depth + 1);
        isolate_roots<N>(fvec, svec, c, b, sc, sb, roots, n_roots, tol, depth + 1);
    } else if (n_rts == 1) {
        ridders_method_newton<N>(fvec, a, b, roots, n_roots, tol);
    }
}

template <int N> inline int bisect_sturm(const real *coeffs, real *roots, real tol = 1e-10) {
    if (coeffs[N] == 0.0)
        return 0; // return bisect_sturm<N-1>(coeffs,roots,tol); // This explodes compile times...

    real fvec[2 * N + 1];
    real svec[3 * N];

    // fvec is the polynomial and its first derivative.
    std::copy(coeffs, coeffs + N + 1, fvec);

    // Normalize w.r.t. leading coeff
    real c_inv = 1.0 / fvec[N];
    for (int i = 0; i < N; ++i)
        fvec[i] *= c_inv;
    fvec[N] = 1.0;

    // Compute the derivative with normalized coefficients
    for (int i = 0; i < N - 1; ++i) {
        fvec[N + 1 + i] = fvec[i + 1] * ((i + 1) / static_cast<real>(N));
    }
    fvec[2 * N] = 1.0;

    // Compute sturm sequences
    build_sturm_seq<N>(fvec, svec);

    // All real roots are in the interval [-r0, r0]
    real r0 = get_bounds<N>(fvec);
    real a = -r0;
    real b = r0;

    int sa = signchanges<N>(svec, a);
    int sb = signchanges<N>(svec, b);

    int n_roots = sa - sb;
    if (n_roots == 0)
        return 0;

    n_roots = 0;
    isolate_roots<N>(fvec, svec, a, b, sa, sb, roots, n_roots, tol, 0);

    return n_roots;
}

template <> inline int bisect_sturm<1>(const real *coeffs, real *roots, real tol) {
    if (coeffs[1] == 0.0) {
        return 0;
    } else {
        roots[0] = -coeffs[0] / coeffs[1];
        return 1;
    }
}

template <> inline int bisect_sturm<0>(const real *coeffs, real *roots, real tol) { return 0; }

template <typename Derived> void charpoly_danilevsky_piv(Eigen::MatrixBase<Derived> &A, real *p) {
    int n = A.rows();

    for (int i = n - 1; i > 0; i--) {

        int piv_ind = i - 1;
        real piv = std::abs(A(i, i - 1));

        // Find largest pivot
        for (int j = 0; j < i - 1; j++) {
            if (std::abs(A(i, j)) > piv) {
                piv = std::abs(A(i, j));
                piv_ind = j;
            }
        }
        if (piv_ind != i - 1) {
            // Perform permutation
            A.row(i - 1).swap(A.row(piv_ind));
            A.col(i - 1).swap(A.col(piv_ind));
        }
        piv = A(i, i - 1);

        VectorX v = A.row(i);
        A.row(i - 1) = v.transpose() * A;

        VectorX vinv = (-1.0) * v;
        vinv(i - 1) = 1;
        vinv /= piv;
        vinv(i - 1) -= 1;
        VectorX Acol = A.col(i - 1);
        for (int j = 0; j <= i; j++)
            A.row(j) = A.row(j) + Acol(j) * vinv.transpose();

        A.row(i).setZero();
        A(i, i - 1) = 1;
    }
    p[n] = 1;
    for (int i = 0; i < n; i++)
        p[i] = -A(0, n - i - 1);
}
} // namespace sturm
} // namespace poselib

#endif