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
#ifndef POSELIB_MISC_RE3Q3_H_
#define POSELIB_MISC_RE3Q3_H_

#include <Eigen/Dense>

namespace poselib {
namespace re3q3 {

/*
 * Re-implementation of E3Q3. Adapted from Jan Heller's original implementation.
 * Added tricks for improving stability based on choosing the elimination variable.
 *  see Zhou et al., A Stable Algebraic Camera Pose Estimation for Minimal Configurations of 2D/3D Point and Line
 * Correspondences, ACCV 2018 Additionally we do a random affine change of variables to handle further degeneracies.
 *
 * Order of coefficients is:  x^2, xy, xz, y^2, yz, z^2, x, y, z, 1.0; *
 */
int re3q3(const Eigen::Matrix<double, 3, 10> &coeffs, Eigen::Matrix<double, 3, 8> *solutions,
          bool try_random_var_change = true);

// Helper functions for setting up 3Q3 problems

/* Homogeneous linear constraints on rotation matrix
        Rcoeffs*R(:) = 0
    converted into 3q3 problem. */
void rotation_to_3q3(const Eigen::Matrix<double, 3, 9> &Rcoeffs, Eigen::Matrix<double, 3, 10> *coeffs);

/* Inhomogeneous linear constraints on rotation matrix
        Rcoeffs*[R(:);1] = 0
    converted into 3q3 problem. */
void rotation_to_3q3(const Eigen::Matrix<double, 3, 10> &Rcoeffs, Eigen::Matrix<double, 3, 10> *coeffs);

void cayley_param(const Eigen::Matrix<double, 3, 1> &c, Eigen::Matrix<double, 3, 3> *R);

/*
    Helper functions which performs a random rotation to avoid the degeneracy with cayley transform.
    The solutions matrix is 4x8 and contains quaternions. To get back rotation matrices you can use
        Eigen::Quaterniond(solutions.col(i)).toRotationMatrix();
*/
int re3q3_rotation(const Eigen::Matrix<double, 3, 9> &Rcoeffs, Eigen::Matrix<double, 4, 8> *solutions,
                   bool try_random_var_change = true);
int re3q3_rotation(const Eigen::Matrix<double, 3, 10> &Rcoeffs, Eigen::Matrix<double, 4, 8> *solutions,
                   bool try_random_var_change = true);

} // namespace re3q3
} // namespace poselib

#endif