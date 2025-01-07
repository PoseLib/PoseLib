// Copyright (c) 2025, Jin Seo
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

#ifndef POSELIB_REAL_TYPES_H_
#define POSELIB_REAL_TYPES_H_

#include <Eigen/Core>

namespace poselib {

#if !defined(POSELIB_FLOAT) || !(POSELIB_FLOAT)
typedef double real_t;
#else
typedef float real_t;
#endif

} // namespace poselib

namespace Eigen {

template <typename PRECISION, int row> struct Vector_t
{
	typedef Matrix<PRECISION, row, 1> Type;
};

typedef Vector_t<poselib::real_t, 2>::Type Vector2_t;
typedef Vector_t<poselib::real_t, 3>::Type Vector3_t;
typedef Vector_t<poselib::real_t, 4>::Type Vector4_t;
typedef Vector_t<poselib::real_t, Eigen::Dynamic>::Type VectorX_t;

template <typename PRECISION, int dim> struct Matrix_t
{
	typedef Matrix<PRECISION, dim, dim> Type;
};

typedef Matrix_t<poselib::real_t, 2>::Type Matrix2_t;
typedef Matrix_t<poselib::real_t, 3>::Type Matrix3_t;
typedef Matrix_t<poselib::real_t, 4>::Type Matrix4_t;
typedef Matrix_t<poselib::real_t, Eigen::Dynamic>::Type MatrixX_t;

typedef Quaternion<poselib::real_t> Quaternion_t;

} // namespace Eigen

#endif