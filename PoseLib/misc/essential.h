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

#pragma once

#include <Eigen/Dense>
#include <vector>
#include "../types.h"

namespace pose_lib {

    // Computes the essential matrix from the camera motion
    void essential_from_motion(const CameraPose& pose, Eigen::Matrix3d* E);


    /**
    * @brief Given an essential matrix computes the 2 rotations and the 2 translations
    * that can generate four possible motions.
    * @param E Essential matrix
    * @param[out] relative_poses The 4 possible relative poses
    * @ref Multiple View Geometry - Richard Hartley, Andrew Zisserman - second edition
    * @see HZ 9.7 page 259 (Result 9.19)
    */
    void motion_from_essential(const Eigen::Matrix3d& E, pose_lib::CameraPoseVector* relative_poses);

    /*
    Factorizes the essential matrix into the relative poses without using SVD. This approach is faster
    but might degenerate for some particular motions (TODO figure this out).
    */
    void motion_from_essential_fast(const Eigen::Matrix3d& E, pose_lib::CameraPoseVector* relative_poses);

    /* 
    Factorizes the essential matrix into the relative poses. Assumes that the essential matrix corresponds to 
    planar motion, i.e. that we have      
          E = [0   e01  0;
               e10  0  e12;
               0   e21  0]

    Only returns the solution where the rotation is on the form
         R = [a 0 -b; 
             0  1  0;
             b  0  a];
    Note that there is another solution where the rotation is on the form
         R = [a 0   b;
             0  -1  0;
             b  0  -a];
    which is not returned!
    */
    void motion_from_essential_planar(double e01, double e21, double e10, double e12, pose_lib::CameraPoseVector* relative_poses);

}