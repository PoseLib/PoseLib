// Copyright (c) 2021, Viktor Larsson
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
// ARE DISCLAIMED. IN NO EVENT SHALL COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef POSELIB_HOMOGRAPHY_H_
#define POSELIB_HOMOGRAPHY_H_

#include "../../types.h"
#include "refiner_base.h"
#include "optim_utils.h"

namespace poselib {

// Non-linear refinement of transfer error |x2 - pi(H*x1)|^2, parameterized by fixing H(2,2) = 1
// I did some preliminary experiments comparing different error functions (e.g. symmetric and transfer)
// as well as other parameterizations (different affine patches, SVD as in Bartoli/Sturm, etc)
// but it does not seem to have a big impact (and is sometimes even worse)
// Implementations of these can be found at https://github.com/vlarsson/homopt
template<typename Accumulator, typename ResidualWeightVector = UniformWeightVector>
class PinholeHomographyRefiner : public RefinerBase<Accumulator, Eigen::Matrix3d>  {
public:
    PinholeHomographyRefiner(const std::vector<Point2D> &points2D_1, const std::vector<Point2D> &points2D_2, const ResidualWeightVector &w = ResidualWeightVector())
        : x1(points2D_1), x2(points2D_2), weights(w) {}

    double compute_residual(Accumulator &acc, const Eigen::Matrix3d &H) {
        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        for (size_t k = 0; k < x1.size(); ++k) {
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double r0 = Hx1_0 * inv_Hx1_2 - x2_0;
            const double r1 = Hx1_1 * inv_Hx1_2 - x2_1;
            acc.add_residual(Eigen::Vector2d(r0,r1), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const Eigen::Matrix3d &H) {
        Eigen::Matrix<double, 2, 8> dH;
        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        for (size_t k = 0; k < x1.size(); ++k) {
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double z0 = Hx1_0 * inv_Hx1_2;
            const double z1 = Hx1_1 * inv_Hx1_2;

            const double r0 = z0 - x2_0;
            const double r1 = z1 - x2_1;

            dH << x1_0, 0.0, -x1_0 * z0, x1_1, 0.0, -x1_1 * z0, 1.0, 0.0, // -z0,
                0.0, x1_0, -x1_0 * z1, 0.0, x1_1, -x1_1 * z1, 0.0, 1.0;   // -z1,
            dH = dH * inv_Hx1_2;

            acc.add_jacobian(Eigen::Vector2d(r0,r1), dH, weights[k]);
        }
    }

    Eigen::Matrix3d step(const Eigen::VectorXd &dp, const Eigen::Matrix3d &H) const {
        Eigen::Matrix3d H_new = H;
        Eigen::Map<Eigen::Matrix<double, 8, 1>>(H_new.data()) += dp;
        return H_new;
    }

    typedef Eigen::Matrix3d param_t;
    static constexpr size_t num_params = 8;
    const std::vector<Point2D> &x1;
    const std::vector<Point2D> &x2;    
    const ResidualWeightVector &weights;
};

template<typename Accumulator, typename ResidualWeightVector = UniformWeightVector>
class PinholeLineHomographyRefiner : public RefinerBase<Accumulator, Eigen::Matrix3d>  {
public:
    PinholeLineHomographyRefiner(const std::vector<Line2D> &lines2D_1, const std::vector<Line2D> &lines2D_2, const ResidualWeightVector &w = ResidualWeightVector())
        : lines1(lines2D_1), weights(w) {

        // Precompute the homogeneous representation for the lines in the second image
        lines2_hom.reserve(lines2D_2.size());
        for(const Line2D &l : lines2D_2) {
            Eigen::Vector3d l_hom = l.x1.homogeneous().cross(l.x2.homogeneous());
            l_hom = l_hom / l_hom.topRows<2>().norm();
            lines2_hom.push_back(l_hom);
        }
    }

    double compute_residual(Accumulator &acc, const Eigen::Matrix3d &H) {

        for (size_t k = 0; k < lines1.size(); ++k) {
            Eigen::Vector2d x1 = (H * lines1[k].x1.homogeneous()).hnormalized();
            Eigen::Vector2d x2 = (H * lines1[k].x2.homogeneous()).hnormalized();
            
            const double r1 = lines2_hom[k].dot(x1.homogeneous());
            const double r2 = lines2_hom[k].dot(x2.homogeneous());

            acc.add_residual(Eigen::Vector2d(r1,r2), weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const Eigen::Matrix3d &H) {
        Eigen::Matrix<double, 2, 8> dH;
        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        for (size_t k = 0; k < lines1.size(); ++k) {
            const double l1 = lines2_hom[k](0);
            const double l2 = lines2_hom[k](1);
            const double l3 = lines2_hom[k](2);
            
            const double x1_0 = lines1[k].x1(0);
            const double x1_1 = lines1[k].x1(1);
            const double x2_0 = lines1[k].x2(0);
            const double x2_1 = lines1[k].x2(1);
            
            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);
            const double z1_0 = Hx1_0 * inv_Hx1_2;
            const double z1_1 = Hx1_1 * inv_Hx1_2;
            
            const double Hx2_0 = H0_0 * x2_0 + H0_1 * x2_1 + H0_2;
            const double Hx2_1 = H1_0 * x2_0 + H1_1 * x2_1 + H1_2;
            const double inv_Hx2_2 = 1.0 / (H2_0 * x2_0 + H2_1 * x2_1 + H2_2);
            const double z2_0 = Hx2_0 * inv_Hx2_2;
            const double z2_1 = Hx2_1 * inv_Hx2_2;
            
            dH << l1*x1_0, l2*x1_0, -x1_0*(l1*z1_0 + l2*z1_1), l1*x1_1, l2*x1_1, -x1_1*(l1*z1_0 + l2*z1_1), l1, l2,
                  l1*x2_0, l2*x2_0, -x2_0*(l1*z2_0 + l2*z2_1), l1*x2_1, l2*x2_1, -x2_1*(l1*z2_0 + l2*z2_1), l1, l2;
            dH.row(0) *= inv_Hx1_2;
            dH.row(1) *= inv_Hx2_2;

            const double r1 = l1*z1_0 + l2*z1_1 + l3;
            const double r2 = l1*z2_0 + l2*z2_1 + l3;
            acc.add_jacobian(Eigen::Vector2d(r1,r2), dH, weights[k]);
        }
    }

    Eigen::Matrix3d step(const Eigen::VectorXd &dp, const Eigen::Matrix3d &H) const {
        Eigen::Matrix3d H_new = H;
        Eigen::Map<Eigen::Matrix<double, 8, 1>>(H_new.data()) += dp;
        return H_new;
    }

    typedef Eigen::Matrix3d param_t;
    static constexpr size_t num_params = 8;
    const std::vector<Line2D> &lines1;
    const ResidualWeightVector &weights;
    std::vector<Eigen::Vector3d> lines2_hom;
};

/*
// Homography refinement using camera model
// Note that this requires undistorted (camera.unproject) points in the first image
// and the camera for the second image
// Error is the transfer error 
//     | x2 - camera.project(H * x1_calib) |
template<typename Accumulator, typename ResidualWeightVector = UniformWeightVector>
class HomographyRefiner {
public:
    HomographyRefiner(const std::vector<Eigen::Vector3d> &points2D_1_calib,
                      const std::vector<Point2D> &points2D_2,
                      //const Camera &camera1,
                      const Camera &camera2,
                      const ResidualWeightVector &w = ResidualWeightVector())
        : x1_calib(points2D_1_calib), x2(points2D_2), cam2(camera2), weights(w) {
    }

    double compute_residual(Accumulator &acc, const Eigen::Matrix3d &H) {
        for (size_t k = 0; k < x1_calib.size(); ++k) {
            Eigen::Vector3d Z = H * x1_calib[k];
            Eigen::Vector2d z;
            cam2.project(Z, &z);
            acc.add_residual(x2[k] - z, weights[k]);
        }
        return acc.get_residual();
    }

    void compute_jacobian(Accumulator &acc, const Eigen::Matrix3d &H) {
        Eigen::Matrix<double, 2, 8> dH;
        const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
        const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
        const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

        for (size_t k = 0; k < x1.size(); ++k) {
            const double x1_0 = x1[k](0), x1_1 = x1[k](1);
            const double x2_0 = x2[k](0), x2_1 = x2[k](1);

            const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
            const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
            const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

            const double z0 = Hx1_0 * inv_Hx1_2;
            const double z1 = Hx1_1 * inv_Hx1_2;

            const double r0 = z0 - x2_0;
            const double r1 = z1 - x2_1;

            dH << x1_0, 0.0, -x1_0 * z0, x1_1, 0.0, -x1_1 * z0, 1.0, 0.0, // -z0,
                0.0, x1_0, -x1_0 * z1, 0.0, x1_1, -x1_1 * z1, 0.0, 1.0;   // -z1,
            dH = dH * inv_Hx1_2;

            // TODO NYI

            acc.add_jacobian(Eigen::Vector2d(r0,r1), dH, weights[k]);
        }
    }

    Eigen::Matrix3d step(const Eigen::VectorXd &dp, const Eigen::Matrix3d &H) const {
        Eigen::Matrix3d H_new = H;
        Eigen::Map<Eigen::Matrix<double, 8, 1>>(H_new.data()) += dp;
        return H_new;
    }

    typedef Eigen::Matrix3d param_t;
    static constexpr size_t num_params = 8;
    const std::vector<Eigen::Vector3d> &x1_calib;
    const std::vector<Point2D> &x2;
    const Camera &cam2;
    const ResidualWeightVector &weights;
};
*/



}

#endif