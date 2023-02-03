// Copyright (c) 2020 Marcus Valtonen Örnhag
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "homography_kukelova_cvpr_2015.h"
#include <Eigen/Geometry>
#include <vector>
#include "PoseLib/misc/gj.h"

namespace poselib {
inline Eigen::Matrix3d construct_homography_from_sols(
    const Eigen::VectorXd& xx,
    const Eigen::VectorXd& tmp,
    const Eigen::MatrixXd& N);

inline Eigen::MatrixXcd solver_kukelova_cvpr_2015(const Eigen::VectorXd& data);

int homography_kukelova_cvpr_2015(
    const std::vector<Eigen::Vector3d> &p1,
    const std::vector<Eigen::Vector3d> &p2,
    std::vector<Eigen::Matrix3d> *H,
    std::vector<double> *distortion_parameter1,
    std::vector<double> *distortion_parameter2
) {
    // This is a five point method
    const int nbr_pts = 5;

    // TODO: Fix this
    // Make homogenous
    Eigen::MatrixXd u1(2, nbr_pts);
    Eigen::MatrixXd u2(2, nbr_pts);
    for (int i = 0; i < nbr_pts; ++i) {
        u1.col(i) = p1[i].hnormalized();
        u2.col(i) = p2[i].hnormalized();
    }

    // We expect inhomogenous input data, i.e. p1 and p2 are 2x5 matrices
    //assert(p1.rows() == 2);
    //assert(p2.rows() == 2);
    //assert(p1.cols() == nbr_pts);
    //assert(p2.cols() == nbr_pts);

    // Compute normalization matrix
    //double scale1 = HomLib::normalize2dpts(p1);
    //double scale2 = HomLib::normalize2dpts(p2);
    //double scale = std::max(scale1, scale2);
    //Eigen::Vector3d s;
    //s << scale, scale, 1.0;
    //Eigen::DiagonalMatrix<double, 3> S = s.asDiagonal();

    //// Normalize data
    //Eigen::MatrixXd x1(3, nbr_pts);
    //Eigen::MatrixXd x2(3, nbr_pts);
    //x1 = p1.colwise().homogeneous();
    //x2 = p2.colwise().homogeneous();

    //x1 = S * x1;
    //x2 = S * x2;

    //Eigen::MatrixXd u1(2, nbr_pts);
    //u1 << x1.colwise().hnormalized();
    //Eigen::MatrixXd u2(2, nbr_pts);
    //u2 << x2.colwise().hnormalized();

    // Compute distance to center for first points
    Eigen::VectorXd r21 = u1.colwise().squaredNorm();

    // Setup matrix for null space computation
    Eigen::MatrixXd M1(nbr_pts, 8);
    M1.setZero();

    for (int k = 0; k < nbr_pts; k++) {
        M1.row(k) << -r21(k) * u2(1, k), r21(k) * u2(0, k), -u2(1, k) * u1.col(k).homogeneous().transpose(),
                     u2(0, k) * u1.col(k).homogeneous().transpose();
    }

    // Find the null space
    // TODO(marcusvaltonen): This might be expensive - find out which is faster!
    // FullPivLU<MatrixXd> lu(M1);
    // MatrixXd N = lu.kernel();

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M1, Eigen::ComputeFullV);
    Eigen::MatrixXd N = svd.matrixV().rightCols(3);

    // Create temporary input vector
    Eigen::MatrixXd tmp(4, nbr_pts);
    tmp << u1, u2;
    Eigen::VectorXd d(44);
    d << Eigen::Map<Eigen::VectorXd>(N.data(), 8*3),
         Eigen::Map<Eigen::VectorXd>(tmp.data(), 4*nbr_pts);

    // Create matrix M
    Eigen::MatrixXd M(7, 13);

    M << -d[24]*d[26], -d[25]*d[26], -std::pow(d[24],2)*d[26] - std::pow(d[25],2)*d[26], -d[26], d[0]*std::pow(d[24],2)*std::pow(d[26],2) + d[0]*std::pow(d[24],2)*std::pow(d[27],2) + d[0]*std::pow(d[25],2)*std::pow(d[26],2) + d[0]*std::pow(d[25],2)*std::pow(d[27],2) + d[2]*d[24]*std::pow(d[26],2) + d[2]*d[24]*std::pow(d[27],2) + d[3]*d[25]*std::pow(d[26],2) + d[3]*d[25]*std::pow(d[27],2) + d[4]*std::pow(d[26],2) + d[4]*std::pow(d[27],2), 0, d[0]*std::pow(d[24],2) + d[0]*std::pow(d[25],2) + d[2]*d[24] + d[3]*d[25] + d[4], d[8]*std::pow(d[24],2)*std::pow(d[26],2) + d[8]*std::pow(d[24],2)*std::pow(d[27],2) + d[8]*std::pow(d[25],2)*std::pow(d[26],2) + d[8]*std::pow(d[25],2)*std::pow(d[27],2) + d[10]*d[24]*std::pow(d[26],2) + d[10]*d[24]*std::pow(d[27],2) + d[11]*d[25]*std::pow(d[26],2) + d[11]*d[25]*std::pow(d[27],2) + d[12]*std::pow(d[26],2) + d[12]*std::pow(d[27],2), 0, d[16]*std::pow(d[24],2)*std::pow(d[26],2) + d[16]*std::pow(d[24],2)*std::pow(d[27],2) + d[16]*std::pow(d[25],2)*std::pow(d[26],2) + d[16]*std::pow(d[25],2)*std::pow(d[27],2) + d[18]*d[24]*std::pow(d[26],2) + d[18]*d[24]*std::pow(d[27],2) + d[19]*d[25]*std::pow(d[26],2) + d[19]*d[25]*std::pow(d[27],2) + d[20]*std::pow(d[26],2) + d[20]*std::pow(d[27],2), 0, d[8]*std::pow(d[24],2) + d[8]*std::pow(d[25],2) + d[10]*d[24] + d[11]*d[25] + d[12], d[16]*std::pow(d[24],2) + d[16]*std::pow(d[25],2) + d[18]*d[24] + d[19]*d[25] + d[20],  // NOLINT
    -d[28]*d[30], -d[29]*d[30], -std::pow(d[28],2)*d[30] - std::pow(d[29],2)*d[30], -d[30], d[0]*std::pow(d[28],2)*std::pow(d[30],2) + d[0]*std::pow(d[28],2)*std::pow(d[31],2) + d[0]*std::pow(d[29],2)*std::pow(d[30],2) + d[0]*std::pow(d[29],2)*std::pow(d[31],2) + d[2]*d[28]*std::pow(d[30],2) + d[2]*d[28]*std::pow(d[31],2) + d[3]*d[29]*std::pow(d[30],2) + d[3]*d[29]*std::pow(d[31],2) + d[4]*std::pow(d[30],2) + d[4]*std::pow(d[31],2), 0, d[0]*std::pow(d[28],2) + d[0]*std::pow(d[29],2) + d[2]*d[28] + d[3]*d[29] + d[4], d[8]*std::pow(d[28],2)*std::pow(d[30],2) + d[8]*std::pow(d[28],2)*std::pow(d[31],2) + d[8]*std::pow(d[29],2)*std::pow(d[30],2) + d[8]*std::pow(d[29],2)*std::pow(d[31],2) + d[10]*d[28]*std::pow(d[30],2) + d[10]*d[28]*std::pow(d[31],2) + d[11]*d[29]*std::pow(d[30],2) + d[11]*d[29]*std::pow(d[31],2) + d[12]*std::pow(d[30],2) + d[12]*std::pow(d[31],2), 0, d[16]*std::pow(d[28],2)*std::pow(d[30],2) + d[16]*std::pow(d[28],2)*std::pow(d[31],2) + d[16]*std::pow(d[29],2)*std::pow(d[30],2) + d[16]*std::pow(d[29],2)*std::pow(d[31],2) + d[18]*d[28]*std::pow(d[30],2) + d[18]*d[28]*std::pow(d[31],2) + d[19]*d[29]*std::pow(d[30],2) + d[19]*d[29]*std::pow(d[31],2) + d[20]*std::pow(d[30],2) + d[20]*std::pow(d[31],2), 0, d[8]*std::pow(d[28],2) + d[8]*std::pow(d[29],2) + d[10]*d[28] + d[11]*d[29] + d[12], d[16]*std::pow(d[28],2) + d[16]*std::pow(d[29],2) + d[18]*d[28] + d[19]*d[29] + d[20],  // NOLINT
    -d[32]*d[34], -d[33]*d[34], -std::pow(d[32],2)*d[34] - std::pow(d[33],2)*d[34], -d[34], d[0]*std::pow(d[32],2)*std::pow(d[34],2) + d[0]*std::pow(d[32],2)*std::pow(d[35],2) + d[0]*std::pow(d[33],2)*std::pow(d[34],2) + d[0]*std::pow(d[33],2)*std::pow(d[35],2) + d[2]*d[32]*std::pow(d[34],2) + d[2]*d[32]*std::pow(d[35],2) + d[3]*d[33]*std::pow(d[34],2) + d[3]*d[33]*std::pow(d[35],2) + d[4]*std::pow(d[34],2) + d[4]*std::pow(d[35],2), 0, d[0]*std::pow(d[32],2) + d[0]*std::pow(d[33],2) + d[2]*d[32] + d[3]*d[33] + d[4], d[8]*std::pow(d[32],2)*std::pow(d[34],2) + d[8]*std::pow(d[32],2)*std::pow(d[35],2) + d[8]*std::pow(d[33],2)*std::pow(d[34],2) + d[8]*std::pow(d[33],2)*std::pow(d[35],2) + d[10]*d[32]*std::pow(d[34],2) + d[10]*d[32]*std::pow(d[35],2) + d[11]*d[33]*std::pow(d[34],2) + d[11]*d[33]*std::pow(d[35],2) + d[12]*std::pow(d[34],2) + d[12]*std::pow(d[35],2), 0, d[16]*std::pow(d[32],2)*std::pow(d[34],2) + d[16]*std::pow(d[32],2)*std::pow(d[35],2) + d[16]*std::pow(d[33],2)*std::pow(d[34],2) + d[16]*std::pow(d[33],2)*std::pow(d[35],2) + d[18]*d[32]*std::pow(d[34],2) + d[18]*d[32]*std::pow(d[35],2) + d[19]*d[33]*std::pow(d[34],2) + d[19]*d[33]*std::pow(d[35],2) + d[20]*std::pow(d[34],2) + d[20]*std::pow(d[35],2), 0, d[8]*std::pow(d[32],2) + d[8]*std::pow(d[33],2) + d[10]*d[32] + d[11]*d[33] + d[12], d[16]*std::pow(d[32],2) + d[16]*std::pow(d[33],2) + d[18]*d[32] + d[19]*d[33] + d[20],  // NOLINT
    -d[36]*d[38], -d[37]*d[38], -std::pow(d[36],2)*d[38] - std::pow(d[37],2)*d[38], -d[38], d[0]*std::pow(d[36],2)*std::pow(d[38],2) + d[0]*std::pow(d[36],2)*std::pow(d[39],2) + d[0]*std::pow(d[37],2)*std::pow(d[38],2) + d[0]*std::pow(d[37],2)*std::pow(d[39],2) + d[2]*d[36]*std::pow(d[38],2) + d[2]*d[36]*std::pow(d[39],2) + d[3]*d[37]*std::pow(d[38],2) + d[3]*d[37]*std::pow(d[39],2) + d[4]*std::pow(d[38],2) + d[4]*std::pow(d[39],2), 0, d[0]*std::pow(d[36],2) + d[0]*std::pow(d[37],2) + d[2]*d[36] + d[3]*d[37] + d[4], d[8]*std::pow(d[36],2)*std::pow(d[38],2) + d[8]*std::pow(d[36],2)*std::pow(d[39],2) + d[8]*std::pow(d[37],2)*std::pow(d[38],2) + d[8]*std::pow(d[37],2)*std::pow(d[39],2) + d[10]*d[36]*std::pow(d[38],2) + d[10]*d[36]*std::pow(d[39],2) + d[11]*d[37]*std::pow(d[38],2) + d[11]*d[37]*std::pow(d[39],2) + d[12]*std::pow(d[38],2) + d[12]*std::pow(d[39],2), 0, d[16]*std::pow(d[36],2)*std::pow(d[38],2) + d[16]*std::pow(d[36],2)*std::pow(d[39],2) + d[16]*std::pow(d[37],2)*std::pow(d[38],2) + d[16]*std::pow(d[37],2)*std::pow(d[39],2) + d[18]*d[36]*std::pow(d[38],2) + d[18]*d[36]*std::pow(d[39],2) + d[19]*d[37]*std::pow(d[38],2) + d[19]*d[37]*std::pow(d[39],2) + d[20]*std::pow(d[38],2) + d[20]*std::pow(d[39],2), 0, d[8]*std::pow(d[36],2) + d[8]*std::pow(d[37],2) + d[10]*d[36] + d[11]*d[37] + d[12], d[16]*std::pow(d[36],2) + d[16]*std::pow(d[37],2) + d[18]*d[36] + d[19]*d[37] + d[20],  // NOLINT
    -d[40]*d[42], -d[41]*d[42], -std::pow(d[40],2)*d[42] - std::pow(d[41],2)*d[42], -d[42], d[0]*std::pow(d[40],2)*std::pow(d[42],2) + d[0]*std::pow(d[40],2)*std::pow(d[43],2) + d[0]*std::pow(d[41],2)*std::pow(d[42],2) + d[0]*std::pow(d[41],2)*std::pow(d[43],2) + d[2]*d[40]*std::pow(d[42],2) + d[2]*d[40]*std::pow(d[43],2) + d[3]*d[41]*std::pow(d[42],2) + d[3]*d[41]*std::pow(d[43],2) + d[4]*std::pow(d[42],2) + d[4]*std::pow(d[43],2), 0, d[0]*std::pow(d[40],2) + d[0]*std::pow(d[41],2) + d[2]*d[40] + d[3]*d[41] + d[4], d[8]*std::pow(d[40],2)*std::pow(d[42],2) + d[8]*std::pow(d[40],2)*std::pow(d[43],2) + d[8]*std::pow(d[41],2)*std::pow(d[42],2) + d[8]*std::pow(d[41],2)*std::pow(d[43],2) + d[10]*d[40]*std::pow(d[42],2) + d[10]*d[40]*std::pow(d[43],2) + d[11]*d[41]*std::pow(d[42],2) + d[11]*d[41]*std::pow(d[43],2) + d[12]*std::pow(d[42],2) + d[12]*std::pow(d[43],2), 0, d[16]*std::pow(d[40],2)*std::pow(d[42],2) + d[16]*std::pow(d[40],2)*std::pow(d[43],2) + d[16]*std::pow(d[41],2)*std::pow(d[42],2) + d[16]*std::pow(d[41],2)*std::pow(d[43],2) + d[18]*d[40]*std::pow(d[42],2) + d[18]*d[40]*std::pow(d[43],2) + d[19]*d[41]*std::pow(d[42],2) + d[19]*d[41]*std::pow(d[43],2) + d[20]*std::pow(d[42],2) + d[20]*std::pow(d[43],2), 0, d[8]*std::pow(d[40],2) + d[8]*std::pow(d[41],2) + d[10]*d[40] + d[11]*d[41] + d[12], d[16]*std::pow(d[40],2) + d[16]*std::pow(d[41],2) + d[18]*d[40] + d[19]*d[41] + d[20],  // NOLINT
    0, 0, 0, 0, 0, -d[4], d[0], 0, -d[12], 0, -d[20], d[8], d[16],
    0, 0, 0, 0, 0, -d[7], d[1], 0, -d[15], 0, -d[23], d[9], d[17];

    // Gauss-Jordan
    poselib::gj(&M);

    // Wrap input data to expected format
    Eigen::VectorXd input(66);
    input << Eigen::Map<Eigen::VectorXd>(N.data(), 8*3),
             Eigen::Map<Eigen::VectorXd>(M.rightCols(6).data(), 6*7);

    // Extract solution
    Eigen::MatrixXcd sols = solver_kukelova_cvpr_2015(input);

    // Pre-processing: Remove complex-valued solutions
    double thresh = 1e-5;
    Eigen::ArrayXd real_sols(3);
    real_sols = sols.imag().cwiseAbs().colwise().sum();
    //int nbr_real_sols = (real_sols <= thresh).count();

    // Create putative solutions
    Eigen::Matrix3d Htmp;
    //std::complex<double> lam1;
    //std::complex<double> lam2;
    //std::vector<HomLib::PoseData> posedata(nbr_real_sols);
    Eigen::ArrayXd xx(3);
    H->clear();
    distortion_parameter1->clear();
    distortion_parameter2->clear();

    for (int i = 0; i < real_sols.size(); i++) {
        if (real_sols(i) <= thresh) {
            // Get parameters.
            xx = sols.col(i).real();

            // Construct putative fundamental matrix
            //Htmp = S.inverse() * poselib::construct_homography_from_sols(xx, d, N) * S;
            Htmp = poselib::construct_homography_from_sols(xx, d, N);
            
            // Package output
            H->push_back(Htmp);
            //distortion_parameter1->push_back(xx(0) * std::pow(scale, 2));
            //distortion_parameter2->push_back(xx(1) * std::pow(scale, 2));
            distortion_parameter1->push_back(xx(0));
            distortion_parameter2->push_back(xx(1));
        }
    }

    return H->size();
}

// Function that utilizes the last equation of the DLT system to discard false solutions
inline Eigen::Matrix3d construct_homography_from_sols(
    const Eigen::VectorXd& xx,
    const Eigen::VectorXd& tmp,
    const Eigen::MatrixXd& N
) {
    Eigen::VectorXd d(51);
    d << 0, 0, 0, xx(0), xx(1), 0, xx(2), tmp;

    Eigen::Matrix<double, 5, 5> M;
    M << -d[31]*d[33], -d[32]*d[33], -d[3]*std::pow(d[31],2)*d[33] - d[3]*std::pow(d[32],2)*d[33] - d[33], d[4]*d[7]*std::pow(d[31],2)*std::pow(d[33],2) + d[4]*d[7]*std::pow(d[31],2)*std::pow(d[34],2) + d[4]*d[7]*std::pow(d[32],2)*std::pow(d[33],2) + d[4]*d[7]*std::pow(d[32],2)*std::pow(d[34],2) + d[4]*d[9]*d[31]*std::pow(d[33],2) + d[4]*d[9]*d[31]*std::pow(d[34],2) + d[4]*d[10]*d[32]*std::pow(d[33],2) + d[4]*d[10]*d[32]*std::pow(d[34],2) + d[4]*d[11]*std::pow(d[33],2) + d[4]*d[11]*std::pow(d[34],2) + d[7]*std::pow(d[31],2) + d[7]*std::pow(d[32],2) + d[9]*d[31] + d[10]*d[32] + d[11], d[4]*d[6]*d[15]*std::pow(d[31],2)*std::pow(d[33],2) + d[4]*d[6]*d[15]*std::pow(d[31],2)*std::pow(d[34],2) + d[4]*d[6]*d[15]*std::pow(d[32],2)*std::pow(d[33],2) + d[4]*d[6]*d[15]*std::pow(d[32],2)*std::pow(d[34],2) + d[4]*d[6]*d[17]*d[31]*std::pow(d[33],2) + d[4]*d[6]*d[17]*d[31]*std::pow(d[34],2) + d[4]*d[6]*d[18]*d[32]*std::pow(d[33],2) + d[4]*d[6]*d[18]*d[32]*std::pow(d[34],2) + d[4]*d[6]*d[19]*std::pow(d[33],2) + d[4]*d[6]*d[19]*std::pow(d[34],2) + d[4]*d[23]*std::pow(d[31],2)*std::pow(d[33],2) + d[4]*d[23]*std::pow(d[31],2)*std::pow(d[34],2) + d[4]*d[23]*std::pow(d[32],2)*std::pow(d[33],2) + d[4]*d[23]*std::pow(d[32],2)*std::pow(d[34],2) + d[4]*d[25]*d[31]*std::pow(d[33],2) + d[4]*d[25]*d[31]*std::pow(d[34],2) + d[4]*d[26]*d[32]*std::pow(d[33],2) + d[4]*d[26]*d[32]*std::pow(d[34],2) + d[4]*d[27]*std::pow(d[33],2) + d[4]*d[27]*std::pow(d[34],2) + d[6]*d[15]*std::pow(d[31],2) + d[6]*d[15]*std::pow(d[32],2) + d[6]*d[17]*d[31] + d[6]*d[18]*d[32] + d[6]*d[19] + d[23]*std::pow(d[31],2) + d[23]*std::pow(d[32],2) + d[25]*d[31] + d[26]*d[32] + d[27],  // NOLINT
    -d[35]*d[37], -d[36]*d[37], -d[3]*std::pow(d[35],2)*d[37] - d[3]*std::pow(d[36],2)*d[37] - d[37], d[4]*d[7]*std::pow(d[35],2)*std::pow(d[37],2) + d[4]*d[7]*std::pow(d[35],2)*std::pow(d[38],2) + d[4]*d[7]*std::pow(d[36],2)*std::pow(d[37],2) + d[4]*d[7]*std::pow(d[36],2)*std::pow(d[38],2) + d[4]*d[9]*d[35]*std::pow(d[37],2) + d[4]*d[9]*d[35]*std::pow(d[38],2) + d[4]*d[10]*d[36]*std::pow(d[37],2) + d[4]*d[10]*d[36]*std::pow(d[38],2) + d[4]*d[11]*std::pow(d[37],2) + d[4]*d[11]*std::pow(d[38],2) + d[7]*std::pow(d[35],2) + d[7]*std::pow(d[36],2) + d[9]*d[35] + d[10]*d[36] + d[11], d[4]*d[6]*d[15]*std::pow(d[35],2)*std::pow(d[37],2) + d[4]*d[6]*d[15]*std::pow(d[35],2)*std::pow(d[38],2) + d[4]*d[6]*d[15]*std::pow(d[36],2)*std::pow(d[37],2) + d[4]*d[6]*d[15]*std::pow(d[36],2)*std::pow(d[38],2) + d[4]*d[6]*d[17]*d[35]*std::pow(d[37],2) + d[4]*d[6]*d[17]*d[35]*std::pow(d[38],2) + d[4]*d[6]*d[18]*d[36]*std::pow(d[37],2) + d[4]*d[6]*d[18]*d[36]*std::pow(d[38],2) + d[4]*d[6]*d[19]*std::pow(d[37],2) + d[4]*d[6]*d[19]*std::pow(d[38],2) + d[4]*d[23]*std::pow(d[35],2)*std::pow(d[37],2) + d[4]*d[23]*std::pow(d[35],2)*std::pow(d[38],2) + d[4]*d[23]*std::pow(d[36],2)*std::pow(d[37],2) + d[4]*d[23]*std::pow(d[36],2)*std::pow(d[38],2) + d[4]*d[25]*d[35]*std::pow(d[37],2) + d[4]*d[25]*d[35]*std::pow(d[38],2) + d[4]*d[26]*d[36]*std::pow(d[37],2) + d[4]*d[26]*d[36]*std::pow(d[38],2) + d[4]*d[27]*std::pow(d[37],2) + d[4]*d[27]*std::pow(d[38],2) + d[6]*d[15]*std::pow(d[35],2) + d[6]*d[15]*std::pow(d[36],2) + d[6]*d[17]*d[35] + d[6]*d[18]*d[36] + d[6]*d[19] + d[23]*std::pow(d[35],2) + d[23]*std::pow(d[36],2) + d[25]*d[35] + d[26]*d[36] + d[27],  // NOLINT
    -d[39]*d[41], -d[40]*d[41], -d[3]*std::pow(d[39],2)*d[41] - d[3]*std::pow(d[40],2)*d[41] - d[41], d[4]*d[7]*std::pow(d[39],2)*std::pow(d[41],2) + d[4]*d[7]*std::pow(d[39],2)*std::pow(d[42],2) + d[4]*d[7]*std::pow(d[40],2)*std::pow(d[41],2) + d[4]*d[7]*std::pow(d[40],2)*std::pow(d[42],2) + d[4]*d[9]*d[39]*std::pow(d[41],2) + d[4]*d[9]*d[39]*std::pow(d[42],2) + d[4]*d[10]*d[40]*std::pow(d[41],2) + d[4]*d[10]*d[40]*std::pow(d[42],2) + d[4]*d[11]*std::pow(d[41],2) + d[4]*d[11]*std::pow(d[42],2) + d[7]*std::pow(d[39],2) + d[7]*std::pow(d[40],2) + d[9]*d[39] + d[10]*d[40] + d[11], d[4]*d[6]*d[15]*std::pow(d[39],2)*std::pow(d[41],2) + d[4]*d[6]*d[15]*std::pow(d[39],2)*std::pow(d[42],2) + d[4]*d[6]*d[15]*std::pow(d[40],2)*std::pow(d[41],2) + d[4]*d[6]*d[15]*std::pow(d[40],2)*std::pow(d[42],2) + d[4]*d[6]*d[17]*d[39]*std::pow(d[41],2) + d[4]*d[6]*d[17]*d[39]*std::pow(d[42],2) + d[4]*d[6]*d[18]*d[40]*std::pow(d[41],2) + d[4]*d[6]*d[18]*d[40]*std::pow(d[42],2) + d[4]*d[6]*d[19]*std::pow(d[41],2) + d[4]*d[6]*d[19]*std::pow(d[42],2) + d[4]*d[23]*std::pow(d[39],2)*std::pow(d[41],2) + d[4]*d[23]*std::pow(d[39],2)*std::pow(d[42],2) + d[4]*d[23]*std::pow(d[40],2)*std::pow(d[41],2) + d[4]*d[23]*std::pow(d[40],2)*std::pow(d[42],2) + d[4]*d[25]*d[39]*std::pow(d[41],2) + d[4]*d[25]*d[39]*std::pow(d[42],2) + d[4]*d[26]*d[40]*std::pow(d[41],2) + d[4]*d[26]*d[40]*std::pow(d[42],2) + d[4]*d[27]*std::pow(d[41],2) + d[4]*d[27]*std::pow(d[42],2) + d[6]*d[15]*std::pow(d[39],2) + d[6]*d[15]*std::pow(d[40],2) + d[6]*d[17]*d[39] + d[6]*d[18]*d[40] + d[6]*d[19] + d[23]*std::pow(d[39],2) + d[23]*std::pow(d[40],2) + d[25]*d[39] + d[26]*d[40] + d[27],  // NOLINT
    -d[43]*d[45], -d[44]*d[45], -d[3]*std::pow(d[43],2)*d[45] - d[3]*std::pow(d[44],2)*d[45] - d[45], d[4]*d[7]*std::pow(d[43],2)*std::pow(d[45],2) + d[4]*d[7]*std::pow(d[43],2)*std::pow(d[46],2) + d[4]*d[7]*std::pow(d[44],2)*std::pow(d[45],2) + d[4]*d[7]*std::pow(d[44],2)*std::pow(d[46],2) + d[4]*d[9]*d[43]*std::pow(d[45],2) + d[4]*d[9]*d[43]*std::pow(d[46],2) + d[4]*d[10]*d[44]*std::pow(d[45],2) + d[4]*d[10]*d[44]*std::pow(d[46],2) + d[4]*d[11]*std::pow(d[45],2) + d[4]*d[11]*std::pow(d[46],2) + d[7]*std::pow(d[43],2) + d[7]*std::pow(d[44],2) + d[9]*d[43] + d[10]*d[44] + d[11], d[4]*d[6]*d[15]*std::pow(d[43],2)*std::pow(d[45],2) + d[4]*d[6]*d[15]*std::pow(d[43],2)*std::pow(d[46],2) + d[4]*d[6]*d[15]*std::pow(d[44],2)*std::pow(d[45],2) + d[4]*d[6]*d[15]*std::pow(d[44],2)*std::pow(d[46],2) + d[4]*d[6]*d[17]*d[43]*std::pow(d[45],2) + d[4]*d[6]*d[17]*d[43]*std::pow(d[46],2) + d[4]*d[6]*d[18]*d[44]*std::pow(d[45],2) + d[4]*d[6]*d[18]*d[44]*std::pow(d[46],2) + d[4]*d[6]*d[19]*std::pow(d[45],2) + d[4]*d[6]*d[19]*std::pow(d[46],2) + d[4]*d[23]*std::pow(d[43],2)*std::pow(d[45],2) + d[4]*d[23]*std::pow(d[43],2)*std::pow(d[46],2) + d[4]*d[23]*std::pow(d[44],2)*std::pow(d[45],2) + d[4]*d[23]*std::pow(d[44],2)*std::pow(d[46],2) + d[4]*d[25]*d[43]*std::pow(d[45],2) + d[4]*d[25]*d[43]*std::pow(d[46],2) + d[4]*d[26]*d[44]*std::pow(d[45],2) + d[4]*d[26]*d[44]*std::pow(d[46],2) + d[4]*d[27]*std::pow(d[45],2) + d[4]*d[27]*std::pow(d[46],2) + d[6]*d[15]*std::pow(d[43],2) + d[6]*d[15]*std::pow(d[44],2) + d[6]*d[17]*d[43] + d[6]*d[18]*d[44] + d[6]*d[19] + d[23]*std::pow(d[43],2) + d[23]*std::pow(d[44],2) + d[25]*d[43] + d[26]*d[44] + d[27],  // NOLINT
    -d[47]*d[49], -d[48]*d[49], -d[3]*std::pow(d[47],2)*d[49] - d[3]*std::pow(d[48],2)*d[49] - d[49], d[4]*d[7]*std::pow(d[47],2)*std::pow(d[49],2) + d[4]*d[7]*std::pow(d[47],2)*std::pow(d[50],2) + d[4]*d[7]*std::pow(d[48],2)*std::pow(d[49],2) + d[4]*d[7]*std::pow(d[48],2)*std::pow(d[50],2) + d[4]*d[9]*d[47]*std::pow(d[49],2) + d[4]*d[9]*d[47]*std::pow(d[50],2) + d[4]*d[10]*d[48]*std::pow(d[49],2) + d[4]*d[10]*d[48]*std::pow(d[50],2) + d[4]*d[11]*std::pow(d[49],2) + d[4]*d[11]*std::pow(d[50],2) + d[7]*std::pow(d[47],2) + d[7]*std::pow(d[48],2) + d[9]*d[47] + d[10]*d[48] + d[11], d[4]*d[6]*d[15]*std::pow(d[47],2)*std::pow(d[49],2) + d[4]*d[6]*d[15]*std::pow(d[47],2)*std::pow(d[50],2) + d[4]*d[6]*d[15]*std::pow(d[48],2)*std::pow(d[49],2) + d[4]*d[6]*d[15]*std::pow(d[48],2)*std::pow(d[50],2) + d[4]*d[6]*d[17]*d[47]*std::pow(d[49],2) + d[4]*d[6]*d[17]*d[47]*std::pow(d[50],2) + d[4]*d[6]*d[18]*d[48]*std::pow(d[49],2) + d[4]*d[6]*d[18]*d[48]*std::pow(d[50],2) + d[4]*d[6]*d[19]*std::pow(d[49],2) + d[4]*d[6]*d[19]*std::pow(d[50],2) + d[4]*d[23]*std::pow(d[47],2)*std::pow(d[49],2) + d[4]*d[23]*std::pow(d[47],2)*std::pow(d[50],2) + d[4]*d[23]*std::pow(d[48],2)*std::pow(d[49],2) + d[4]*d[23]*std::pow(d[48],2)*std::pow(d[50],2) + d[4]*d[25]*d[47]*std::pow(d[49],2) + d[4]*d[25]*d[47]*std::pow(d[50],2) + d[4]*d[26]*d[48]*std::pow(d[49],2) + d[4]*d[26]*d[48]*std::pow(d[50],2) + d[4]*d[27]*std::pow(d[49],2) + d[4]*d[27]*std::pow(d[50],2) + d[6]*d[15]*std::pow(d[47],2) + d[6]*d[15]*std::pow(d[48],2) + d[6]*d[17]*d[47] + d[6]*d[18]*d[48] + d[6]*d[19] + d[23]*std::pow(d[47],2) + d[23]*std::pow(d[48],2) + d[25]*d[47] + d[26]*d[48] + d[27];  // NOLINT

    // Perform SVD and extract coeffs
    Eigen::JacobiSVD<Eigen::Matrix<double, 5, 5>> svd(M, Eigen::ComputeFullV);
    Eigen::Matrix<double, 4, 1>  v = svd.matrixV().col(4).hnormalized();

    // Construct homography
    Eigen::Matrix<double, 8, 1> v1 = N.col(0) * v(3) + N.col(1) * xx(2) + N.col(2);
    Eigen::Matrix3d H;
    H.row(0) = v1.segment<3>(2);
    H.row(1) = v1.segment<3>(5);
    H.row(2) = v.head<3>();

    return H;
}


Eigen::MatrixXcd solver_kukelova_cvpr_2015(const Eigen::VectorXd& data) {
    // Compute coefficients
    const double* d = data.data();
    Eigen::VectorXd coeffs(22);
    coeffs[0] = -d[34];
    coeffs[1] = -d[27];
    coeffs[2] = -d[48];
    coeffs[3] = -d[41];
    coeffs[4] = d[33] - d[55];
    coeffs[5] = d[26];
    coeffs[6] = d[47] - d[62];
    coeffs[7] = d[40];
    coeffs[8] = d[54];
    coeffs[9] = d[61];
    coeffs[10] = -d[37];
    coeffs[11] = -d[51];
    coeffs[12] = d[35];
    coeffs[13] = d[28] - d[58];
    coeffs[14] = d[49];
    coeffs[15] = d[42] - d[65];
    coeffs[16] = d[56];
    coeffs[17] = d[63];
    coeffs[18] = d[36] - d[58];
    coeffs[19] = d[50] - d[65];
    coeffs[20] = d[57];
    coeffs[21] = d[64];

    // Setup elimination template
    static const int coeffs0_ind[] = { 10,10,0,10,1,10,10,12,2,0,10,11,13,3,11,1,10,10,18,11,4,12,0,10,18,5,13,1,10,18,15,3,11,11,19,16,6,14,4,12,2,18,0,10,11,19,7,15,5,13,3,11,1,18,20,10,19,5,13,20,7,15,5,20,13,21,17,6,14,19,2,11,7,15,3,19,21,11,9,17,8,16,6,14,20,4,18,19,12,21,8,16,4,12,18,20 };  // NOLINT
    static const int coeffs1_ind[] = { 9,21,17,9,17,21,6,19,14,7,21,15,9,17,8,20,21,16,8,16,20 };  // NOLINT
    static const int C0_ind[] = { 0,12,17,30,33,34,47,48,49,51,55,62,64,65,66,67,68,75,76,79,81,82,85,90,94,97,98,101,102,111,112,115,116,123,124,128,129,130,131,132,133,135,136,137,138,142,145,146,147,148,149,150,152,155,156,157,159,165,166,175,181,182,184,187,189,191,192,195,196,199,200,201,211,212,216,219,220,221,225,226,227,228,229,230,231,232,233,234,237,238,241,242,245,246,250,254 };  // NOLINT
    static const int C1_ind[] = { 8,9,13,19,20,23,24,25,29,40,43,45,53,54,56,57,58,61,69,70,74 };  // NOLINT

    Eigen::Matrix<double, 16, 16> C0; C0.setZero();
    Eigen::Matrix<double, 16, 5> C1; C1.setZero();
    for (int i = 0; i < 96; i++) { C0(C0_ind[i]) = coeffs(coeffs0_ind[i]); }
    for (int i = 0; i < 21; i++) { C1(C1_ind[i]) = coeffs(coeffs1_ind[i]); }

    Eigen::Matrix<double, 16, 5> C12 = C0.partialPivLu().solve(C1);

    // Setup action matrix
    Eigen::Matrix<double, 9, 5> RR;
    RR << -C12.bottomRows(4), Eigen::Matrix<double, 5, 5>::Identity(5, 5);

    static const int AM_ind[] = { 5,0,1,2,3 };  // NOLINT
    Eigen::Matrix<double, 5, 5> AM;
    for (int i = 0; i < 5; i++) {
        AM.row(i) = RR.row(AM_ind[i]);
    }

    Eigen::Matrix<std::complex<double>, 3, 5> sols;
    sols.setZero();

    // Solve eigenvalue problem
    Eigen::EigenSolver<Eigen::Matrix<double, 5, 5> > es(AM);
    Eigen::ArrayXcd D = es.eigenvalues();
    Eigen::ArrayXXcd V = es.eigenvectors();
    V = (V / V.row(0).array().replicate(5, 1)).eval();

    sols.row(0) = D.transpose().array();
    sols.row(1) = V.row(2).array();
    sols.row(2) = V.row(3).array();

    return sols;
}
}  // namespace poselib
