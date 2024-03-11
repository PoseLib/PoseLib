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

#include "utils.h"

#include "PoseLib/misc/essential.h"

namespace poselib {

// Returns MSAC score
double compute_msac_score(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                          double sq_threshold, size_t *inlier_count) {
    *inlier_count = 0;
    double score = 0.0;
    const Eigen::Matrix3d R = pose.R();
    const double P0_0 = R(0, 0), P0_1 = R(0, 1), P0_2 = R(0, 2), P0_3 = pose.t(0);
    const double P1_0 = R(1, 0), P1_1 = R(1, 1), P1_2 = R(1, 2), P1_3 = pose.t(1);
    const double P2_0 = R(2, 0), P2_1 = R(2, 1), P2_2 = R(2, 2), P2_3 = pose.t(2);

    for (size_t k = 0; k < x.size(); ++k) {
        const double X0 = X[k](0), X1 = X[k](1), X2 = X[k](2);
        const double x0 = x[k](0), x1 = x[k](1);
        const double z0 = P0_0 * X0 + P0_1 * X1 + P0_2 * X2 + P0_3;
        const double z1 = P1_0 * X0 + P1_1 * X1 + P1_2 * X2 + P1_3;
        const double z2 = P2_0 * X0 + P2_1 * X1 + P2_2 * X2 + P2_3;
        const double inv_z2 = 1.0 / z2;

        const double r_0 = z0 * inv_z2 - x0;
        const double r_1 = z1 * inv_z2 - x1;
        const double r_sq = r_0 * r_0 + r_1 * r_1;
        if (r_sq < sq_threshold && z2 > 0.0) {
            (*inlier_count)++;
            score += r_sq;
        }
    }
    score += (x.size() - *inlier_count) * sq_threshold;
    return score;
}
double compute_msac_score(const CameraPose &pose, const std::vector<Line2D> &lines2D,
                          const std::vector<Line3D> &lines3D, double sq_threshold, size_t *inlier_count) {
    *inlier_count = 0;
    double score = 0.0;
    const Eigen::Matrix3d R = pose.R();

    for (size_t k = 0; k < lines2D.size(); ++k) {
        Eigen::Vector3d Z1 = (R * lines3D[k].X1 + pose.t);
        Eigen::Vector3d Z2 = (R * lines3D[k].X2 + pose.t);
        Eigen::Vector3d proj_line = Z1.cross(Z2);
        proj_line /= proj_line.topRows<2>().norm();

        const double r =
            std::abs(proj_line.dot(lines2D[k].x1.homogeneous())) + std::abs(proj_line.dot(lines2D[k].x2.homogeneous()));
        const double r2 = r * r;
        if (r2 < sq_threshold) {
            // TODO Cheirality check?
            (*inlier_count)++;
            score += r2;
        } else {
            score += sq_threshold;
        }
    }
    return score;
}

// Returns MSAC score of the Sampson error (checks cheirality of points as well)
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Point2D> &x1,
                                  const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count) {
    *inlier_count = 0;
    Eigen::Matrix3d E;
    essential_from_motion(pose, &E);

    // For some reason this is a lot faster than just using nice Eigen expressions...
    const double E0_0 = E(0, 0), E0_1 = E(0, 1), E0_2 = E(0, 2);
    const double E1_0 = E(1, 0), E1_1 = E(1, 1), E1_2 = E(1, 2);
    const double E2_0 = E(2, 0), E2_1 = E(2, 1), E2_2 = E(2, 2);

    double score = 0.0;
    for (size_t k = 0; k < x1.size(); ++k) {
        const double x1_0 = x1[k](0), x1_1 = x1[k](1);
        const double x2_0 = x2[k](0), x2_1 = x2[k](1);

        const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
        const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
        const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

        const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
        const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;
        // const double Ex2_2 = E0_2 * x2_0 + E1_2 * x2_1 + E2_2;

        const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;
        const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
        const double r2 = C * C / (Cx + Cy);

        if (r2 < sq_threshold) {
            bool cheirality =
                check_cheirality(pose, x1[k].homogeneous().normalized(), x2[k].homogeneous().normalized(), 0.01);
            if (cheirality) {
                (*inlier_count)++;
                score += r2;
            } else {
                score += sq_threshold;
            }
        } else {
            score += sq_threshold;
        }
    }
    return score;
}

// Returns MSAC score of the Sampson error (no cheirality check)
double compute_sampson_msac_score(const Eigen::Matrix3d &E, const std::vector<Point2D> &x1,
                                  const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count) {
    *inlier_count = 0;

    // For some reason this is a lot faster than just using nice Eigen expressions...
    const double E0_0 = E(0, 0), E0_1 = E(0, 1), E0_2 = E(0, 2);
    const double E1_0 = E(1, 0), E1_1 = E(1, 1), E1_2 = E(1, 2);
    const double E2_0 = E(2, 0), E2_1 = E(2, 1), E2_2 = E(2, 2);

    double score = 0.0;
    for (size_t k = 0; k < x1.size(); ++k) {
        const double x1_0 = x1[k](0), x1_1 = x1[k](1);
        const double x2_0 = x2[k](0), x2_1 = x2[k](1);

        const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
        const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
        const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

        const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
        const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;
        // const double Ex2_2 = E0_2 * x2_0 + E1_2 * x2_1 + E2_2;

        const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;
        const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
        const double r2 = C * C / (Cx + Cy);

        if (r2 < sq_threshold) {
            (*inlier_count)++;
            score += r2;
        } else {
            score += sq_threshold;
        }
    }
    return score;
}

double compute_homography_msac_score(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1,
                                     const std::vector<Point2D> &x2, double sq_threshold, size_t *inlier_count) {
    *inlier_count = 0;
    double score = 0;

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
        const double r2 = r0 * r0 + r1 * r1;

        if (r2 < sq_threshold) {
            (*inlier_count)++;
            score += r2;
        } else {
            score += sq_threshold;
        }
    }
    return score;
}

void get_homography_inliers(const Eigen::Matrix3d &H, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                            double sq_threshold, std::vector<char> *inliers) {
    const double H0_0 = H(0, 0), H0_1 = H(0, 1), H0_2 = H(0, 2);
    const double H1_0 = H(1, 0), H1_1 = H(1, 1), H1_2 = H(1, 2);
    const double H2_0 = H(2, 0), H2_1 = H(2, 1), H2_2 = H(2, 2);

    inliers->resize(x1.size());
    for (size_t k = 0; k < x1.size(); ++k) {
        const double x1_0 = x1[k](0), x1_1 = x1[k](1);
        const double x2_0 = x2[k](0), x2_1 = x2[k](1);

        const double Hx1_0 = H0_0 * x1_0 + H0_1 * x1_1 + H0_2;
        const double Hx1_1 = H1_0 * x1_0 + H1_1 * x1_1 + H1_2;
        const double inv_Hx1_2 = 1.0 / (H2_0 * x1_0 + H2_1 * x1_1 + H2_2);

        const double r0 = Hx1_0 * inv_Hx1_2 - x2_0;
        const double r1 = Hx1_1 * inv_Hx1_2 - x2_1;
        const double r2 = r0 * r0 + r1 * r1;
        (*inliers)[k] = (r2 < sq_threshold);
    }
}

// Returns MSAC score for the 1D radial camera model
double compute_msac_score_1D_radial(const CameraPose &pose, const std::vector<Point2D> &x,
                                    const std::vector<Point3D> &X, double sq_threshold, size_t *inlier_count) {
    *inlier_count = 0;
    const Eigen::Matrix3d R = pose.R();
    double score = 0.0;
    for (size_t k = 0; k < x.size(); ++k) {
        Eigen::Vector2d z = (R * X[k] + pose.t).topRows<2>().normalized();
        const double alpha = z.dot(x[k]);
        const double r2 = (x[k] - alpha * z).squaredNorm();
        if (r2 < sq_threshold && alpha > 0.0) {
            (*inlier_count)++;
            score += r2;
        } else {
            score += sq_threshold;
        }
    }
    return score;
}

// Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
void get_inliers(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                 double sq_threshold, std::vector<char> *inliers) {
    inliers->resize(x.size());
    const Eigen::Matrix3d R = pose.R();

    for (size_t k = 0; k < x.size(); ++k) {
        Eigen::Vector3d Z = (R * X[k] + pose.t);
        double r2 = (Z.hnormalized() - x[k]).squaredNorm();
        (*inliers)[k] = (r2 < sq_threshold && Z(2) > 0.0);
    }
}

void get_inliers(const CameraPose &pose, const std::vector<Line2D> &lines2D, const std::vector<Line3D> &lines3D,
                 double sq_threshold, std::vector<char> *inliers) {
    inliers->resize(lines2D.size());
    const Eigen::Matrix3d R = pose.R();

    for (size_t k = 0; k < lines2D.size(); ++k) {
        Eigen::Vector3d Z1 = (R * lines3D[k].X1 + pose.t);
        Eigen::Vector3d Z2 = (R * lines3D[k].X2 + pose.t);
        Eigen::Vector3d proj_line = Z1.cross(Z2);
        proj_line /= proj_line.topRows<2>().norm();

        const double r =
            std::abs(proj_line.dot(lines2D[k].x1.homogeneous())) + std::abs(proj_line.dot(lines2D[k].x2.homogeneous()));
        const double r2 = r * r;
        (*inliers)[k] = (r2 < sq_threshold);
    }
}

// Compute inliers for relative pose estimation (using Sampson error)
int get_inliers(const CameraPose &pose, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                double sq_threshold, std::vector<char> *inliers) {
    inliers->resize(x1.size());
    Eigen::Matrix3d E;
    essential_from_motion(pose, &E);
    const double E0_0 = E(0, 0), E0_1 = E(0, 1), E0_2 = E(0, 2);
    const double E1_0 = E(1, 0), E1_1 = E(1, 1), E1_2 = E(1, 2);
    const double E2_0 = E(2, 0), E2_1 = E(2, 1), E2_2 = E(2, 2);

    size_t inlier_count = 0.0;
    for (size_t k = 0; k < x1.size(); ++k) {
        const double x1_0 = x1[k](0), x1_1 = x1[k](1);
        const double x2_0 = x2[k](0), x2_1 = x2[k](1);

        const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
        const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
        const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

        const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
        const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;
        // const double Ex2_2 = E0_2 * x2_0 + E1_2 * x2_1 + E2_2;

        const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;

        const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;

        const double r2 = C * C / (Cx + Cy);

        bool inlier = (r2 < sq_threshold);
        if (inlier) {
            bool cheirality =
                check_cheirality(pose, x1[k].homogeneous().normalized(), x2[k].homogeneous().normalized(), 0.01);
            if (cheirality) {
                inlier_count++;
            } else {
                inlier = false;
            }
        }
        (*inliers)[k] = inlier;
    }
    return inlier_count;
}

// Compute inliers for relative pose estimation (using Sampson error)
int get_inliers(const Eigen::Matrix3d &E, const std::vector<Point2D> &x1, const std::vector<Point2D> &x2,
                double sq_threshold, std::vector<char> *inliers) {
    inliers->resize(x1.size());
    const double E0_0 = E(0, 0), E0_1 = E(0, 1), E0_2 = E(0, 2);
    const double E1_0 = E(1, 0), E1_1 = E(1, 1), E1_2 = E(1, 2);
    const double E2_0 = E(2, 0), E2_1 = E(2, 1), E2_2 = E(2, 2);

    size_t inlier_count = 0.0;
    for (size_t k = 0; k < x1.size(); ++k) {
        const double x1_0 = x1[k](0), x1_1 = x1[k](1);
        const double x2_0 = x2[k](0), x2_1 = x2[k](1);

        const double Ex1_0 = E0_0 * x1_0 + E0_1 * x1_1 + E0_2;
        const double Ex1_1 = E1_0 * x1_0 + E1_1 * x1_1 + E1_2;
        const double Ex1_2 = E2_0 * x1_0 + E2_1 * x1_1 + E2_2;

        const double Ex2_0 = E0_0 * x2_0 + E1_0 * x2_1 + E2_0;
        const double Ex2_1 = E0_1 * x2_0 + E1_1 * x2_1 + E2_1;
        // const double Ex2_2 = E0_2 * x2_0 + E1_2 * x2_1 + E2_2;

        const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;

        const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;

        const double r2 = C * C / (Cx + Cy);

        bool inlier = (r2 < sq_threshold);
        if (inlier) {
            inlier_count++;
        }
        (*inliers)[k] = inlier;
    }
    return inlier_count;
}

// Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
void get_inliers_1D_radial(const CameraPose &pose, const std::vector<Point2D> &x, const std::vector<Point3D> &X,
                           double sq_threshold, std::vector<char> *inliers) {
    inliers->resize(x.size());
    const Eigen::Matrix3d R = pose.R();

    for (size_t k = 0; k < x.size(); ++k) {
        Eigen::Vector2d z = (R * X[k] + pose.t).topRows<2>().normalized();
        const double alpha = z.dot(x[k]);
        const double r2 = (x[k] - alpha * z).squaredNorm();
        (*inliers)[k] = (r2 < sq_threshold && alpha > 0.0);
    }
}

double normalize_points(std::vector<Eigen::Vector2d> &x1, std::vector<Eigen::Vector2d> &x2, Eigen::Matrix3d &T1,
                        Eigen::Matrix3d &T2, bool normalize_scale, bool normalize_centroid, bool shared_scale) {

    T1.setIdentity();
    T2.setIdentity();

    if (normalize_centroid) {
        Eigen::Vector2d c1(0, 0), c2(0, 0);
        for (size_t k = 0; k < x1.size(); ++k) {
            c1 += x1[k];
            c2 += x2[k];
        }
        c1 /= x1.size();
        c2 /= x2.size();

        T1.block<2, 1>(0, 2) = -c1;
        T2.block<2, 1>(0, 2) = -c2;
        for (size_t k = 0; k < x1.size(); ++k) {
            x1[k] -= c1;
            x2[k] -= c2;
        }
    }

    if (normalize_scale && shared_scale) {
        double scale = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            scale += x1[k].norm();
            scale += x2[k].norm();
        }
        scale /= std::sqrt(2) * x1.size();

        for (size_t k = 0; k < x1.size(); ++k) {
            x1[k] /= scale;
            x2[k] /= scale;
        }

        T1.block<2, 3>(0, 0) *= 1.0 / scale;
        T2.block<2, 3>(0, 0) *= 1.0 / scale;

        return scale;
    } else if (normalize_scale && !shared_scale) {
        double scale1 = 0.0, scale2 = 0.0;
        for (size_t k = 0; k < x1.size(); ++k) {
            scale1 += x1[k].norm();
            scale2 += x2[k].norm();
        }
        scale1 /= x1.size() / std::sqrt(2);
        scale2 /= x2.size() / std::sqrt(2);

        for (size_t k = 0; k < x1.size(); ++k) {
            x1[k] /= scale1;
            x2[k] /= scale2;
        }

        T1.block<2, 3>(0, 0) *= 1.0 / scale1;
        T2.block<2, 3>(0, 0) *= 1.0 / scale2;

        return std::sqrt(scale1 * scale2);
    }
    return 1.0;
}

bool calculate_RFC(const Eigen::Matrix3d &F) {
    float den, num;

    den = F(0, 0) * F(0, 1) * F(2, 0) * F(2, 2) - F(0, 0) * F(0, 2) * F(2, 0) * F(2, 1) +
          F(0, 1) * F(0, 1) * F(2, 1) * F(2, 2) - F(0, 1) * F(0, 2) * F(2, 1) * F(2, 1) +
          F(1, 0) * F(1, 1) * F(2, 0) * F(2, 2) - F(1, 0) * F(1, 2) * F(2, 0) * F(2, 1) +
          F(1, 1) * F(1, 1) * F(2, 1) * F(2, 2) - F(1, 1) * F(1, 2) * F(2, 1) * F(2, 1);

    num = -F(2, 2) * (F(0, 1) * F(0, 2) * F(2, 2) - F(0, 2) * F(0, 2) * F(2, 1) + F(1, 1) * F(1, 2) * F(2, 2) -
                      F(1, 2) * F(1, 2) * F(2, 1));

    if (num * den < 0)
        return false;

    den = F(0, 0) * F(1, 0) * F(0, 2) * F(2, 2) - F(0, 0) * F(2, 0) * F(0, 2) * F(1, 2) +
          F(1, 0) * F(1, 0) * F(1, 2) * F(2, 2) - F(1, 0) * F(2, 0) * F(1, 2) * F(1, 2) +
          F(0, 1) * F(1, 1) * F(0, 2) * F(2, 2) - F(0, 1) * F(2, 1) * F(0, 2) * F(1, 2) +
          F(1, 1) * F(1, 1) * F(1, 2) * F(2, 2) - F(1, 1) * F(2, 1) * F(1, 2) * F(1, 2);

    num = -F(2, 2) * (F(1, 0) * F(2, 0) * F(2, 2) - F(2, 0) * F(2, 0) * F(1, 2) + F(1, 1) * F(2, 1) * F(2, 2) -
                      F(2, 1) * F(2, 1) * F(1, 2));

    if (num * den < 0)
        return false;
    return true;
}

} // namespace poselib