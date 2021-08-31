#include "utils.h"
#include <PoseLib/misc/essential.h>

namespace pose_lib {

// Returns MSAC score
double compute_msac_score(const CameraPose &pose, const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, double sq_threshold, size_t *inlier_count) {
    *inlier_count = 0;
    double score = 0.0;
    for (size_t k = 0; k < x.size(); ++k) {
        Eigen::Vector3d Z = (pose.R * X[k] + pose.t);
        double r2 = (Z.hnormalized() - x[k]).squaredNorm();
        if (r2 < sq_threshold && Z(2) > 0.0) {
            (*inlier_count)++;
            score += r2;
        } else {
            score += sq_threshold;
        }
    }
    return score;
}

// Returns MSAC score of the Sampson error
double compute_sampson_msac_score(const CameraPose &pose, const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2, double sq_threshold, size_t *inlier_count) {
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
        //const double Ex2_2 = E0_2 * x2_0 + E1_2 * x2_1 + E2_2;

        const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;
        const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;
        const double r2 = C * C / (Cx + Cy);

        if (r2 < sq_threshold) {
            bool cheirality = check_cheirality(pose, x1[k].homogeneous().normalized(), x2[k].homogeneous().normalized(), 0.01);
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

// Compute inliers for absolute pose estimation (using reprojection error and cheirality check)
void get_inliers(const CameraPose &pose, const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, double sq_threshold, std::vector<char> *inliers) {
    inliers->resize(x.size());
    for (size_t k = 0; k < x.size(); ++k) {
        Eigen::Vector3d Z = (pose.R * X[k] + pose.t);
        double r2 = (Z.hnormalized() - x[k]).squaredNorm();
        (*inliers)[k] = (r2 < sq_threshold && Z(2) > 0.0);
    }
}

// Compute inliers for relative pose estimation (using Sampson error)
int get_inliers(const CameraPose &pose, const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2, double sq_threshold, std::vector<char> *inliers) {
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
        //const double Ex2_2 = E0_2 * x2_0 + E1_2 * x2_1 + E2_2;

        const double C = x2_0 * Ex1_0 + x2_1 * Ex1_1 + Ex1_2;

        const double Cx = Ex1_0 * Ex1_0 + Ex1_1 * Ex1_1;
        const double Cy = Ex2_0 * Ex2_0 + Ex2_1 * Ex2_1;

        const double r2 = C * C / (Cx + Cy);

        bool inlier = (r2 < sq_threshold);
        if (inlier) {
            bool cheirality = check_cheirality(pose, x1[k].homogeneous().normalized(), x2[k].homogeneous().normalized(), 0.01);
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

// Splitmix64 PRNG
typedef uint64_t RNG_t;
int random_int(RNG_t &state) {
    state += 0x9e3779b97f4a7c15;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// Draws a random sample
void draw_sample(size_t sample_sz, size_t N, std::vector<size_t> *sample, RNG_t &rng) {
    for (int i = 0; i < sample_sz; ++i) {
        bool done = false;
        while (!done) {
            (*sample)[i] = random_int(rng) % N;

            done = true;
            for (int j = 0; j < i; ++j) {
                if ((*sample)[i] == (*sample)[j]) {
                    done = false;
                    break;
                }
            }
        }
    }
}
// Sampling for multi-camera systems
void draw_sample(size_t sample_sz, const std::vector<size_t> &N, std::vector<std::pair<size_t, size_t>> *sample, RNG_t &rng) {
    for (int i = 0; i < sample_sz; ++i) {
        bool done = false;
        while (!done) {
            (*sample)[i].first = random_int(rng) % N.size();
            if(N[(*sample)[i].first] == 0) {
                continue;
            }
            (*sample)[i].second = random_int(rng) % N[(*sample)[i].first];

            done = true;
            for (int j = 0; j < i; ++j) {
                if ((*sample)[i] == (*sample)[j]) {
                    done = false;
                    break;
                }
            }
        }
    }
}

}