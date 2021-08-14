#include "ransac.h"
#include "bundle.h"
#include <PoseLib/gp3p.h>
#include <PoseLib/misc/essential.h>
#include <PoseLib/p3p.h>
#include <PoseLib/relpose_5pt.h>

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
    double score = 0.0;
    for (size_t k = 0; k < x1.size(); ++k) {
        const double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());
        const double JC = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() + (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

        const double r2 = C * C / JC;
        if (r2 < sq_threshold) {
            (*inlier_count)++;
            score += r2;
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
void get_inliers(const CameraPose &pose, const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2, double sq_threshold, std::vector<char> *inliers) {
    inliers->resize(x1.size());
    Eigen::Matrix3d E;
    essential_from_motion(pose, &E);
    double score = 0.0;
    for (size_t k = 0; k < x1.size(); ++k) {
        const double C = x2[k].homogeneous().dot(E * x1[k].homogeneous());
        const double JC = (E.block<2, 3>(0, 0) * x1[k].homogeneous()).squaredNorm() + (E.block<3, 2>(0, 0).transpose() * x2[k].homogeneous()).squaredNorm();

        const double r2 = C * C / JC;
        (*inliers)[k] = (r2 < sq_threshold);
    }
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

class AbsolutePoseEstimator {
  public:
    AbsolutePoseEstimator(const RansacOptions &ransac_opt,
                          const std::vector<Eigen::Vector2d> &points2D,
                          const std::vector<Eigen::Vector3d> &points3D)
        : num_data(points2D.size()), opt(ransac_opt), x(points2D), X(points3D) {
        rng = opt.seed;
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models) {
        draw_sample(sample_sz, num_data, &sample, rng);
        for (size_t k = 0; k < sample_sz; ++k) {
            xs[k] = x[sample[k]].homogeneous().normalized();
            Xs[k] = X[sample[k]];
        }
        p3p(xs, Xs, models);
    }

    double score_model(const CameraPose &pose, size_t *inlier_count) const {
        return compute_msac_score(pose, x, X, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
    }

    void refine_model(CameraPose *pose) const {
        BundleOptions bundle_opt;
        bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
        bundle_opt.loss_scale = opt.max_reproj_error;
        bundle_opt.max_iterations = 25;

        // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
        // TODO: experiment with good thresholds for copy vs iterating full point set
        bundle_adjust(x, X, pose, bundle_opt);
    }

    const size_t sample_sz = 3;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Eigen::Vector2d> &x;
    const std::vector<Eigen::Vector3d> &X;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> xs, Xs;
    std::vector<size_t> sample;
};

class GeneralizedAbsolutePoseEstimator {
  public:
    GeneralizedAbsolutePoseEstimator(const RansacOptions &ransac_opt,
                                     const std::vector<std::vector<Eigen::Vector2d>> &points2D,
                                     const std::vector<std::vector<Eigen::Vector3d>> &points3D,
                                     const std::vector<CameraPose> &camera_ext)
        : num_cams(points2D.size()), opt(ransac_opt),
          x(points2D), X(points3D), rig_poses(camera_ext) {
        rng = opt.seed;
        ps.resize(sample_sz);
        xs.resize(sample_sz);
        Xs.resize(sample_sz);
        sample.resize(sample_sz);
        camera_centers.resize(num_cams);
        for (size_t k = 0; k < num_cams; ++k) {
            camera_centers[k] = -camera_ext[k].R.transpose() * camera_ext[k].t;
        }

        num_data = 0;
        num_pts_camera.resize(num_cams);
        for (size_t k = 0; k < num_cams; ++k) {
            num_pts_camera[k] = points2D[k].size();
            num_data += num_pts_camera[k];
        }
    }

    void generate_models(std::vector<CameraPose> *models) {
        draw_sample(sample_sz, num_pts_camera, &sample, rng);

        for (size_t k = 0; k < sample_sz; ++k) {
            const size_t cam_k = sample[k].first;
            const size_t pt_k = sample[k].second;
            ps[k] = camera_centers[cam_k];
            xs[k] = rig_poses[cam_k].R.transpose() * (x[cam_k][pt_k].homogeneous().normalized());
            Xs[k] = X[cam_k][pt_k];
        }
        gp3p(ps, xs, Xs, models);
    }

    double score_model(const CameraPose &pose, size_t *inlier_count) const {
        const double sq_threshold = opt.max_reproj_error * opt.max_reproj_error;
        double score = 0;
        *inlier_count = 0;
        size_t cam_inlier_count;
        for (size_t k = 0; k < num_cams; ++k) {
            CameraPose full_pose;
            full_pose.R = rig_poses[k].R * pose.R;
            full_pose.t = rig_poses[k].R * pose.t + rig_poses[k].t;

            score += compute_msac_score(full_pose, x[k], X[k], sq_threshold, &cam_inlier_count);
            *inlier_count += cam_inlier_count;
        }
        return score;
    }

    void refine_model(CameraPose *pose) const {
        BundleOptions bundle_opt;
        bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
        bundle_opt.loss_scale = opt.max_reproj_error;
        bundle_opt.max_iterations = 25;
        generalized_bundle_adjust(x, X, rig_poses, pose, bundle_opt);
    }

    const size_t sample_sz = 3;
    size_t num_data;
    const size_t num_cams;

  private:
    const RansacOptions &opt;
    const std::vector<std::vector<Eigen::Vector2d>> &x;
    const std::vector<std::vector<Eigen::Vector3d>> &X;
    const std::vector<CameraPose> &rig_poses;
    std::vector<Eigen::Vector3d> camera_centers;
    std::vector<size_t> num_pts_camera; // number of points in each camera

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> ps, xs, Xs;
    std::vector<std::pair<size_t, size_t>> sample;
};

class RelativePoseEstimator {
  public:
    RelativePoseEstimator(const RansacOptions &ransac_opt,
                          const std::vector<Eigen::Vector2d> &points2D_1,
                          const std::vector<Eigen::Vector2d> &points2D_2)
        : num_data(points2D_1.size()), opt(ransac_opt), x1(points2D_1), x2(points2D_2) {
        rng = opt.seed;
        x1s.resize(sample_sz);
        x2s.resize(sample_sz);
        sample.resize(sample_sz);
    }

    void generate_models(std::vector<CameraPose> *models) {
        draw_sample(sample_sz, num_data, &sample, rng);
        for (size_t k = 0; k < sample_sz; ++k) {
            x1s[k] = x1[sample[k]].homogeneous().normalized();
            x2s[k] = x2[sample[k]].homogeneous().normalized();
        }
        relpose_5pt(x1s, x2s, models);
    }

    double score_model(const CameraPose &pose, size_t *inlier_count) const {
        return compute_sampson_msac_score(pose, x1, x2, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
    }

    void refine_model(CameraPose *pose) const {
        BundleOptions bundle_opt;
        bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
        bundle_opt.loss_scale = opt.max_reproj_error;
        bundle_opt.max_iterations = 25;

        // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
        // TODO: experiment with good thresholds for copy vs iterating full point set
        refine_sampson(x1, x2, pose, bundle_opt);
    }

    const size_t sample_sz = 5;
    const size_t num_data;

  private:
    const RansacOptions &opt;
    const std::vector<Eigen::Vector2d> &x1;
    const std::vector<Eigen::Vector2d> &x2;

    RNG_t rng;
    // pre-allocated vectors for sampling
    std::vector<Eigen::Vector3d> x1s, x2s;
    std::vector<size_t> sample;
};

template <typename Solver>
int ransac(Solver &estimator, const RansacOptions &opt, CameraPose *best_model) {
    if (estimator.num_data < estimator.sample_sz) {
        best_model->R.setIdentity();
        best_model->t.setZero();
        return 0;
    }

    // Score/Inliers for best model found so far
    double best_msac_score = std::numeric_limits<double>::max();
    size_t best_inlier_count = 0;
    size_t best_minimal_inlier_count = 0;

    const double log_prob_missing_model = std::log(1.0 - opt.success_prob);
    size_t inlier_count = 0;
    std::vector<CameraPose> models;
    size_t dynamic_max_iter = opt.max_iterations;
    size_t iter;
    for (iter = 0; iter < opt.max_iterations; iter++) {

        if (iter > opt.min_iterations && iter > dynamic_max_iter) {
            break;
        }
        models.clear();
        estimator.generate_models(&models);

        // Find best model among candidates
        int best_model_ind = -1;
        for (size_t i = 0; i < models.size(); ++i) {
            double score_msac = estimator.score_model(models[i], &inlier_count);

            if (best_minimal_inlier_count < inlier_count) {
                best_minimal_inlier_count = inlier_count;
                best_model_ind = i;

                // check if we should update best model already
                if (score_msac < best_msac_score) {
                    best_msac_score = score_msac;
                    *best_model = models[i];
                    best_inlier_count = inlier_count;
                }
            }
        }

        if (best_model_ind == -1)
            continue;

        // Refinement
        CameraPose refined_model = models[best_model_ind];
        estimator.refine_model(&refined_model);
        double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
        if (refined_msac_score < best_msac_score) {
            best_msac_score = refined_msac_score;
            best_inlier_count = inlier_count;
            *best_model = refined_model;
        }

        // update number of iterations
        const double inlier_ratio = static_cast<double>(best_inlier_count) / static_cast<double>(estimator.num_data);
        if (inlier_ratio >= 0.9999) {
            // this is to avoid log(prob_outlier) = -inf below
            dynamic_max_iter = opt.min_iterations;
        } else if (inlier_ratio <= 0.0001) {
            // this is to avoid log(prob_outlier) = 0 below
            dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(inlier_ratio, estimator.sample_sz);
            dynamic_max_iter = std::ceil(log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }
    }

    // Final refinement
    CameraPose refined_model = *best_model;
    estimator.refine_model(&refined_model);
    double refined_msac_score = estimator.score_model(refined_model, &inlier_count);
    if (refined_msac_score < best_msac_score) {
        *best_model = refined_model;
        best_inlier_count = inlier_count;
    }

    return best_inlier_count;
}

int ransac_pose(const std::vector<Eigen::Vector2d> &x, const std::vector<Eigen::Vector3d> &X, const RansacOptions &opt,
                CameraPose *best_model, std::vector<char> *best_inliers) {

    AbsolutePoseEstimator estimator(opt, x, X);
    size_t num_inl = ransac<AbsolutePoseEstimator>(estimator, opt, best_model);

    get_inliers(*best_model, x, X, opt.max_reproj_error * opt.max_reproj_error, best_inliers);

    return num_inl;
}

int ransac_gen_pose(const std::vector<std::vector<Eigen::Vector2d>> &x, const std::vector<std::vector<Eigen::Vector3d>> &X, const std::vector<CameraPose> &camera_ext, const RansacOptions &opt,
                    CameraPose *best_model, std::vector<std::vector<char>> *best_inliers) {

    GeneralizedAbsolutePoseEstimator estimator(opt, x, X, camera_ext);
    size_t num_inl = ransac<GeneralizedAbsolutePoseEstimator>(estimator, opt, best_model);

    best_inliers->resize(camera_ext.size());
    for (size_t k = 0; k < camera_ext.size(); ++k) {
        CameraPose full_pose;
        full_pose.R = camera_ext[k].R * best_model->R;
        full_pose.t = camera_ext[k].R * best_model->t + camera_ext[k].t;
        get_inliers(full_pose, x[k], X[k], opt.max_reproj_error * opt.max_reproj_error, &(*best_inliers)[k]);
    }

    return num_inl;
}

int ransac_relpose(const std::vector<Eigen::Vector2d> &x1, const std::vector<Eigen::Vector2d> &x2,
                   const RansacOptions &opt, CameraPose *best_model, std::vector<char> *best_inliers) {

    RelativePoseEstimator estimator(opt, x1, x2);
    size_t num_inl = ransac<RelativePoseEstimator>(estimator, opt, best_model);

    get_inliers(*best_model, x1, x2, opt.max_reproj_error * opt.max_reproj_error, best_inliers);

    return num_inl;
}

} // namespace pose_lib