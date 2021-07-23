#include "ransac.h"
#include "bundle.h"
#include <PoseLib/p3p.h>
#include <random>

namespace pose_lib {

// Returns MSAC score
double score_model(const CameraPose &pose, const std::vector<Eigen::Vector2d>& x, const std::vector<Eigen::Vector3d>& X, double sq_threshold, size_t *inlier_count, std::vector<char> *inliers) {
    inliers->resize(x.size());
    *inlier_count = 0;
    double score = 0.0;

    for(size_t k = 0; k < x.size(); ++k) {
        Eigen::Vector3d Z = (pose.R * X[k] + pose.t);
        double r2 = (Z.hnormalized() - x[k]).squaredNorm();

        if (r2 < sq_threshold && Z(2) > 0.0) {
            (*inliers)[k] = 1;
            (*inlier_count)++;
            score += r2;
        } else {
            (*inliers)[k] = 0;
            score += sq_threshold;
        }
    }		
    return score;
}

// Non-linear refinement of MSAC score
int refine_model(const std::vector<Eigen::Vector2d>& x, const std::vector<Eigen::Vector3d>& X, CameraPose *pose, double max_reproj) {
    
    BundleOptions opt;
    opt.loss_type = BundleOptions::LossType::TRUNCATED;
    opt.loss_scale = max_reproj;    
    opt.max_iterations = 25;
    
    // TODO: for high outlier scenarios, make a copy of (x,X) and find points close to inlier threshold
    // TODO: experiment with good thresholds for copy vs iterating full point set
    return bundle_adjust(x, X, pose, opt);
}

void draw_sample(size_t sample_sz, size_t N, std::vector<size_t> *sample, std::mt19937 &rng ) {
    std::uniform_int_distribution<size_t> dist(0, N-1);
    // TODO redo implementation based on benchmarking
    //      this is probably okay for now, but I guess mt19937 is quite slow
    for (int i = 0; i < sample_sz; ++i) {
        bool done = false;
        while (!done) {
            (*sample)[i] = dist(rng);
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


int ransac_pose(const std::vector<Eigen::Vector2d>& x, const std::vector<Eigen::Vector3d>& X, const RansacOptions& opt,
    CameraPose* best_model, std::vector<char>* best_inliers) {

    // Constants
    const size_t num_pts = x.size();
    const size_t sample_sz = 3;
    const double sq_threshold = opt.max_reproj_error * opt.max_reproj_error;
    const double log_prob_missing_model = std::log(1.0 - opt.success_prob);
    std::mt19937 rng(opt.seed);

    if (num_pts < sample_sz) {
        best_model->R.setIdentity();
        best_model->t.setZero();
        best_inliers->clear();
        return 0;
    }

    // Pre-allocate vectors for inliers and sampled points etc.    
    std::vector<Eigen::Vector3d> xs(sample_sz), Xs(sample_sz);
    std::vector<CameraPose> models;
    std::vector<size_t> sample(sample_sz);
    best_inliers->resize(num_pts);

    // Score/Inliers for best model found so far
    double best_msac_score = std::numeric_limits<double>::max();
    size_t best_inlier_count = 0;
    size_t best_minimal_inlier_count = 0;
    
    // for storing scores for model hypothesis
    size_t inlier_count = 0;
    std::vector<char> inliers(num_pts);

    size_t dynamic_max_iter = opt.max_iterations;
    size_t iter;
    for (iter = 0; iter < opt.max_iterations; iter++) {

        if (iter > opt.min_iterations && iter > dynamic_max_iter) {
            break;
        }

        draw_sample(sample_sz, num_pts, &sample, rng);

        for (size_t i = 0; i < sample_sz; ++i) {
            xs[i] = x[sample[i]].homogeneous().normalized();
            Xs[i] = X[sample[i]];
        }

        p3p(xs, Xs, &models);

        // Find best model among candidates
        int best_model_ind = -1;
        
        
        for (size_t i = 0; i < models.size(); ++i) {            
            double score_msac = score_model(models[i], x, X, sq_threshold, &inlier_count, &inliers);
            
            if (best_minimal_inlier_count < inlier_count) {
                best_minimal_inlier_count = inlier_count;					
                best_model_ind = i;			

                // check if we should update best model already
                if (score_msac < best_msac_score) {
                    best_msac_score = score_msac;
                    *best_model = models[i];
                    best_inlier_count = inlier_count;
                    std::swap(inliers, *best_inliers);
                }										
            }
        }

        if (best_model_ind == -1)
            continue;

        // new best model: perform LO and update stopping criterion
        
        // Refinement
        CameraPose refined_model = models[best_model_ind];
        refine_model(x, X, &refined_model, opt.max_reproj_error);
        double refined_msac_score = score_model(refined_model, x, X, sq_threshold, &inlier_count, &inliers);
        if (refined_msac_score < best_msac_score) {
            best_msac_score = refined_msac_score;
            best_inlier_count = inlier_count;
            *best_model = refined_model;
            std::swap(inliers, *best_inliers);
        }

        // update number of iterations
        const double inlier_ratio = static_cast<double>(best_inlier_count) / static_cast<double>(num_pts);
        if (inlier_ratio >= 0.9999) {
            // this is to avoid log(prob_outlier) = -inf below
            dynamic_max_iter = opt.min_iterations;
        } else if (inlier_ratio <= 0.0001) {
            // this is to avoid log(prob_outlier) = 0 below
            dynamic_max_iter = opt.max_iterations;
        } else {
            const double prob_outlier = 1.0 - std::pow(inlier_ratio, sample_sz);
            dynamic_max_iter = std::ceil(log_prob_missing_model / std::log(prob_outlier) * opt.dyn_num_trials_mult);
        }


    }

    // Final refinement
    CameraPose refined_model = *best_model;    
    refine_model(x, X, &refined_model, opt.max_reproj_error);
    double refined_msac_score = score_model(refined_model, x, X, sq_threshold, &inlier_count, &inliers);
    if (refined_msac_score < best_msac_score) {
        *best_model = refined_model;
        best_inlier_count = inlier_count;
        std::swap(inliers, *best_inliers);
    }

    return best_inlier_count;
}

} // namespace pose_lib