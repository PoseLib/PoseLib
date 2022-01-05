#include "homography.h"
#include "../../solvers/homography_4pt.h"
#include "../bundle.h"

namespace pose_lib {

void HomographyEstimator::generate_models(std::vector<Eigen::Matrix3d> *models) {
    sampler.generate_sample(&sample);
    for (size_t k = 0; k < sample_sz; ++k) {
        x1s[k] = x1[sample[k]].homogeneous().normalized();
        x2s[k] = x2[sample[k]].homogeneous().normalized();
    }
    Eigen::Matrix3d H;
    int sols = homography_4pt(x1s, x2s, &H, true);
    if(sols > 0) {
        models->push_back(H);
    }
}

double HomographyEstimator::score_model(const Eigen::Matrix3d &H, size_t *inlier_count) const {
    return compute_homography_msac_score(H, x1, x2, opt.max_reproj_error * opt.max_reproj_error, inlier_count);
}

void HomographyEstimator::refine_model(Eigen::Matrix3d *H) const {
    BundleOptions bundle_opt;
    bundle_opt.loss_type = BundleOptions::LossType::TRUNCATED;
    bundle_opt.loss_scale = opt.max_reproj_error;
    bundle_opt.max_iterations = 25;

    refine_homography(x1, x2, H, bundle_opt);
}

}