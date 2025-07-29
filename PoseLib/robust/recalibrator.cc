#include "recalibrator.h"

#include "optim/jacobian_accumulator.h"
#include "optim/lm_impl.h"

namespace poselib {

BundleStats recalibrate(const std::vector<Point2D> &x, const Camera &source, Camera *target, const BundleOptions &opt,
                        bool rescale) {
    // Rescale points to improve numerics
    if (rescale) {
        double s0 = source.focal();
        std::vector<Point2D> x_rescale = x;
        for (Point2D &p : x_rescale) {
            p = p / s0;
        }
        Camera source_rescale = source;
        source_rescale.rescale(1 / s0);
        BundleStats stats = recalibrate(x_rescale, source_rescale, target, opt, false);
        target->rescale(s0);
        return stats;
    }

    // Unproject points using source camera model
    std::vector<Point3D> x_unproj;
    source.unproject(x, &x_unproj);

    // Initialize target camera
    target->init_params();
    target->width = source.width;
    target->height = source.height;
    Eigen::Vector2d pp = source.principal_point();
    target->set_principal_point(pp(0), pp(1));
    target->set_focal(source.focal());

    BundleStats stats;
    BundleOptions opt0;
    opt0.loss_type = BundleOptions::LossType::TRIVIAL;
    opt0.loss_scale = 1.0;
    opt0.refine_principal_point = false;

    if (opt.refine_extra_params) {
        // First only refine extra parameters
        opt0.refine_focal_length = false;
        opt0.refine_extra_params = true;
        std::vector<size_t> cam_ref_idx = target->get_param_refinement_idx(opt0);
        RecalibratorRefiner<> refiner(x, x_unproj, cam_ref_idx);
        stats = lm_impl(refiner, target, opt0);
    }

    if (opt.refine_focal_length) {
        // and then only focal length
        opt0.refine_focal_length = true;
        opt0.refine_extra_params = false;
        std::vector<size_t> cam_ref_idx = target->get_param_refinement_idx(opt0);
        RecalibratorRefiner<> refiner(x, x_unproj, cam_ref_idx);
        stats = lm_impl(refiner, target, opt0);
    }

    // Refine everyhing
    opt0.refine_focal_length = opt.refine_focal_length;
    opt0.refine_extra_params = opt.refine_extra_params;
    opt0.refine_principal_point = opt.refine_principal_point;
    std::vector<size_t> cam_ref_idx = target->get_param_refinement_idx(opt0);
    RecalibratorRefiner<> refiner(x, x_unproj, cam_ref_idx);
    stats = lm_impl(refiner, target, opt0);

    return stats;
}

} // namespace poselib