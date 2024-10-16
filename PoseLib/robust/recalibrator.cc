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

    std::vector<size_t> cam_ref_idx = target->get_param_refinement_idx(opt);
    if (cam_ref_idx.size() == 0) {
        return BundleStats();
    }

    std::vector<Point3D> x_unproj;
    source.unproject(x, &x_unproj);
    RecalibratorRefiner<> refiner(x, x_unproj, cam_ref_idx);

    // Initialize target camera
    target->init_params();
    target->width = source.width;
    target->height = source.height;
    Eigen::Vector2d pp = source.principal_point();
    ;
    target->set_principal_point(pp(0), pp(1));
    target->set_focal(source.focal());

    BundleStats stats;
    if (opt.verbose) {
        stats = lm_impl(refiner, target, opt, print_iteration);
    } else {
        stats = lm_impl(refiner, target, opt);
    }
    return stats;
}

} // namespace poselib