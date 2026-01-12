#include "example_cameras.h"
#include "optim_test_utils.h"
#include "test.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/recalibrator.h>

using namespace poselib;

//////////////////////////////
// Recalibrator

namespace test::recalibrator {

double compute_rms_error(const std::vector<Point2D> &x, const std::vector<Point3D> &x_unproj, const Camera &camera) {
    double residual = 0;
    for (int i = 0; i < x.size(); ++i) {
        Eigen::Vector2d xp;
        camera.project(x_unproj[i], &xp);
        residual += (xp - x[i]).squaredNorm();
    }
    return std::sqrt(residual / x.size());
}

void sample_grid_points(int N, const Camera &camera, std::vector<Point2D> *x) {
    x->clear();
    int w = camera.width;
    int h = camera.height;
    for (int i = 1; i < N + 2; ++i) {
        for (int j = 1; j < N + 2; ++j) {
            double ii = w * static_cast<double>(i) / N;
            double jj = h * static_cast<double>(j) / N;

            Eigen::Vector2d p(ii, jj);
            // Check that camera model works for this point
            Eigen::Vector3d x_unproj;
            camera.unproject(p, &x_unproj);
            Eigen::Vector2d xp;
            camera.project(x_unproj, &xp);
            double res = (xp - p).norm();
            if (res > 1e-6) {
                continue;
            }
            x->emplace_back(ii, jj);
        }
    }
}

bool test_recalibrator_pinhole() {
    std::string camera_str = "1 PINHOLE 6214 4138 3425.62 3426.29 3118.41 2069.07";

    Camera source(camera_str);
    std::vector<Point2D> x;
    sample_grid_points(3, source, &x);
    std::vector<Point3D> x_unproj;
    source.unproject(x, &x_unproj);

    Camera target;
    target.model_id = CameraModelId::SIMPLE_PINHOLE;

    BundleOptions opt;
    opt.verbose = true;
    opt.refine_focal_length = true;
    opt.refine_principal_point = true;
    recalibrate(x, source, &target, opt);

    double rms_error = compute_rms_error(x, x_unproj, target);

    REQUIRE_SMALL(rms_error, 1.0);
    return true;
}

bool test_recalibrator_division_to_radial_fisheye() {
    std::string camera_str = "17 DIVISION 2560 1152 1265.772 1262.546 1257.333 571.722 0.005963";

    Camera source(camera_str);
    std::vector<Point2D> x;
    sample_grid_points(10, source, &x);
    std::vector<Point3D> x_unproj;
    source.unproject(x, &x_unproj);

    std::cout << "x.size() = " << x.size() << "\n";
    Camera target;
    target.model_id = CameraModelId::RADIAL;

    BundleOptions opt;
    opt.verbose = true;
    opt.refine_focal_length = true;
    opt.refine_principal_point = false;
    opt.refine_extra_params = true;
    BundleStats stats = recalibrate(x, source, &target, opt);

    double rms_error = compute_rms_error(x, x_unproj, target);

    std::cout << target.to_cameras_txt() << "\n";
    std::cout << "rms_error = " << rms_error << "\n";
    std::cout << "iter = " << stats.iterations << "\n";
    std::cout << "initial_cost = " << stats.initial_cost << "\n";
    std::cout << "cost = " << stats.cost << "\n";
    std::cout << "lambda = " << stats.lambda << "\n";
    std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    std::cout << "step_norm = " << stats.step_norm << "\n";
    std::cout << "grad_norm = " << stats.grad_norm << "\n";

    REQUIRE_SMALL(rms_error, 1.0);
    return true;
}

bool test_recalibrator_division_to_radial() {
    std::string camera_str = "17 DIVISION 2560 1152 1265.772 1262.546 1257.333 571.722 0.005963";

    Camera source(camera_str);
    std::vector<Point2D> x;
    sample_grid_points(10, source, &x);
    std::vector<Point3D> x_unproj;
    source.unproject(x, &x_unproj);

    Camera target;
    target.model_id = CameraModelId::RADIAL;

    BundleOptions opt;
    opt.verbose = true;
    opt.refine_focal_length = true;
    opt.refine_principal_point = false;
    opt.refine_extra_params = true;
    recalibrate(x, source, &target, opt);

    double rms_error = compute_rms_error(x, x_unproj, target);

    REQUIRE_SMALL(rms_error, 1.0);
    return true;
}

bool test_recalibrator_division_to_thin_prism_fisheye() {
    std::string camera_str = "17 DIVISION 2560 1152 1265.772 1262.546 1257.333 571.722 0.005963";

    Camera source(camera_str);
    std::vector<Point2D> x;
    sample_grid_points(10, source, &x);
    std::vector<Point3D> x_unproj;
    source.unproject(x, &x_unproj);

    std::cout << "x.size() = " << x.size() << "\n";
    Camera target;
    target.model_id = CameraModelId::THIN_PRISM_FISHEYE;

    BundleOptions opt;
    opt.verbose = true;
    opt.refine_focal_length = true;
    opt.refine_principal_point = false;
    opt.refine_extra_params = true;
    recalibrate(x, source, &target, opt);

    double rms_error = compute_rms_error(x, x_unproj, target);

    REQUIRE_SMALL(rms_error, 1.0);
    return true;
}

} // namespace test::recalibrator

using namespace test::recalibrator;
std::vector<Test> register_recalibrator_test() {
    return {TEST(test_recalibrator_pinhole), TEST(test_recalibrator_division_to_radial_fisheye),
            TEST(test_recalibrator_division_to_radial), TEST(test_recalibrator_division_to_thin_prism_fisheye)};
}
