#include "test.h"
#include "optim_test_utils.h"
#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/jacobian_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/homography.h>
#include <PoseLib/robust/optim/lm_impl.h>


using namespace poselib;


//////////////////////////////
// Relative pose

namespace test::homography {


CameraPose random_camera() {
    Eigen::Vector3d cc;
    cc.setRandom();
    cc.normalize();
    cc *= 2.0;

    // Lookat point
    Eigen::Vector3d p;
    p.setRandom();
    p *= 0.1;

    Eigen::Vector3d r3 = p - cc;
    r3.normalize();

    Eigen::Vector3d r2 = r3.cross(Eigen::Vector3d::UnitX());
    r2.normalize();
    Eigen::Vector3d r1 = r2.cross(r3);

    Eigen::Matrix3d R;
    R.row(0) = r1;
    R.row(1) = r2;
    R.row(2) = r3;

    return CameraPose(R, -R * cc);
}

void setup_scene(int N, Eigen::Matrix3d &H, std::vector<Point2D> &x1,
                 std::vector<Point2D> &x2, Camera &cam1, Camera &cam2) {

    CameraPose p1 = random_camera();
    CameraPose p2 = random_camera();
    CameraPose p = p2.compose(p1.inverse());
        
    Eigen::Vector3d normal;
    normal.setRandom();
    normal.normalize();
    Eigen::Vector3d X0;
    X0.setRandom();

    X0 = p1.apply_inverse(X0);
    normal = p1.derotate(normal);

    // normal.dot(X0) = alpha
    normal = normal / normal.dot(X0);

    // X = lambda * x1
    // 1 = n'*X = lambda * n'*x1
    // lambda = 1 / n'*x1

    // (1/*n'*x1) * R * x1 + t = 
    // (R + t*n') * x1

    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector3d Xi;
        Xi.setRandom();
        Xi = p1.apply(Xi);

        // n'*(lambda * Xi) = 1
        // lambda = 1 / n'*Xi
        Xi = (1.0 / normal.dot(Xi)) * Xi;

        Eigen::Vector2d xi;
        cam1.project(Xi, &xi);
        x1.push_back(xi);

        cam2.project(p.apply(Xi), &xi);
        x2.push_back(xi);
    }

    H = p.R() + p.t * normal.transpose();
}


bool test_homography_normal_acc() {
    
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Eigen::Matrix3d H;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, H, x1, x2, camera, camera);

    NormalAccumulator<TrivialLoss> acc(8);
    PinholeHomographyRefiner<decltype(acc)> refiner(x1,x2);

    // Check that residual is zero
    acc.reset_residual();
    refiner.compute_residual(acc, H);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    // Check the gradient is zero
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, H);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}

bool test_homography_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Eigen::Matrix3d H;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, H, x1, x2, camera, camera);

    PinholeHomographyRefiner<TestAccumulator> refiner(x1,x2);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner),Eigen::Matrix3d,8>(refiner, H, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, H);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, H);
    double r2 = 0.0;
    for(int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}



bool test_homography_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    Eigen::Matrix3d H;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, H, x1, x2, camera, camera);

    // Add some noise
    for(int i = 0; i < N; ++i) {
        Eigen::Vector2d n;
        n.setRandom();        
        x1[i] += 0.001 * n;
        n.setRandom();        
        x2[i] += 0.001 * n;
    }

    NormalAccumulator acc(8);
    PinholeHomographyRefiner<decltype(acc)> refiner(x1,x2);
    
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, acc, &H, bundle_opt, print_iteration);

    
    //std::cout << "iter = " << stats.iterations << "\n";
    //std::cout << "initial_cost = " << stats.initial_cost << "\n";
    //std::cout << "cost = " << stats.cost << "\n";
    //std::cout << "lambda = " << stats.lambda << "\n";
    //std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    //std::cout << "step_norm = " << stats.step_norm << "\n";
    //std::cout << "grad_norm = " << stats.grad_norm << "\n";
    

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);
    
    return true;
}

}

using namespace test::homography;
std::vector<Test> register_optim_homography_test() {
    return {
        TEST(test_homography_normal_acc),
        TEST(test_homography_jacobian),
        TEST(test_homography_refinement)
    };
}