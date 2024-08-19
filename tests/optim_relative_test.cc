#include "test.h"
#include "optim_test_utils.h"
#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/jacobian_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/relative.h>
#include <PoseLib/robust/optim/lm_impl.h>


using namespace poselib;


//////////////////////////////
// Relative pose

namespace test::relative {


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

void setup_scene(int N, CameraPose &pose, std::vector<Point2D> &x1,
                 std::vector<Point2D> &x2, Camera &cam1, Camera &cam2) {

    CameraPose p1 = random_camera();
    CameraPose p2 = random_camera();
        
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector3d Xi;
        Xi.setRandom();

        Eigen::Vector2d xi;
        cam1.project(p1.apply(Xi), &xi);
        x1.push_back(xi);

        cam2.project(p2.apply(Xi), &xi);
        x2.push_back(xi);
    }

    Eigen::Matrix3d R = p2.R() * p1.R().transpose();
    Eigen::Vector3d t = p2.t - p2.R() * p1.R().transpose() * p1.t;
    
    pose = CameraPose(R,t);
}


bool test_relative_pose_normal_acc() {
    
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    NormalAccumulator acc;
    PinholeRelativePoseRefiner refiner(x1,x2);
    acc.initialize(refiner.num_params);

    // Check that residual is zero
    acc.reset_residual();
    refiner.compute_residual(acc, pose);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    // Check the gradient is zero
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}


bool test_relative_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    PinholeRelativePoseRefiner<UniformWeightVector, TestAccumulator> refiner(x1,x2);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner),CameraPose>(refiner, pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, pose);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, pose);
    double r2 = 0.0;
    for(int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}


bool test_relative_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);

    CameraPose pose;
    std::vector<Eigen::Vector2d> x1, x2;
    setup_scene(N, pose, x1, x2, camera, camera);

    // Add some noise
    for(int i = 0; i < N; ++i) {
        Eigen::Vector2d n;
        n.setRandom();        
        x1[i] += 0.001 * n;
        n.setRandom();        
        x2[i] += 0.001 * n;
    }
    
    PinholeRelativePoseRefiner refiner(x1,x2);
    
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &pose, bundle_opt, print_iteration);

    
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

using namespace test::relative;
std::vector<Test> register_optim_relative_test() {
    return {
        TEST(test_relative_pose_normal_acc),
        TEST(test_relative_pose_jacobian),
        TEST(test_relative_pose_refinement)
    };
}