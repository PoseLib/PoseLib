#include "test.h"
#include "optim_test_utils.h"
#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/jacobian_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/absolute.h>

using namespace poselib;

//////////////////////////////
// Absolute pose

bool test_absolute_pose_normal_acc() {
    CameraPose pose;
    pose.q.setRandom();
    pose.q.normalize();
    pose.t.setRandom();

    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d xi;
        xi.setRandom(); // 90 deg fov
        Eigen::Vector3d Xi;
        camera.unproject(xi,&Xi);
        Xi *= (2.0 + rand()); // backproject
        x.push_back(xi);
        X.push_back(pose.apply_inverse(Xi));
    }

    NormalAccumulator<TrivialLoss> acc(6, 2);
    AbsolutePoseRefiner<decltype(acc)> refiner(x,X,camera);

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

bool test_absolute_pose_jacobian() {
    CameraPose pose;
    pose.q.setRandom();
    pose.q.normalize();
    pose.t.setRandom();

    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d xi;
        xi.setRandom(); // 90 deg fov
        Eigen::Vector3d Xi;
        camera.unproject(xi,&Xi);
        Xi *= (2.0 + rand()); // backproject
        // add a small amount of noice
        Eigen::Vector2d noise;
        noise.setRandom();
        x.push_back(xi + 0.001 * noise);
        X.push_back(pose.apply_inverse(Xi));
    }

    TestAccumulator acc;
    AbsolutePoseRefiner<decltype(acc)> refiner(x,X,camera);

    const double delta = 1e-6;
    double jac_err = test_refiner<decltype(refiner),CameraPose,6>(refiner, pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    return true;
}


std::vector<Test> register_optim_absolute_test() {
    return {
        TEST(test_absolute_pose_normal_acc),
        TEST(test_absolute_pose_jacobian)
    };
}