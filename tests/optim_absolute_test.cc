#include "test.h"
#include "optim_test_utils.h"
#include "example_cameras.h"

#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/jacobian_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/absolute.h>
#include <PoseLib/robust/optim/hybrid.h>
#include <PoseLib/robust/optim/lm_impl.h>


using namespace poselib;


//////////////////////////////
// Absolute pose

namespace test::absolute {


void setup_scene(int N, CameraPose &pose, std::vector<Point2D> &x,
                 std::vector<Point3D> &X, Camera &cam, std::vector<double> &weights) {

    pose.q.setRandom();
    pose.q.normalize();
    pose.t.setRandom();
    Eigen::VectorXd depth_factor(N);
    depth_factor.setRandom();
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d xi;
        // we sample points in [0.2, 0.8] of the image
        xi.setRandom();
        xi *= 0.3;
        xi += Eigen::Vector2d(0.5, 0.5);        
        // xi = [-1, 1] -> xi = [0.2, 0.8]
        xi << xi(0) * cam.width, xi(1) * cam.height;

        Eigen::Vector3d Xi;
        cam.unproject(xi,&Xi);
        Xi *= (2.0 + 10.0 * depth_factor(i)); // backproject
        x.push_back(xi);
        X.push_back(pose.apply_inverse(Xi));
        weights.push_back(1.0 * (i + 1.0));
    }
}


void setup_scene_w_lines(int N_pts, int N_lines, CameraPose &pose,
                std::vector<Point2D> &x, std::vector<Point3D> &X,
                std::vector<Line2D> &lin2D, std::vector<Line3D> &lin3D, 
                Camera &cam, std::vector<double> &weights_pts, std::vector<double> &weights_lin) {

    std::vector<Point2D> x_all;
    std::vector<Point3D> X_all;
    std::vector<double> w_all;
    setup_scene(N_pts + 2 * N_lines, pose, x_all, X_all, cam, w_all);

    for(int i = 0; i < N_pts; ++i) {
        x.push_back(x_all[i]);
        X.push_back(X_all[i]);
        weights_pts.push_back(w_all[i]);
    }

    for(int i = N_pts; i < N_pts + 2*N_lines; i+=2) {
        Line2D l2D;
        Line3D l3D;
        l2D.x1 = x_all[i];
        l2D.x2 = x_all[i+1];
        l3D.X1 = X_all[i];
        l3D.X2 = X_all[i+1];
        lin2D.push_back(l2D);
        lin3D.push_back(l3D);
        weights_lin.push_back(w_all[i]);
    }
}

////////////////////////////
// Point (2D-3D correspondence)

bool test_absolute_pose_normal_acc() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<double> weights;
    setup_scene(N, pose, x, X, camera, weights);


    NormalAccumulator<6, TrivialLoss> acc;
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
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<double> weights;
    setup_scene(N, pose, x, X, camera, weights);

    // add noise    
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d noise;
        noise.setRandom();
        x[i] += 0.01 * noise;        
    }

    AbsolutePoseRefiner<TestAccumulator,std::vector<double>> refiner(x,X,camera,weights);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner),CameraPose,6>(refiner, pose, delta);
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


bool test_absolute_pose_jacobian_cameras() {
    const size_t N = 10;
    for(std::string camera_str : example_cameras) {
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;    
        std::vector<Eigen::Vector2d> x;
        std::vector<Eigen::Vector3d> X;    
        std::vector<double> weights;
        setup_scene(N, pose, x, X, camera, weights);

        // Rescale points
        double f = camera.focal();
        for(int i = 0; i < N; ++i) {
            x[i] /= f;
        }
        camera.rescale(1.0 / f);

        AbsolutePoseRefiner<TestAccumulator,std::vector<double>> refiner(x,X,camera,weights);

        const double delta = 1e-6;
        double jac_err = verify_jacobian<decltype(refiner),CameraPose,6>(refiner, pose, delta);
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

    }
    return true;
}


bool test_absolute_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<double> weights;
    setup_scene(N, pose, x, X, camera, weights);

    // add noise    
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d noise;
        noise.setRandom();
        x[i] += 0.01 * noise;        
    }

    NormalAccumulator<6> acc;
    AbsolutePoseRefiner<decltype(acc)> refiner(x,X,camera);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, acc, &pose, bundle_opt, print_iteration);

    
    std::cout << "iter = " << stats.iterations << "\n";
    std::cout << "initial_cost = " << stats.initial_cost << "\n";
    std::cout << "cost = " << stats.cost << "\n";
    std::cout << "lambda = " << stats.lambda << "\n";
    std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    std::cout << "step_norm = " << stats.step_norm << "\n";
    std::cout << "grad_norm = " << stats.grad_norm << "\n";
    

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);
    
    return true;
}


bool test_absolute_pose_weighted_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<double> weights;
    setup_scene(N, pose, x, X, camera, weights);

    // add noise    
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d noise;
        noise.setRandom();
        x[i] += 0.001 * noise;        
    }

    NormalAccumulator<6> acc;
    AbsolutePoseRefiner<decltype(acc),std::vector<double>> refiner(x,X,camera, weights);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, acc, &pose, bundle_opt, print_iteration);

    /*
    std::cout << "iter = " << stats.iterations << "\n";
    std::cout << "initial_cost = " << stats.initial_cost << "\n";
    std::cout << "cost = " << stats.cost << "\n";
    std::cout << "lambda = " << stats.lambda << "\n";
    std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    std::cout << "step_norm = " << stats.step_norm << "\n";
    std::cout << "grad_norm = " << stats.grad_norm << "\n";
    */

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);
    
    return true;
}



bool test_absolute_pose_cameras_refinement() {
    const size_t N = 25;
    for(std::string camera_str : example_cameras) {
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;    
        std::vector<Eigen::Vector2d> x;
        std::vector<Eigen::Vector3d> X;    
        std::vector<double> weights;
        setup_scene(N, pose, x, X, camera, weights);

        // add noise    
        for(size_t i = 0; i < N; ++i) {
            Eigen::Vector2d noise;
            noise.setRandom();
            x[i] += 0.001 * std::max(camera.width, camera.height) * noise;
        }

        // Rescale points
        double f = camera.focal();
        for(int i = 0; i < N; ++i) {
            x[i] /= f;
        }
        camera.rescale(1.0 / f);

        NormalAccumulator<6> acc;
        AbsolutePoseRefiner<decltype(acc)> refiner(x,X,camera);

        BundleOptions bundle_opt;
        bundle_opt.step_tol = 1e-12;
        BundleStats stats = lm_impl(refiner, acc, &pose, bundle_opt, print_iteration);

        REQUIRE_SMALL(stats.grad_norm, 1e-8);
        REQUIRE(stats.cost < stats.initial_cost);
    }
    
    return true;
}

/////////////////////////////
// Line (2D-3D)

bool test_line_absolute_pose_normal_acc() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D; 
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);


    NormalAccumulator<6,TrivialLoss> acc;
    PinholeLineAbsolutePoseRefiner<decltype(acc)> refiner(lin2D,lin3D);

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

bool test_line_absolute_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D; 
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    // add noise    
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d noise;
        noise.setRandom();
        x[i] += 0.01 * noise;
        noise.setRandom();
        lin2D[i].x2 += 0.01 * noise;
        noise.setRandom();
        lin2D[i].x2 += 0.01 * noise;
    }

    PinholeLineAbsolutePoseRefiner<TestAccumulator,std::vector<double>> refiner(lin2D, lin3D, w_lin);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner),CameraPose,6>(refiner, pose, delta);
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


bool test_line_absolute_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D; 
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    // add noise    
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d noise;
        noise.setRandom();
        x[i] += 0.01 * noise;
        noise.setRandom();
        lin2D[i].x2 += 0.01 * noise;
        noise.setRandom();
        lin2D[i].x2 += 0.01 * noise;
    }

    NormalAccumulator<6> acc;
    PinholeLineAbsolutePoseRefiner<decltype(acc),decltype(w_lin)> refiner(lin2D, lin3D, w_lin);

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, acc, &pose, bundle_opt, print_iteration);

    
    std::cout << "iter = " << stats.iterations << "\n";
    std::cout << "initial_cost = " << stats.initial_cost << "\n";
    std::cout << "cost = " << stats.cost << "\n";
    std::cout << "lambda = " << stats.lambda << "\n";
    std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    std::cout << "step_norm = " << stats.step_norm << "\n";
    std::cout << "grad_norm = " << stats.grad_norm << "\n";
    

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);
    
    return true;
}


////////////////////////////////////////////////
// Point + Line 

bool test_point_line_absolute_pose_jacobian() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D; 
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    // add noise    
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d noise;
        noise.setRandom();
        x[i] += 0.01 * noise;
        noise.setRandom();
        lin2D[i].x2 += 0.01 * noise;
        noise.setRandom();
        lin2D[i].x2 += 0.01 * noise;
    }

    AbsolutePoseRefiner<TestAccumulator,decltype(w_lin)> pts_refiner(x, X, camera, w_pts);
    PinholeLineAbsolutePoseRefiner<TestAccumulator,decltype(w_lin)> lin_refiner(lin2D, lin3D, w_lin);
    HybridRefiner<TestAccumulator> refiner;
    refiner.register_refiner(&pts_refiner);
    refiner.register_refiner(&lin_refiner);
    
    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner),CameraPose,6>(refiner, pose, delta);
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


bool test_point_line_absolute_pose_refinement() {
    const size_t N = 10;
    std::string camera_str = "0 PINHOLE 1 1 1.0 1.0 0.0 0.0";
    Camera camera;
    camera.initialize_from_txt(camera_str);
    CameraPose pose;    
    std::vector<Eigen::Vector2d> x;
    std::vector<Eigen::Vector3d> X;    
    std::vector<Line2D> lin2D;
    std::vector<Line3D> lin3D; 
    std::vector<double> w_pts, w_lin;
    setup_scene_w_lines(N, N, pose, x, X, lin2D, lin3D, camera, w_pts, w_lin);

    // add noise    
    for(size_t i = 0; i < N; ++i) {
        Eigen::Vector2d noise;
        noise.setRandom();
        x[i] += 0.01 * noise;
        noise.setRandom();
        lin2D[i].x2 += 0.01 * noise;
        noise.setRandom();
        lin2D[i].x2 += 0.01 * noise;
    }

    NormalAccumulator<6> acc;
    AbsolutePoseRefiner<decltype(acc),decltype(w_lin)> pts_refiner(x, X, camera, w_pts);
    PinholeLineAbsolutePoseRefiner<decltype(acc),decltype(w_lin)> lin_refiner(lin2D, lin3D, w_lin);
    HybridRefiner<decltype(acc)> refiner;
    refiner.register_refiner(&pts_refiner);
    refiner.register_refiner(&lin_refiner);
    

    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, acc, &pose, bundle_opt, print_iteration);

    
    std::cout << "iter = " << stats.iterations << "\n";
    std::cout << "initial_cost = " << stats.initial_cost << "\n";
    std::cout << "cost = " << stats.cost << "\n";
    std::cout << "lambda = " << stats.lambda << "\n";
    std::cout << "invalid_steps = " << stats.invalid_steps << "\n";
    std::cout << "step_norm = " << stats.step_norm << "\n";
    std::cout << "grad_norm = " << stats.grad_norm << "\n";
    

    REQUIRE_SMALL(stats.grad_norm, 1e-8);
    REQUIRE(stats.cost < stats.initial_cost);
    
    return true;
}



////////////////////////////////////////////////
// 1D Radial Camera model


bool test_1d_radial_absolute_pose_jacobian_cameras() {
    const size_t N = 25;
    for(std::string camera_str : radially_symmetric_example_cameras) {
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;    
        std::vector<Eigen::Vector2d> x;
        std::vector<Eigen::Vector3d> X;    
        std::vector<double> weights;
        setup_scene(N, pose, x, X, camera, weights);

        NormalAccumulator<5,TrivialLoss> normal_acc(5);
        Radial1DAbsolutePoseRefiner<decltype(normal_acc),std::vector<double>> norm_refiner(x,X,camera,weights);
        // Check that residual is zero
        normal_acc.reset_residual();
        norm_refiner.compute_residual(normal_acc, pose);
        double residual = normal_acc.get_residual();
        REQUIRE_SMALL(residual, 1e-6);

         // add noise    
        for(size_t i = 0; i < N; ++i) {
            Eigen::Vector2d noise;
            noise.setRandom();
            x[i] += 0.01 * noise;
        }

        std::cout << camera_str << std::endl;

        Radial1DAbsolutePoseRefiner<TestAccumulator,std::vector<double>> refiner(x,X,camera,weights);
        const double delta = 1e-8;
        double jac_err = verify_jacobian<decltype(refiner),CameraPose,5>(refiner, pose, delta);
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

    }
    return true;
}


bool test_1d_radial_absolute_pose_cameras_refinement() {
    const size_t N = 25;
    for(std::string camera_str : radially_symmetric_example_cameras) {
        Camera camera;
        camera.initialize_from_txt(camera_str);
        CameraPose pose;    
        std::vector<Eigen::Vector2d> x;
        std::vector<Eigen::Vector3d> X;    
        std::vector<double> weights;
        setup_scene(N, pose, x, X, camera, weights);

        // add noise    
        for(size_t i = 0; i < N; ++i) {
            Eigen::Vector2d noise;
            noise.setRandom();
            x[i] += 0.001 * std::max(camera.width, camera.height) * noise;
        }

        // Rescale points
        double f = camera.focal();
        for(int i = 0; i < N; ++i) {
            x[i] /= f;
        }
        camera.rescale(1.0 / f);

        NormalAccumulator<5> acc(5);
        Radial1DAbsolutePoseRefiner<decltype(acc)> refiner(x,X,camera);

        BundleOptions bundle_opt;
        bundle_opt.step_tol = 1e-12;
        BundleStats stats = lm_impl(refiner, acc, &pose, bundle_opt, print_iteration);

        REQUIRE_SMALL(stats.grad_norm, 1e-8);
        REQUIRE(stats.cost < stats.initial_cost);
    }
    
    return true;
}

}

using namespace test::absolute;
std::vector<Test> register_optim_absolute_test() {
    return {
        // Points
        TEST(test_absolute_pose_normal_acc),
        TEST(test_absolute_pose_jacobian),
        TEST(test_absolute_pose_jacobian_cameras),
        TEST(test_absolute_pose_refinement),
        TEST(test_absolute_pose_weighted_refinement),
        TEST(test_absolute_pose_cameras_refinement),
        // Lines
        TEST(test_line_absolute_pose_normal_acc),
        TEST(test_line_absolute_pose_jacobian),
        TEST(test_line_absolute_pose_refinement),
        // Poiints + Lines
        TEST(test_point_line_absolute_pose_jacobian),
        TEST(test_point_line_absolute_pose_refinement),
        // 1D radial camera
        TEST(test_1d_radial_absolute_pose_jacobian_cameras),
        TEST(test_1d_radial_absolute_pose_cameras_refinement)
    };
}
