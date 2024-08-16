#include "test.h"
#include "optim_test_utils.h"
#include <PoseLib/misc/camera_models.h>
#include <PoseLib/robust/jacobian_impl.h>
#include <PoseLib/robust/robust_loss.h>
#include <PoseLib/robust/optim/jacobian_accumulator.h>
#include <PoseLib/robust/optim/generalized_relative.h>
#include <PoseLib/robust/optim/lm_impl.h>


using namespace poselib;


//////////////////////////////
// Generalized Relative pose

namespace test::gen_relative {


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

void setup_scene(int Ncam1, int Ncam2, int N, CameraPose &pose,
                 std::vector<CameraPose> &cam1_ext,
                 std::vector<CameraPose> &cam2_ext,
                 std::vector<Camera> &cam1_int,
                 std::vector<Camera> &cam2_int,
                 std::vector<PairwiseMatches> &matches,
                 std::vector<std::vector<double>> &weights) {

    for(int i = 0; i < Ncam1; ++i)
        cam1_ext.push_back(random_camera());
    for(int i = 0; i < Ncam2; ++i)
        cam2_ext.push_back(random_camera());
    if(cam1_int.size() == 0)
        cam1_int.push_back(Camera("0 PINHOLE 1 1 1.0 1.0 0.0 0.0"));
    if(cam2_int.size() == 0)
        cam2_int.push_back(Camera("0 PINHOLE 1 1 1.0 1.0 0.0 0.0"));
    while(cam1_int.size() < Ncam1)
        cam1_int.push_back(cam1_int[0]);
    while(cam2_int.size() < Ncam2)
        cam2_int.push_back(cam2_int[0]);

    
    for(int cam_id1 = 0; cam_id1 < Ncam1; ++cam_id1) {
        for(int cam_id2 = 0; cam_id2 < Ncam2; ++cam_id2) {
            PairwiseMatches m;
            m.cam_id1 = cam_id1;
            m.cam_id2 = cam_id2;
            std::vector<double> w;

            for(size_t i = 0; i < N; ++i) {
                Eigen::Vector3d Xi;
                Xi.setRandom();

                Eigen::Vector2d xi;
                cam1_int[cam_id1].project(cam1_ext[cam_id1].apply(Xi), &xi);
                m.x1.push_back(xi);

                cam2_int[cam_id2].project(cam2_ext[cam_id2].apply(Xi), &xi);
                m.x2.push_back(xi);

                w.push_back(1.0 + i);
            }
            weights.push_back(w);
            matches.push_back(m);
        }
    }

    CameraPose p1 = cam1_ext[0];
    CameraPose p2 = cam2_ext[0];
    pose = p2.compose(p1.inverse());

    CameraPose p1_inv = cam1_ext[0].inverse();
    CameraPose p2_inv = cam2_ext[0].inverse();
    for(CameraPose &p : cam1_ext) {
        p = p.compose(p1_inv);
    }
    for(CameraPose &p : cam2_ext) {
        p = p.compose(p2_inv);
    }
}


bool test_gen_relative_pose_normal_acc() {
    const size_t Ncam1 = 4;
    const size_t Ncam2 = 4;
    const size_t N = 10;
  
    std::vector<CameraPose> cam1_ext;
    std::vector<CameraPose> cam2_ext;
    std::vector<Camera> cam1_int;
    std::vector<Camera> cam2_int;
    std::vector<std::vector<double>> weights;
    std::vector<PairwiseMatches> matches;
    CameraPose rel_pose;
    setup_scene(Ncam1, Ncam2, N, rel_pose, cam1_ext, cam2_ext, cam1_int, cam2_int, matches, weights);

    NormalAccumulator acc;
    GeneralizedPinholeRelativePoseRefiner refiner(matches, cam1_ext, cam2_ext);
    acc.initialize(refiner.num_params);

    // Check that residual is zero
    acc.reset_residual();
    refiner.compute_residual(acc, rel_pose);
    double residual = acc.get_residual();
    REQUIRE_SMALL(residual, 1e-6);

    // Check the gradient is zero
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, rel_pose);
    REQUIRE_SMALL(acc.Jtr.norm(), 1e-6);

    return true;
}



bool test_gen_relative_pose_jacobian() {
    const size_t Ncam1 = 4;
    const size_t Ncam2 = 4;
    const size_t N = 10;
  
    std::vector<CameraPose> cam1_ext;
    std::vector<CameraPose> cam2_ext;
    std::vector<Camera> cam1_int;
    std::vector<Camera> cam2_int;
    std::vector<std::vector<double>> weights;
    std::vector<PairwiseMatches> matches;
    CameraPose rel_pose;
    setup_scene(Ncam1, Ncam2, N, rel_pose, cam1_ext, cam2_ext, cam1_int, cam2_int, matches, weights);

    GeneralizedPinholeRelativePoseRefiner<UniformWeightVectors,TestAccumulator> refiner(matches, cam1_ext, cam2_ext);

    const double delta = 1e-6;
    double jac_err = verify_jacobian<decltype(refiner),CameraPose,6>(refiner, rel_pose, delta);
    REQUIRE_SMALL(jac_err, 1e-6)

    // Test that compute_residual and compute_jacobian are compatible
    TestAccumulator acc;
    acc.reset_residual();
    double r1 = refiner.compute_residual(acc, rel_pose);
    acc.reset_jacobian();
    refiner.compute_jacobian(acc, rel_pose);
    double r2 = 0.0;
    for(int i = 0; i < acc.rs.size(); ++i) {
        r2 += acc.weights[i] * acc.rs[i].squaredNorm();
    }
    REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

    return true;
}



bool test_gen_relative_pose_jacobian_varying_cams() {
    const size_t N = 10;
    for(int Ncam1 = 1; Ncam1 < 5; ++Ncam1) {
        for(int Ncam2 = 1; Ncam2 < 5; ++Ncam2) {
            if(Ncam1 == 1 && Ncam2 == 1) {
                continue; // Skip central case
            }
        
            std::vector<CameraPose> cam1_ext;
            std::vector<CameraPose> cam2_ext;
            std::vector<Camera> cam1_int;
            std::vector<Camera> cam2_int;
            std::vector<std::vector<double>> weights;
            std::vector<PairwiseMatches> matches;
            CameraPose rel_pose;
            setup_scene(Ncam1, Ncam2, N, rel_pose, cam1_ext, cam2_ext, cam1_int, cam2_int, matches, weights);

            for(PairwiseMatches &m : matches) {
                // Add some noise
                for(int i = 0; i < N; ++i) {
                    Eigen::Vector2d n;
                    n.setRandom();        
                    m.x1[i] += 0.001 * n;
                    n.setRandom();        
                    m.x2[i] += 0.001 * n;
                }
            }

            GeneralizedPinholeRelativePoseRefiner<UniformWeightVectors,TestAccumulator> refiner(matches, cam1_ext, cam2_ext);

            const double delta = 1e-6;
            double jac_err = verify_jacobian<decltype(refiner),CameraPose,6>(refiner, rel_pose, delta);
            REQUIRE_SMALL(jac_err, 1e-6)

            // Test that compute_residual and compute_jacobian are compatible
            TestAccumulator acc;
            acc.reset_residual();
            double r1 = refiner.compute_residual(acc, rel_pose);
            acc.reset_jacobian();
            refiner.compute_jacobian(acc, rel_pose);
            double r2 = 0.0;
            for(int i = 0; i < acc.rs.size(); ++i) {
                r2 += acc.weights[i] * acc.rs[i].squaredNorm();
            }
            REQUIRE_SMALL(std::abs(r1 - r2), 1e-10);

        }
    }
    return true;
}


bool test_gen_relative_pose_refinement() {
    const size_t Ncam1 = 4;
    const size_t Ncam2 = 3;
    const size_t N = 10;

    std::vector<CameraPose> cam1_ext;
    std::vector<CameraPose> cam2_ext;
    std::vector<Camera> cam1_int;
    std::vector<Camera> cam2_int;
    std::vector<std::vector<double>> weights;
    std::vector<PairwiseMatches> matches;
    CameraPose rel_pose;
    setup_scene(Ncam1, Ncam2, N, rel_pose, cam1_ext, cam2_ext, cam1_int, cam2_int, matches, weights);

    for(PairwiseMatches &m : matches) {
        // Add some noise
        for(int i = 0; i < N; ++i) {
            Eigen::Vector2d n;
            n.setRandom();        
            m.x1[i] += 0.01 * n;
            n.setRandom();        
            m.x2[i] += 0.01 * n;
        }
    }

    
    GeneralizedPinholeRelativePoseRefiner refiner(matches, cam1_ext, cam2_ext);
    
    BundleOptions bundle_opt;
    bundle_opt.step_tol = 1e-12;
    BundleStats stats = lm_impl(refiner, &rel_pose, bundle_opt, print_iteration);

    
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

using namespace test::gen_relative;
std::vector<Test> register_optim_gen_relative_test() {
    return {
        TEST(test_gen_relative_pose_normal_acc),
        TEST(test_gen_relative_pose_jacobian),
        TEST(test_gen_relative_pose_jacobian_varying_cams),
        TEST(test_gen_relative_pose_refinement)
    };
}