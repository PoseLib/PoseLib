#include "benchmark.h"

#include "problem_generator.h"

#include <chrono>
#include <iomanip>
#include <iostream>

namespace poselib {

template <typename Solver> BenchmarkResult benchmark(int n_problems, const ProblemOptions &options, double tol = 1e-6) {

    std::vector<AbsolutePoseProblemInstance> problem_instances;
    generate_abspose_problems(n_problems, &problem_instances, options);

    BenchmarkResult result;
    result.instances_ = n_problems;
    result.name_ = Solver::name();
    if (options.additional_name_ != "") {
        result.name_ += options.additional_name_;
    }
    result.options_ = options;
    std::cout << "Running benchmark: " << result.name_ << std::flush;

    // Run benchmark where we check solution quality
    for (const AbsolutePoseProblemInstance &instance : problem_instances) {
        CameraPoseVector solutions;

        int sols = Solver::solve(instance, &solutions);

        double pose_error = std::numeric_limits<double>::max();

        result.solutions_ += sols;
        // std::cout << "\nGt: " << instance.pose_gt.R() << "\n"<< instance.pose_gt.t << "\n";
        // std::cout << "gt valid = " << Solver::validator::is_valid(instance, instance.pose_gt, 1.0, tol) << "\n";
        for (const CameraPose &pose : solutions) {
            if (Solver::validator::is_valid(instance, pose, 1.0, tol))
                result.valid_solutions_++;
            // std::cout << "Pose: " << pose.R() << "\n" << pose.t << "\n";
            pose_error = std::min(pose_error, Solver::validator::compute_pose_error(instance, pose, 1.0));
        }
        if (pose_error < tol)
            result.found_gt_pose_++;
    }

    std::vector<long> runtimes;
    CameraPoseVector solutions;
    for (int iter = 0; iter < 10; ++iter) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (const AbsolutePoseProblemInstance &instance : problem_instances) {
            solutions.clear();
            Solver::solve(instance, &solutions);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    }

    std::sort(runtimes.begin(), runtimes.end());
    result.runtime_ns_ = runtimes[runtimes.size() / 2];
    std::cout << "\r                                                                                \r";
    return result;
}

template <typename Solver>
BenchmarkResult benchmark_w_extra(int n_problems, const ProblemOptions &options, double tol = 1e-6) {

    std::vector<AbsolutePoseProblemInstance> problem_instances;
    generate_abspose_problems(n_problems, &problem_instances, options);

    BenchmarkResult result;
    result.instances_ = n_problems;
    result.name_ = Solver::name();
    if (options.additional_name_ != "") {
        result.name_ += options.additional_name_;
    }
    result.options_ = options;
    std::cout << "Running benchmark: " << result.name_ << std::flush;

    // Run benchmark where we check solution quality
    for (const AbsolutePoseProblemInstance &instance : problem_instances) {
        CameraPoseVector solutions;
        std::vector<double> extra;

        int sols = Solver::solve(instance, &solutions, &extra);

        double pose_error = std::numeric_limits<double>::max();

        result.solutions_ += sols;
        for (size_t k = 0; k < solutions.size(); ++k) {
            if (Solver::validator::is_valid(instance, solutions[k], extra[k], tol))
                result.valid_solutions_++;
            pose_error = std::min(pose_error, Solver::validator::compute_pose_error(instance, solutions[k], extra[k]));
        }

        if (pose_error < tol)
            result.found_gt_pose_++;
    }

    std::vector<long> runtimes;
    CameraPoseVector solutions;
    std::vector<double> extra;
    for (int iter = 0; iter < 10; ++iter) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (const AbsolutePoseProblemInstance &instance : problem_instances) {
            solutions.clear();
            extra.clear();

            Solver::solve(instance, &solutions, &extra);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    }

    std::sort(runtimes.begin(), runtimes.end());
    result.runtime_ns_ = runtimes[runtimes.size() / 2];
    std::cout << "\r                                                                                \r";
    return result;
}

template <typename Solver>
BenchmarkResult benchmark_relative(int n_problems, const ProblemOptions &options, double tol = 1e-6) {

    std::vector<RelativePoseProblemInstance> problem_instances;
    generate_relpose_problems(n_problems, &problem_instances, options);

    BenchmarkResult result;
    result.instances_ = n_problems;
    result.name_ = Solver::name();
    if (options.additional_name_ != "") {
        result.name_ += options.additional_name_;
    }
    result.options_ = options;
    std::cout << "Running benchmark: " << result.name_ << std::flush;

    // Run benchmark where we check solution quality
    for (const RelativePoseProblemInstance &instance : problem_instances) {
        // CameraPoseVector solutions;
        std::vector<typename Solver::Solution> solutions;

        int sols = Solver::solve(instance, &solutions);

        double pose_error = std::numeric_limits<double>::max();

        result.solutions_ += sols;
        // std::cout << "Gt: " << instance.pose_gt.R << "\n"<< instance.pose_gt.t << "\n";
        for (const typename Solver::Solution &pose : solutions) {
            if (Solver::validator::is_valid(instance, pose, tol))
                result.valid_solutions_++;
            // std::cout << "Pose: " << pose.R << "\n" << pose.t << "\n";
            pose_error = std::min(pose_error, Solver::validator::compute_pose_error(instance, pose));
        }

        if (pose_error < tol)
            result.found_gt_pose_++;
    }

    std::vector<long> runtimes;
    std::vector<typename Solver::Solution> solutions;
    for (int iter = 0; iter < 10; ++iter) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (const RelativePoseProblemInstance &instance : problem_instances) {
            solutions.clear();

            Solver::solve(instance, &solutions);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    }

    std::sort(runtimes.begin(), runtimes.end());

    result.runtime_ns_ = runtimes[runtimes.size() / 2];
    std::cout << "\r                                                                                \r";
    return result;
}

template <typename Solver>
BenchmarkResult benchmark_homography(int n_problems, const ProblemOptions &options, double tol = 1e-6) {

    std::vector<RelativePoseProblemInstance> problem_instances;
    generate_homography_problems(n_problems, &problem_instances, options);

    BenchmarkResult result;
    result.instances_ = n_problems;
    result.name_ = Solver::name();
    if (options.additional_name_ != "") {
        result.name_ += options.additional_name_;
    }
    result.options_ = options;
    std::cout << "Running benchmark: " << result.name_ << std::flush;

    // Run benchmark where we check solution quality
    for (const RelativePoseProblemInstance &instance : problem_instances) {
        std::vector<Eigen::Matrix3d> solutions;

        int sols = Solver::solve(instance, &solutions);

        double hom_error = std::numeric_limits<double>::max();

        result.solutions_ += sols;
        // std::cout << "Gt: " << instance.pose_gt.R << "\n"<< instance.pose_gt.t << "\n";
        for (const Eigen::Matrix3d &H : solutions) {
            if (Solver::validator::is_valid(instance, H, tol))
                result.valid_solutions_++;
            // std::cout << "Pose: " << pose.R << "\n" << pose.t << "\n";
            hom_error = std::min(hom_error, Solver::validator::compute_pose_error(instance, H));
        }

        if (hom_error < tol)
            result.found_gt_pose_++;
    }

    std::vector<long> runtimes;
    std::vector<Eigen::Matrix3d> solutions;
    for (int iter = 0; iter < 10; ++iter) {
        auto start_time = std::chrono::high_resolution_clock::now();
        for (const RelativePoseProblemInstance &instance : problem_instances) {
            solutions.clear();

            Solver::solve(instance, &solutions);
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    }

    std::sort(runtimes.begin(), runtimes.end());

    result.runtime_ns_ = runtimes[runtimes.size() / 2];
    std::cout << "\r                                                                                \r";
    return result;
}

} // namespace poselib

void print_runtime(double runtime_ns) {
    if (runtime_ns < 1e3) {
        std::cout << runtime_ns << " ns";
    } else if (runtime_ns < 1e6) {
        std::cout << runtime_ns / 1e3 << " us";
    } else if (runtime_ns < 1e9) {
        std::cout << runtime_ns / 1e6 << " ms";
    } else {
        std::cout << runtime_ns / 1e9 << " s";
    }
}

void display_result(const std::vector<poselib::BenchmarkResult> &results) {
    // Print PoseLib version and buidling type
    std::cout << "\n" << poselib_info() << "\n\n";

    int w = 13;
    // display header
    std::cout << std::setw(2 * w) << "Solver";
    std::cout << std::setw(w) << "Solutions";
    std::cout << std::setw(w) << "Valid";
    std::cout << std::setw(w) << "GT found";
    std::cout << std::setw(w) << "Runtime\n";
    for (int i = 0; i < w * 6; ++i)
        std::cout << "-";
    std::cout << "\n";

    int prec = 6;

    for (const poselib::BenchmarkResult &result : results) {
        double num_tests = static_cast<double>(result.instances_);
        double solutions = result.solutions_ / num_tests;
        double valid_sols = result.valid_solutions_ / static_cast<double>(result.solutions_) * 100.0;
        double gt_found = result.found_gt_pose_ / num_tests * 100.0;
        double runtime_ns = result.runtime_ns_ / num_tests;

        std::cout << std::setprecision(prec) << std::setw(2 * w) << result.name_;
        std::cout << std::setprecision(prec) << std::setw(w) << solutions;
        std::cout << std::setprecision(prec) << std::setw(w) << valid_sols;
        std::cout << std::setprecision(prec) << std::setw(w) << gt_found;
        std::cout << std::setprecision(prec) << std::setw(w - 3);
        print_runtime(runtime_ns);
        std::cout << "\n";
    }
}

int main() {

    std::vector<poselib::BenchmarkResult> results;

    poselib::ProblemOptions options;
    // options.camera_fov_ = 45; // Narrow
    options.camera_fov_ = 75; // Medium
    // options.camera_fov_ = 120; // Wide

    double tol = 1e-6;

    // P3P
    poselib::ProblemOptions p3p_opt = options;
    p3p_opt.n_point_point_ = 3;
    p3p_opt.n_point_line_ = 0;
    results.push_back(poselib::benchmark<poselib::SolverP3P>(1e5, p3p_opt, tol));
    results.push_back(poselib::benchmark<poselib::SolverP3P_lambdatwist>(1e5, p3p_opt, tol));

    // gP3P
    poselib::ProblemOptions gp3p_opt = options;
    gp3p_opt.n_point_point_ = 3;
    gp3p_opt.n_point_line_ = 0;
    gp3p_opt.generalized_ = true;
    results.push_back(poselib::benchmark<poselib::SolverGP3P>(1e4, gp3p_opt, tol));

    // gP4Ps
    poselib::ProblemOptions gp4p_opt = options;
    gp4p_opt.n_point_point_ = 4;
    gp4p_opt.n_point_line_ = 0;
    gp4p_opt.generalized_ = true;
    gp4p_opt.unknown_scale_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverGP4PS>(1e4, gp4p_opt, tol));

    // gP4Ps Quasi-degenerate case (same 3D point observed twice)
    gp4p_opt.generalized_duplicate_obs_ = true;
    gp4p_opt.additional_name_ = "(Deg.)";
    results.push_back(poselib::benchmark_w_extra<poselib::SolverGP4PS>(1e4, gp4p_opt, tol));

    // P4Pf
    poselib::ProblemOptions p4pf_opt = options;
    p4pf_opt.n_point_point_ = 4;
    p4pf_opt.n_point_line_ = 0;
    p4pf_opt.unknown_focal_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP4PF>(1e4, p4pf_opt, tol));

    // P2P2PL
    poselib::ProblemOptions p2p2pl_opt = options;
    p2p2pl_opt.n_point_point_ = 2;
    p2p2pl_opt.n_point_line_ = 2;
    results.push_back(poselib::benchmark<poselib::SolverP2P2PL>(1e3, p2p2pl_opt, tol));

    // P6LP
    poselib::ProblemOptions p6lp_opt = options;
    p6lp_opt.n_line_point_ = 6;
    results.push_back(poselib::benchmark<poselib::SolverP6LP>(1e4, p6lp_opt, tol));

    // P5LP Radial
    poselib::ProblemOptions p5lp_radial_opt = options;
    p5lp_radial_opt.n_line_point_ = 5;
    p5lp_radial_opt.radial_lines_ = true;
    results.push_back(poselib::benchmark<poselib::SolverP5LP_Radial>(1e5, p5lp_radial_opt, tol));

    // P3P1LLf
    poselib::ProblemOptions p3p1llf_opt = options;
    p3p1llf_opt.n_point_point_ = 3;
    p3p1llf_opt.n_line_line_ = 1;
    p3p1llf_opt.unknown_focal_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP3P1LLF>(1e4, p3p1llf_opt, tol));

    // P2P2LLf
    poselib::ProblemOptions p2p2llf_opt = options;
    p2p2llf_opt.n_point_point_ = 2;
    p2p2llf_opt.n_line_line_ = 2;
    p2p2llf_opt.unknown_focal_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP2P2LLF>(1e4, p2p2llf_opt, tol));

    // P1P3LLf
    poselib::ProblemOptions p1p3llf_opt = options;
    p1p3llf_opt.n_point_point_ = 1;
    p1p3llf_opt.n_line_line_ = 3;
    p1p3llf_opt.unknown_focal_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP1P3LLF>(1e4, p1p3llf_opt, tol));

    // P4LLf
    poselib::ProblemOptions p4llf_opt = options;
    p4llf_opt.n_line_line_ = 4;
    p4llf_opt.unknown_focal_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverP4LLF>(1e4, p4llf_opt, tol));

    // P2P1LL
    poselib::ProblemOptions p2p1ll_opt = options;
    p2p1ll_opt.n_point_point_ = 2;
    p2p1ll_opt.n_line_line_ = 1;
    results.push_back(poselib::benchmark<poselib::SolverP2P1LL>(1e4, p2p1ll_opt, tol));

    // P1P2LL
    poselib::ProblemOptions p1p2ll_opt = options;
    p1p2ll_opt.n_point_point_ = 1;
    p1p2ll_opt.n_line_line_ = 2;
    results.push_back(poselib::benchmark<poselib::SolverP1P2LL>(1e4, p1p2ll_opt, tol));

    // P3LL
    poselib::ProblemOptions p3ll_opt = options;
    p3ll_opt.n_line_line_ = 3;
    results.push_back(poselib::benchmark<poselib::SolverP3LL>(1e4, p3ll_opt, tol));

    // uP2P
    poselib::ProblemOptions up2p_opt = options;
    up2p_opt.n_point_point_ = 2;
    up2p_opt.n_point_line_ = 0;
    up2p_opt.upright_ = true;
    results.push_back(poselib::benchmark<poselib::SolverUP2P>(1e6, up2p_opt, tol));

    // uP1P1LL
    poselib::ProblemOptions up1p1ll_opt = options;
    up1p1ll_opt.n_point_point_ = 1;
    up1p1ll_opt.n_point_line_ = 0;
    up1p1ll_opt.n_line_line_ = 1;
    up1p1ll_opt.upright_ = true;
    results.push_back(poselib::benchmark<poselib::SolverUP1P1LL>(1e6, up1p1ll_opt, tol));

    // uGP2P
    poselib::ProblemOptions ugp2p_opt = options;
    ugp2p_opt.n_point_point_ = 2;
    ugp2p_opt.n_point_line_ = 0;
    ugp2p_opt.upright_ = true;
    ugp2p_opt.generalized_ = true;
    results.push_back(poselib::benchmark<poselib::SolverUGP2P>(1e6, ugp2p_opt, tol));

    // uGP3Ps
    poselib::ProblemOptions ugp3ps_opt = options;
    ugp3ps_opt.n_point_point_ = 3;
    ugp3ps_opt.n_point_line_ = 0;
    ugp3ps_opt.upright_ = true;
    ugp3ps_opt.generalized_ = true;
    ugp3ps_opt.unknown_scale_ = true;
    results.push_back(poselib::benchmark_w_extra<poselib::SolverUGP3PS>(1e5, ugp3ps_opt, tol));

    // uP1P2PL
    poselib::ProblemOptions up1p2pl_opt = options;
    up1p2pl_opt.n_point_point_ = 1;
    up1p2pl_opt.n_point_line_ = 2;
    up1p2pl_opt.upright_ = true;
    results.push_back(poselib::benchmark<poselib::SolverUP1P2PL>(1e4, up1p2pl_opt, tol));

    // uP4PL
    poselib::ProblemOptions up4pl_opt = options;
    up4pl_opt.n_point_point_ = 0;
    up4pl_opt.n_point_line_ = 4;
    up4pl_opt.upright_ = true;
    results.push_back(poselib::benchmark<poselib::SolverUP4PL>(1e4, up4pl_opt, tol));

    // ugP4PL
    poselib::ProblemOptions ugp4pl_opt = options;
    ugp4pl_opt.n_point_point_ = 0;
    ugp4pl_opt.n_point_line_ = 4;
    ugp4pl_opt.upright_ = true;
    ugp4pl_opt.generalized_ = true;
    results.push_back(poselib::benchmark<poselib::SolverUGP4PL>(1e4, ugp4pl_opt, tol));

    // Relative Pose Upright
    poselib::ProblemOptions relupright3pt_opt = options;
    relupright3pt_opt.n_point_point_ = 3;
    relupright3pt_opt.upright_ = true;
    results.push_back(poselib::benchmark_relative<poselib::SolverRelUpright3pt>(1e4, relupright3pt_opt, tol));

    // Generalized Relative Pose Upright
    poselib::ProblemOptions genrelupright4pt_opt = options;
    genrelupright4pt_opt.n_point_point_ = 4;
    genrelupright4pt_opt.upright_ = true;
    genrelupright4pt_opt.generalized_ = true;
    results.push_back(poselib::benchmark_relative<poselib::SolverGenRelUpright4pt>(1e4, genrelupright4pt_opt, tol));

    // Relative Pose 8pt
    poselib::ProblemOptions rel8pt_opt = options;
    rel8pt_opt.n_point_point_ = 8;
    results.push_back(poselib::benchmark_relative<poselib::SolverRel8pt>(1e4, rel8pt_opt, tol));

    rel8pt_opt.additional_name_ = "(100 pts)";
    rel8pt_opt.n_point_point_ = 100;
    results.push_back(poselib::benchmark_relative<poselib::SolverRel8pt>(1e4, rel8pt_opt, tol));

    // Relative Pose 5pt
    poselib::ProblemOptions rel5pt_opt = options;
    rel5pt_opt.n_point_point_ = 5;
    results.push_back(poselib::benchmark_relative<poselib::SolverRel5pt>(1e4, rel5pt_opt, tol));

    // Monodepth relative Pose 3pt
    poselib::ProblemOptions monodepth_rel3pt_opt = options;
    monodepth_rel3pt_opt.n_point_point_ = 3;
    monodepth_rel3pt_opt.use_monodepth_ = true;
    results.push_back(poselib::benchmark_relative<poselib::SolverMonodepthRel3pt>(1e4, monodepth_rel3pt_opt, tol));

    // Relative Pose With Single Unknown Focal 6pt
    poselib::ProblemOptions rel_focal_6pt_opt = options;
    rel_focal_6pt_opt.n_point_point_ = 6;
    rel_focal_6pt_opt.min_focal_ = 0.1;
    rel_focal_6pt_opt.max_focal_ = 5.0;
    rel_focal_6pt_opt.unknown_focal_ = true;
    results.push_back(poselib::benchmark_relative<poselib::SolverSharedFocalRel6pt>(1e4, rel_focal_6pt_opt, tol));

    // Relative Pose With Single Unknown Focal 3pt Using Monodepth
    poselib::ProblemOptions rel_monodepth_shared_focal_3pt_opt = options;
    rel_monodepth_shared_focal_3pt_opt.n_point_point_ = 3;
    rel_monodepth_shared_focal_3pt_opt.min_focal_ = 0.1;
    rel_monodepth_shared_focal_3pt_opt.max_focal_ = 5.0;
    rel_monodepth_shared_focal_3pt_opt.unknown_focal_ = true;
    rel_monodepth_shared_focal_3pt_opt.use_monodepth_ = true;
    results.push_back(poselib::benchmark_relative<poselib::SolverMonodepthSharedFocalRel3pt>(
        1e4, rel_monodepth_shared_focal_3pt_opt, tol));

    // Relative Pose With Different Unknown Focals 3pt Using Monodepth
    poselib::ProblemOptions rel_monodepth_varying_focal_3pt_opt = options;
    rel_monodepth_varying_focal_3pt_opt.n_point_point_ = 3;
    rel_monodepth_varying_focal_3pt_opt.min_focal_ = 0.1;
    rel_monodepth_varying_focal_3pt_opt.max_focal_ = 5.0;
    rel_monodepth_varying_focal_3pt_opt.unknown_focal_ = true;
    rel_monodepth_varying_focal_3pt_opt.varying_focal_ = true;
    rel_monodepth_varying_focal_3pt_opt.use_monodepth_ = true;
    results.push_back(poselib::benchmark_relative<poselib::SolverMonodepthVaryingFocalRel3pt>(
        1e4, rel_monodepth_varying_focal_3pt_opt, tol));

    // Relative Pose Upright Planar 2pt
    poselib::ProblemOptions reluprightplanar2pt_opt = options;
    reluprightplanar2pt_opt.n_point_point_ = 2;
    reluprightplanar2pt_opt.upright_ = true;
    reluprightplanar2pt_opt.planar_ = true;
    results.push_back(
        poselib::benchmark_relative<poselib::SolverRelUprightPlanar2pt>(1e4, reluprightplanar2pt_opt, tol));

    // Relative Pose Upright Planar 3pt
    poselib::ProblemOptions reluprightplanar3pt_opt = options;
    reluprightplanar3pt_opt.n_point_point_ = 3;
    reluprightplanar3pt_opt.upright_ = true;
    reluprightplanar3pt_opt.planar_ = true;
    results.push_back(
        poselib::benchmark_relative<poselib::SolverRelUprightPlanar3pt>(1e4, reluprightplanar3pt_opt, tol));

    // Generalized Relative Pose (5+1pt)
    poselib::ProblemOptions genrel5p1pt_opt = options;
    genrel5p1pt_opt.n_point_point_ = 6;
    genrel5p1pt_opt.generalized_ = true;
    genrel5p1pt_opt.generalized_first_cam_obs_ = 5;
    results.push_back(poselib::benchmark_relative<poselib::SolverGenRel5p1pt>(1e4, genrel5p1pt_opt, tol));

    // Generalized Relative Pose (6pt)
    poselib::ProblemOptions genrel6pt_opt = options;
    genrel6pt_opt.n_point_point_ = 6;
    genrel6pt_opt.generalized_ = true;
    results.push_back(poselib::benchmark_relative<poselib::SolverGenRel6pt>(1e3, genrel6pt_opt, tol));

    // Homograpy (4pt)
    poselib::ProblemOptions homo4pt_opt = options;
    homo4pt_opt.n_point_point_ = 4;
    results.push_back(poselib::benchmark_homography<poselib::SolverHomography4pt<false>>(1e5, homo4pt_opt, tol));
    results.push_back(poselib::benchmark_homography<poselib::SolverHomography4pt<true>>(1e5, homo4pt_opt, tol));

    display_result(results);

    return 0;
}
