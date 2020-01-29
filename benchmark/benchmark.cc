#include "benchmark.h"
#include "problem_generator.h"
#include <iostream>
#include <chrono>
#include <iomanip>
namespace pose_lib {



template<typename Solver>
BenchmarkResult benchmark(int n_problems, const ProblemOptions &options, double tol = 1e-6) {
    
    std::vector<ProblemInstance> problem_instances;
    generate_problems(n_problems, &problem_instances, options);


    BenchmarkResult result;
    result.instances_ = n_problems;
    result.name_ = Solver::name();
    result.options_ = options;
    
    // Run benchmark where we check solution quality
    for(const ProblemInstance &instance : problem_instances) {
        CameraPoseVector solutions;
        
        int sols = Solver::solve(instance, &solutions);

        double pose_error = std::numeric_limits<double>::max();

        result.solutions_ += sols;

        for(const CameraPose &pose : solutions) {
            if(instance.is_valid(pose, tol))
                result.valid_solutions_++;
            

            pose_error = std::min(pose_error, instance.compute_pose_error(pose));            
        }

        if(pose_error < tol)
            result.found_gt_pose_++;
    }


    std::vector<long> runtimes;
    CameraPoseVector solutions;
    for(int iter = 0; iter < 10; ++iter) {
        int total_sols = 0;
        auto start_time = std::chrono::high_resolution_clock::now();
        for(const ProblemInstance &instance : problem_instances) {
                solutions.clear();
                
                int sols = Solver::solve(instance, &solutions);

                total_sols += sols;            
            }

        auto end_time = std::chrono::high_resolution_clock::now();
        runtimes.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count());
    }

    std::sort(runtimes.begin(), runtimes.end());

    result.runtime_ns_ = runtimes[runtimes.size() / 2];
    return result;
}



} // namespace


void print_runtime(double runtime_ns) {
    if(runtime_ns < 1e3) {
        std::cout << runtime_ns << " ns";
    } else if(runtime_ns < 1e6) {
        std::cout << runtime_ns / 1e3 << " us";
    } else if(runtime_ns < 1e9) {
        std::cout << runtime_ns / 1e6 << " ms";
    } else {
        std::cout << runtime_ns / 1e9 << " s";
    }


}

void display_result(std::vector<pose_lib::BenchmarkResult> &results) {

    int w = 13;
    // display header
    std::cout << std::setw(w) << "Solver";
    std::cout << std::setw(w) << "Solutions";
    std::cout << std::setw(w) << "Valid";
    std::cout << std::setw(w) << "GT found";
    std::cout << std::setw(w) << "Runtime" << "\n";
    for (int i = 0; i < w * 5; ++i)
        std::cout << "-";
    std::cout << "\n";


    int prec = 6;

    for(pose_lib::BenchmarkResult &result : results) {
        double num_tests = static_cast<double>(result.instances_);
        double solutions = result.solutions_ / num_tests;
        double valid_sols = result.valid_solutions_ / static_cast<double>(result.solutions_) * 100.0;
        double gt_found = result.found_gt_pose_ / num_tests * 100.0;
        double runtime_ns = result.runtime_ns_ / num_tests;

        std::cout << std::setprecision(prec) << std::setw(w) << result.name_;
        std::cout << std::setprecision(prec) << std::setw(w) << solutions;
        std::cout << std::setprecision(prec) << std::setw(w) << valid_sols;
        std::cout << std::setprecision(prec) << std::setw(w) << gt_found;
        std::cout << std::setprecision(prec) << std::setw(w-3);
        print_runtime(runtime_ns);
        std::cout << "\n";   
    }

}

int main() {

    std::vector<pose_lib::BenchmarkResult> results;

    pose_lib::ProblemOptions options;
    // options.camera_fov_ = 45; // Narrow
    // options.camera_fov_ = 75; // Medium
    options.camera_fov_ = 120; // Wide

    double tol = 1e-6;

    
    // P3P
    pose_lib::ProblemOptions p3p_opt = options;
    p3p_opt.n_point_point_ = 3; p3p_opt.n_point_line_ = 0;
    results.push_back( pose_lib::benchmark<pose_lib::SolverP3P>(1e5, p3p_opt, tol) );

    // gP3P
    pose_lib::ProblemOptions gp3p_opt = options;
    gp3p_opt.n_point_point_ = 3; gp3p_opt.n_point_line_ = 0;
    gp3p_opt.generalized_ = true;
    results.push_back( pose_lib::benchmark<pose_lib::SolverGP3P>(1e4, gp3p_opt, tol) );

    // gP4Ps
    pose_lib::ProblemOptions gp4p_opt = options;
    gp4p_opt.n_point_point_ = 4; gp4p_opt.n_point_line_ = 0;
    gp4p_opt.generalized_ = true;
    gp4p_opt.unknown_scale_ = true;
    results.push_back( pose_lib::benchmark<pose_lib::SolverGP4PS>(1e4, gp4p_opt, tol) );


    // P2P2L
    pose_lib::ProblemOptions p2p2l_opt = options;
    p2p2l_opt.n_point_point_ = 2; p2p2l_opt.n_point_line_ = 2;
    results.push_back( pose_lib::benchmark<pose_lib::SolverP2P2L>(1e3, p2p2l_opt, tol) );

    // uP2P
    pose_lib::ProblemOptions up2p_opt = options;
    up2p_opt.n_point_point_ = 2; up2p_opt.n_point_line_ = 0;
    up2p_opt.upright_ = true;
    results.push_back( pose_lib::benchmark<pose_lib::SolverUP2P>(1e6, up2p_opt, tol) );

    // uP1P2L
    pose_lib::ProblemOptions up1p2l_opt = options;
    up1p2l_opt.n_point_point_ = 1; up1p2l_opt.n_point_line_ = 2;
    up1p2l_opt.upright_ = true;
    results.push_back( pose_lib::benchmark<pose_lib::SolverUP1P2L>(1e5, up1p2l_opt, tol) );


    // uP4L
    pose_lib::ProblemOptions up4l_opt = options;
    up4l_opt.n_point_point_ = 0; up4l_opt.n_point_line_ = 4;
    up4l_opt.upright_ = true;
    results.push_back( pose_lib::benchmark<pose_lib::SolverUP4L>(1e3, up4l_opt, tol) );
    
    
    display_result(results);

    return 0;    
}