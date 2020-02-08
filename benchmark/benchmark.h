#pragma once

#include <Eigen/Dense>
#include <vector>
#include <PoseLib/types.h>
#include <stdint.h>
#include <string>
#include "problem_generator.h"

#include <PoseLib/p3p.h>
#include <PoseLib/p4pf.h>
#include <PoseLib/gp3p.h>
#include <PoseLib/gp4ps.h>
#include <PoseLib/up2p.h>
#include <PoseLib/up1p2l.h>
#include <PoseLib/p2p2l.h>
#include <PoseLib/up4l.h>

namespace pose_lib {

    struct BenchmarkResult {
        std::string name_;
        ProblemOptions options_;
        int instances_ = 0;
        int solutions_ = 0;
        int valid_solutions_ = 0;
        int found_gt_pose_ = 0;
        int runtime_ns_ = 0;
    };


    // Wrappers for the Benchmarking code

    struct SolverP3P {
        static inline int solve(const ProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
            return p3p(instance.x_point_, instance.X_point_, solutions);
        }
		typedef CalibPoseValidator validator;
        static std::string name() { return "p3p"; }
    };

	struct SolverP4Pf {
		static inline int solve(const ProblemInstance& instance, pose_lib::CameraPoseVector* solutions) {
			return p4pf(instance.x_point_, instance.X_point_, solutions);
		}
		typedef UnknownFocalValidator validator;
		static std::string name() { return "p4pf"; }
	};



    struct SolverGP3P {
        static inline int solve(const ProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
            return gp3p(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
        }
		typedef CalibPoseValidator validator;
        static std::string name() { return "gp3p"; }
    };

    struct SolverGP4PS {
        static inline int solve(const ProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
            return gp4ps(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
        }
		typedef CalibPoseValidator validator;
        static std::string name() { return "gp4ps"; }
    };


    struct SolverP2P2L {
        static inline int solve(const ProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
            return p2p2l(instance.x_point_, instance.X_point_, instance.x_line_, instance.X_line_, instance.V_line_, solutions);
        }
		typedef CalibPoseValidator validator;
        static std::string name() { return "p2p2l"; }
    };

    struct SolverUP2P {
        static inline int solve(const ProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
            return up2p(instance.x_point_, instance.X_point_, solutions);
        }
		typedef CalibPoseValidator validator;
        static std::string name() { return "up2p"; }
    };

    struct SolverUP1P2L {
        static inline int solve(const ProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
            return up1p2l(instance.x_point_, instance.X_point_, instance.x_line_, instance.X_line_, instance.V_line_, solutions);
        }
		typedef CalibPoseValidator validator;
        static std::string name() { return "up1p2l"; }
    };

    struct SolverUP4L {
        static inline int solve(const ProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
            return up4l(instance.x_line_, instance.X_line_, instance.V_line_, solutions);
        }
		typedef CalibPoseValidator validator;
        static std::string name() { return "up4l"; }
    };

}