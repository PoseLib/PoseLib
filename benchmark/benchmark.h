#pragma once

#include "PoseLib/poselib.h"

#include "problem_generator.h"
#include <Eigen/Dense>
#include <stdint.h>
#include <string>
#include <vector>

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
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return p3p(instance.x_point_, instance.X_point_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "p3p"; }
};

struct SolverP4PF {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return p4pf(instance.x_point_, instance.X_point_, solutions);
  }
  typedef UnknownFocalValidator validator;
  static std::string name() { return "p4pf"; }
};

struct SolverGP3P {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return gp3p(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "gp3p"; }
};

struct SolverGP4PS {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return gp4ps(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "gp4ps"; }
};

struct SolverP2P2PL {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return p2p2pl(instance.x_point_, instance.X_point_, instance.x_line_, instance.X_line_, instance.V_line_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "p2p2pl"; }
};

struct SolverP6LP {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return p6lp(instance.l_line_point_, instance.X_line_point_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "p6lp"; }
};
struct SolverP5LP_Radial {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return p5lp_radial(instance.l_line_point_, instance.X_line_point_, solutions);
  }
  typedef RadialPoseValidator validator;
  static std::string name() { return "p5lp_radial"; }
};

struct SolverP2P1LL {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return p2p1ll(instance.x_point_, instance.X_point_, instance.l_line_line_, instance.X_line_line_, instance.V_line_line_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "p2p1ll"; }
};

struct SolverP1P2LL {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return p1p2ll(instance.x_point_, instance.X_point_, instance.l_line_line_, instance.X_line_line_, instance.V_line_line_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "p1p2ll"; }
};

struct SolverP3LL {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return p3ll(instance.l_line_line_, instance.X_line_line_, instance.V_line_line_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "p3ll"; }
};

struct SolverUP2P {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return up2p(instance.x_point_, instance.X_point_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "up2p"; }
};

struct SolverUGP2P {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return ugp2p(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "ugp2p"; }
};

struct SolverUGP3PS {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return ugp3ps(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "ugp3ps"; }
};

struct SolverUP1P2PL {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return up1p2pl(instance.x_point_, instance.X_point_, instance.x_line_, instance.X_line_, instance.V_line_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "up1p2pl"; }
};

struct SolverUP4PL {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return up4pl(instance.x_line_, instance.X_line_, instance.V_line_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "up4pl"; }
};

struct SolverUGP4PL {
  static inline int solve(const AbsolutePoseProblemInstance &instance, pose_lib::CameraPoseVector *solutions) {
    return ugp4pl(instance.p_line_, instance.x_line_, instance.X_line_, instance.V_line_, solutions);
  }
  typedef CalibPoseValidator validator;
  static std::string name() { return "ugp4pl"; }
};

struct SolverGenRelUpright4pt {
    static inline int solve(const RelativePoseProblemInstance& instance, pose_lib::CameraPoseVector* solutions) {
        return gen_relpose_upright_4pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "GenRelUpright4pt"; }
};

} // namespace pose_lib