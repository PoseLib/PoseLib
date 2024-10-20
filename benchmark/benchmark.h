#pragma once

#include "PoseLib/poselib.h"
#include "problem_generator.h"

#include <Eigen/Dense>
#include <iostream>
#include <stdint.h>
#include <string>
#include <vector>

namespace poselib {

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
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p3p(instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p3p"; }
};

struct SolverP3P_lambdatwist {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p3p_lambdatwist(instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p3p_lambdatwist"; }
};

struct SolverP4PF {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        std::vector<Eigen::Vector2d> p2d(4);
        for (int i = 0; i < 4; ++i) {
            p2d[i] = instance.x_point_[i].hnormalized();
        }
        return p4pf(p2d, instance.X_point_, solutions, focals, true);
    }
    typedef UnknownFocalValidator validator;
    static std::string name() { return "p4pf"; }
};

struct SolverGP3P {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gp3p(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "gp3p"; }
};

struct SolverGP4PS {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *scales) {
        return gp4ps(instance.p_point_, instance.x_point_, instance.X_point_, solutions, scales);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "gp4ps"; }
};

struct SolverP2P2PL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p2p2pl(instance.x_point_, instance.X_point_, instance.x_line_, instance.X_line_, instance.V_line_,
                      solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p2p2pl"; }
};

struct SolverP6LP {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p6lp(instance.l_line_point_, instance.X_line_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p6lp"; }
};
struct SolverP5LP_Radial {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p5lp_radial(instance.l_line_point_, instance.X_line_point_, solutions);
    }
    typedef RadialPoseValidator validator;
    static std::string name() { return "p5lp_radial"; }
};

struct SolverP2P1LL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p2p1ll(instance.x_point_, instance.X_point_, instance.l_line_line_, instance.X_line_line_,
                      instance.V_line_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p2p1ll"; }
};

struct SolverP1P2LL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p1p2ll(instance.x_point_, instance.X_point_, instance.l_line_line_, instance.X_line_line_,
                      instance.V_line_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p1p2ll"; }
};

struct SolverP3LL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return p3ll(instance.l_line_line_, instance.X_line_line_, instance.V_line_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "p3ll"; }
};

struct SolverUP2P {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return up2p(instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "up2p"; }
};

struct SolverUP1P1LL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return up1p1ll(instance.x_point_[0], instance.X_point_[0], instance.l_line_line_[0], instance.X_line_line_[0],
                       instance.V_line_line_[0], solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "up1p1ll"; }
};

struct SolverUGP2P {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return ugp2p(instance.p_point_, instance.x_point_, instance.X_point_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "ugp2p"; }
};

struct SolverUGP3PS {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *scales) {
        return ugp3ps(instance.p_point_, instance.x_point_, instance.X_point_, solutions, scales);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "ugp3ps"; }
};

struct SolverUP1P2PL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return up1p2pl(instance.x_point_, instance.X_point_, instance.x_line_, instance.X_line_, instance.V_line_,
                       solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "up1p2pl"; }
};

struct SolverUP4PL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return up4pl(instance.x_line_, instance.X_line_, instance.V_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "up4pl"; }
};

struct SolverUGP4PL {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return ugp4pl(instance.p_line_, instance.x_line_, instance.X_line_, instance.V_line_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "ugp4pl"; }
};

struct SolverRelUpright3pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_upright_3pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "RelUpright3pt"; }
};

struct SolverGenRelUpright4pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_upright_4pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "GenRelUpright4pt"; }
};

struct SolverRel8pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_8pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "Rel8pt"; }
};

struct SolverSharedFocalRel6pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::ImagePairVector *solutions) {
        return relpose_6pt_shared_focal(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef ImagePair Solution;
    static std::string name() { return "SharedFocalRel6pt"; }
};

struct SolverRel5pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_5pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "Rel5pt"; }
};

struct SolverGenRel5p1pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_5p1pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "GenRel5p1pt"; }
};

struct SolverGenRel6pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_6pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "GenRel6pt"; }
};

struct SolverRelUprightPlanar2pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_upright_planar_2pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "RelUprightPlanar2pt"; }
};

struct SolverRelUprightPlanar3pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_upright_planar_3pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    typedef CameraPose Solution;
    static std::string name() { return "RelUprightPlanar3pt"; }
};

template <bool CheiralCheck = false> struct SolverHomography4pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<Eigen::Matrix3d> *solutions) {
        Eigen::Matrix3d H;
        int sols = homography_4pt(instance.x1_, instance.x2_, &H, CheiralCheck);
        solutions->clear();
        if (sols == 1) {
            solutions->push_back(H);
        }
        return sols;
    }
    typedef HomographyValidator validator;
    static std::string name() {
        if (CheiralCheck) {
            return "Homography4pt(C)";
        } else {
            return "Homography4pt";
        }
    }
};

} // namespace poselib
