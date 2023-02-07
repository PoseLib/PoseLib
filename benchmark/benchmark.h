#pragma once

#include "PoseLib/poselib.h"
#include "problem_generator.h"
#include "PoseLib/misc/radial.h"

#include <Eigen/Dense>
#include <stdint.h>
#include <string>
#include <vector>
#include <iostream> // HACK

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

struct SolverP4PF {
    static inline int solve(const AbsolutePoseProblemInstance &instance, poselib::CameraPoseVector *solutions,
                            std::vector<double> *focals) {
        return p4pf(instance.x_point_, instance.X_point_, solutions, focals);
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
    static std::string name() { return "RelUpright3pt"; }
};

struct SolverGenRelUpright4pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_upright_4pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "GenRelUpright4pt"; }
};

struct SolverRel8pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_8pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "Rel8pt"; }
};

struct SolverRel5pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_5pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "Rel5pt"; }
};

struct SolverGenRel5p1pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_5p1pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "GenRel5p1pt"; }
};

struct SolverGenRel6pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return gen_relpose_6pt(instance.p1_, instance.x1_, instance.p2_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "GenRel6pt"; }
};

struct SolverRelUprightPlanar2pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_upright_planar_2pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
    static std::string name() { return "RelUprightPlanar2pt"; }
};

struct SolverRelUprightPlanar3pt {
    static inline int solve(const RelativePoseProblemInstance &instance, poselib::CameraPoseVector *solutions) {
        return relpose_upright_planar_3pt(instance.x1_, instance.x2_, solutions);
    }
    typedef CalibPoseValidator validator;
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

struct SolverHomographyRadialFitzgibbon5pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<Eigen::Matrix3d> *solutions, std::vector<double> *distortion_parameters, std::vector<double> *dummy) {
        Eigen::Matrix3d H;
        double r;
        int sols = homography_fitzgibbon_cvpr_2001(instance.x1_, instance.x2_, &H, &r);
        solutions->clear();
        distortion_parameters->clear();
        dummy->clear();
        if (sols == 1) {
            solutions->push_back(H);
            distortion_parameters->push_back(r);
            dummy->push_back(r);
        }
        return sols;
    }
    typedef RadialHomographyValidator validator;
    static std::string name() { return "Homography5pt Fitz"; }
};

struct SolverHomographyRadialKukelova5pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<Eigen::Matrix3d> *solutions, std::vector<double> *distortion_parameters1, std::vector<double> *distortion_parameters2) {
        return homography_kukelova_cvpr_2015(instance.x1_, instance.x2_, solutions, distortion_parameters1, distortion_parameters2);
    }
    typedef RadialHomographyValidator validator;
    static std::string name() { return "Homography5pt Kukelova"; }
};

struct SolverHomographyValtonenOrnhagICPR4pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<Eigen::Matrix3d> *solutions, std::vector<double> *focal_lengths, std::vector<double> *dummies) {
        Eigen::Matrix3d H;
        Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
        double f;
        int sols = homography_valtonenornhag_icpr_2020(instance.x1_, instance.x2_, R1, instance.pose_gt.R(), &H, &f);
        solutions->clear();
        focal_lengths->clear();
        dummies->clear();
        if (sols == 1) {
            solutions->push_back(H);
            focal_lengths->push_back(f);
            dummies->push_back(f);
        }
        return sols;
    }
    typedef UnknownFocalHomographyValidator validator;
    static std::string name() { return "Homography4pt ICPR"; }
};

struct SolverHomographyValtonenOrnhagWACV3pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<Eigen::Matrix3d> *solutions, std::vector<double> *focal_lengths,  std::vector<double> *dummies) {
        Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
        int sols = homography_valtonenornhag_wacv_2021_fHf(instance.x1_, instance.x2_, R1, instance.pose_gt.R(), solutions, focal_lengths);
        *dummies = *focal_lengths;
        return sols;
    }
    typedef UnknownFocalHomographyValidator validator;
    static std::string name() { return "Homography3pt WACV fHf"; }
};

struct SolverHomographyRadialValtonenOrnhagWACV4pt {
    static inline int solve(const RelativePoseProblemInstance &instance, std::vector<Eigen::Matrix3d> *solutions, std::vector<double> *focal_lengths, std::vector<double> *distortion_parameters) {
        Eigen::Matrix3d H;
        Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
        double f;
        double r;
        int sols = homography_valtonenornhag_wacv_2021_frHfr(instance.x1_, instance.x2_, R1, instance.pose_gt.R(), &H, &f, &r);
        solutions->clear();
        focal_lengths->clear();
        distortion_parameters->clear();
        if (sols == 1) {
            solutions->push_back(H);
            focal_lengths->push_back(f);
            distortion_parameters->push_back(r);
        }
        return sols;
    }
    typedef UnknownFocalAndRadialHomographyValidator validator;
    static std::string name() { return "Homography4pt WACV frHfr"; }
};

} // namespace poselib
