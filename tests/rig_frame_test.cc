// Regression test for the frame-dependent singularities of gp3p / gp4ps / up2p / ugp2p / ugp3ps.
// Each case is a well-posed minimal problem with exact data; the solver must return the true pose.
#include "PoseLib/solvers/gp3p.h"
#include "PoseLib/solvers/gp4ps.h"
#include "PoseLib/solvers/ugp2p.h"
#include "PoseLib/solvers/ugp3ps.h"
#include "PoseLib/solvers/up2p.h"
#include "test.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

using namespace poselib;
namespace {
using V3 = Eigen::Vector3d;
static std::mt19937 rng(12345);
static double U(double a, double b) { return std::uniform_real_distribution<double>(a, b)(rng); }
static Eigen::Matrix3d randrot() {
    Eigen::Quaterniond q(U(-1, 1), U(-1, 1), U(-1, 1), U(-1, 1));
    return q.normalized().toRotationMatrix();
}
static Eigen::Matrix3d roty(double a) {
    Eigen::Matrix3d R;
    R << std::cos(a), 0, std::sin(a), 0, 1, 0, -std::sin(a), 0, std::cos(a);
    return R;
}
static double best_err(const std::vector<CameraPose> &sols, const Eigen::Matrix3d &R, const V3 &t) {
    double e = INFINITY;
    for (const auto &s : sols) {
        double ei = (s.R() - R).cwiseAbs().maxCoeff() + (s.t - t).cwiseAbs().maxCoeff();
        if (std::isfinite(ei) && ei < e)
            e = ei;
    }
    return e;
}
enum Kind { GENERIC, SAME_COLUMN_01, Z0_RAY0 };
struct Scene {
    std::vector<V3> p, x, X;
    Eigen::Matrix3d R;
    V3 t;
    double s;
};
static Scene make(int n, Kind kind, bool upright, bool central, bool with_scale) {
    Scene sc;
    sc.R = upright ? roty(U(-3, 3)) : randrot();
    sc.t = V3(U(-1, 1), U(-1, 1), U(-1, 1));
    sc.s = with_scale ? U(0.5, 2.0) : 1.0;
    for (int i = 0; i < n; ++i) {
        V3 d(U(-.6, .6), U(-.6, .6), 1.0);
        if (kind == SAME_COLUMN_01 && i == 1)
            d(0) = sc.x[0](0) / sc.x[0](2);
        if (kind == Z0_RAY0 && i == 0)
            d = V3(1.0, U(-.6, .6), 0.0);
        V3 x = d.normalized();
        V3 p = central ? V3::Zero() : V3(U(-.5, .5), U(-.5, .5), U(-.5, .5));
        V3 Y = sc.s * p + U(2, 6) * x;
        sc.p.push_back(p);
        sc.x.push_back(x);
        sc.X.push_back(sc.R.transpose() * (Y - sc.t));
    }
    return sc;
}
bool test_rig_frame_conditioning() {
    rng.seed(12345);
    const char *names[] = {"generic", "same_column_01", "z0_ray0"};
    int failures = 0;
    for (int k = 0; k < 3; ++k) {
        Kind kind = static_cast<Kind>(k);
        int f_gp3p = 0, f_gp4ps = 0, f_up2p = 0, f_ugp2p = 0, f_ugp3ps = 0, N = 500;
        double w_gp3p = 0, w_gp4ps = 0, w_up2p = 0, w_ugp2p = 0, w_ugp3ps = 0;
        for (int it = 0; it < N; ++it) {
            std::vector<CameraPose> sols;
            std::vector<double> scales;
            double e;
            Scene a = make(3, kind, false, false, false);
            gp3p(a.p, a.x, a.X, &sols);
            e = best_err(sols, a.R, a.t);
            f_gp3p += !(e < 1e-5);
            if (!(e < 1e-5) && std::getenv("DEBUG")) {
                std::printf("  gp3p fail kind=%d it=%d nsols=%zu err=%.3e\n", k, it, sols.size(), e);
                for (int i = 0; i < 3; ++i)
                    std::printf("    x%d = %.17g %.17g %.17g  p = %.17g %.17g %.17g\n", i, a.x[i](0), a.x[i](1),
                                a.x[i](2), a.p[i](0), a.p[i](1), a.p[i](2));
                for (auto &s : sols)
                    std::printf("    sol err %.3e\n",
                                (s.R() - a.R).cwiseAbs().maxCoeff() + (s.t - a.t).cwiseAbs().maxCoeff());
            }
            if (e < INFINITY)
                w_gp3p = std::max(w_gp3p, e);
            Scene b = make(4, kind, false, false, true);
            gp4ps(b.p, b.x, b.X, &sols, &scales, false);
            e = best_err(sols, b.R, b.t);
            f_gp4ps += !(e < 1e-5);
            if (e < INFINITY)
                w_gp4ps = std::max(w_gp4ps, e);
            if (kind != SAME_COLUMN_01) {
                Scene c = make(2, kind, true, true, false);
                up2p(c.x, c.X, &sols);
                e = best_err(sols, c.R, c.t);
                f_up2p += !(e < 1e-5);
                if (e < INFINITY)
                    w_up2p = std::max(w_up2p, e);
                Scene d = make(2, kind, true, false, false);
                ugp2p(d.p, d.x, d.X, &sols);
                e = best_err(sols, d.R, d.t);
                f_ugp2p += !(e < 1e-5);
                if (e < INFINITY)
                    w_ugp2p = std::max(w_ugp2p, e);
                Scene g = make(3, kind, true, false, true);
                ugp3ps(g.p, g.x, g.X, &sols, &scales, false);
                e = best_err(sols, g.R, g.t);
                f_ugp3ps += !(e < 1e-5);
                if (e < INFINITY)
                    w_ugp3ps = std::max(w_ugp3ps, e);
            }
        }
        std::printf("%-15s failures/%d: gp3p %3d  gp4ps %3d  up2p %3d  ugp2p %3d  ugp3ps %3d   "
                    "(worst finite err %.1e %.1e %.1e %.1e %.1e)\n",
                    names[k], N, f_gp3p, f_gp4ps, f_up2p, f_ugp2p, f_ugp3ps, w_gp3p, w_gp4ps, w_up2p, w_ugp2p,
                    w_ugp3ps);
        failures += f_gp3p + f_gp4ps + f_up2p + f_ugp2p + f_ugp3ps;
    }
    std::printf(failures ? "RESULT: FAIL (%d)\n" : "RESULT: PASS\n", failures);
    return failures == 0;
}

} // namespace

std::vector<Test> register_rig_frame_test() { return {TEST(test_rig_frame_conditioning)}; }
