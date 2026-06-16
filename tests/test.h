#ifndef POSELIB_TEST_H_
#define POSELIB_TEST_H_
#include "PoseLib/alignment.h"

#include <Eigen/Dense>
#include <iostream>
#include <sstream>
#include <string>
#include <tuple>

#define REQUIRE(COND)                                                                                                  \
    do {                                                                                                               \
        if (!(COND)) {                                                                                                 \
            std::cout << "Failure: " #COND " was not satisfied.\n";                                                    \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0);
#define REQUIRE_EQ(VAL1, VAL2)                                                                                         \
    do {                                                                                                               \
        auto _v1 = VAL1;                                                                                               \
        auto _v2 = VAL2;                                                                                               \
        if (_v1 != _v2) {                                                                                              \
            std::cout << "Failure: " #VAL1 " was not equal to " #VAL2 ". (" << _v1 << " vs. " << _v2 << ")\n";         \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0);
#define REQUIRE_SMALL(VAL, THR)                                                                                        \
    do {                                                                                                               \
        auto _v1 = VAL;                                                                                                \
        auto _v2 = THR;                                                                                                \
        if (std::isnan(_v1) || std::abs(_v1) > _v2) {                                                                  \
            std::cout << "Failure: " #VAL " was not small " #THR ". (" << _v1 << " vs. " << _v2 << ")\n";              \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0);

#define REQUIRE_M(COND, MSG)                                                                                           \
    do {                                                                                                               \
        if (!(COND)) {                                                                                                 \
            std::cout << "Failure: " #COND " was not satisfied. (" + MSG + ").\n";                                     \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0);
#define REQUIRE_EQ_M(VAL1, VAL2, MSG)                                                                                  \
    do {                                                                                                               \
        auto _v1 = VAL1;                                                                                               \
        auto _v2 = VAL2;                                                                                               \
        if (_v1 != _v2) {                                                                                              \
            std::cout << "Failure: " #VAL1 " was not equal to " #VAL2 ". (" << _v1 << " vs. " << _v2                   \
                      << ") (" + MSG + ")\n";                                                                          \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0);
#define REQUIRE_SMALL_M(VAL, THR, MSG)                                                                                 \
    do {                                                                                                               \
        auto _v1 = VAL;                                                                                                \
        auto _v2 = THR;                                                                                                \
        if (std::isnan(_v1) || std::abs(_v1) > _v2) {                                                                  \
            std::cout << "Failure: " #VAL " was not small " #THR ". (" << _v1 << " vs. " << _v2                        \
                      << ") (" + MSG + ")\n";                                                                          \
            return false;                                                                                              \
        }                                                                                                              \
    } while (0);

static inline std::string to_string(const Eigen::MatrixXd &mat) {
    std::stringstream ss;
    ss << mat;
    return ss.str();
}

typedef bool (*TestFunc)();
typedef std::pair<TestFunc, std::string> Test;
#define TEST(FUNC) std::make_pair(FUNC, std::string(#FUNC))

#endif