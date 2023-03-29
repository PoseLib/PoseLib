#include "PoseLib/alignment.h"
#include <iostream>
#include <tuple>
#define REQUIRE(COND) if(!(COND)) { std::cout << "Failure: "#COND" was not satisfied.\n"; return false; }
#define REQUIRE_EQ(VAL1,VAL2) {auto _v1 = VAL1; auto _v2 = VAL2; if(_v1 != _v2) { std::cout << "Failure: "#VAL1" was not equal to "#VAL2". (" << _v1 << " vs. " << _v2 <<  ")\n"; return false; }}
#define REQUIRE_SMALL(VAL,THR) {auto _v1 = VAL; auto _v2 = THR; if(std::isnan(_v1) || std::abs(_v1) > _v2) { std::cout << "Failure: "#VAL" was not small "#THR". (" << _v1 << " vs. " << _v2 <<  ")\n"; return false; }}

typedef bool (*TestFunc)();
typedef std::pair<TestFunc,std::string> Test;
#define TEST(FUNC) std::make_pair(FUNC, std::string(#FUNC))

