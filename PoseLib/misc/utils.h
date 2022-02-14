#pragma once
#include <Eigen/Eigen>
#include <cfloat>

// converts from unit quaternion to rotation matrix
void quat2R(Eigen::Vector4d const q, Eigen::Matrix3d & R);

// creates a skew symmetric matrix 
 Eigen::Matrix3d X_(Eigen::Vector3d const v);


void colEchelonForm(Eigen::MatrixXd &M, std::list<int> &b);

void inputSwitchDirection( Eigen::MatrixXd &Xo, Eigen::MatrixXd &u);

template<typename T>
T outputSwitchDirection(const T & in);
