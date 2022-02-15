#include "utils.h"


// converts from unit quaternion to rotation matrix
void quat2R(Eigen::Vector4d const q, Eigen::Matrix3d & R) {
	double q1 = q(0);
	double q2 = q(1);
	double q3 = q(2);
	double q4 = q(3);

	R << q1*q1 + q2*q2 - q3*q3 - q4*q4, 2 * q2*q3 - 2 * q1*q4, 2 * q1*q3 + 2 * q2*q4,
		2 * q1*q4 + 2 * q2*q3, q1*q1 - q2*q2 + q3*q3 - q4*q4, 2 * q3*q4 - 2 * q1*q2,
		2 * q2*q4 - 2 * q1*q3, 2 * q1*q2 + 2 * q3*q4, q1*q1 - q2*q2 - q3*q3 + q4*q4;

}

// creates a skew symmetric matrix 
 Eigen::Matrix3d X_(Eigen::Vector3d const v) {
	 Eigen::Matrix3d S;
	 S << 0, -v(2), v(1), v(2), 0, -v(0), -v(1), v(0), 0;
	 return S;
}


void colEchelonForm(Eigen::MatrixXd &M, std::list<int> &b)
{
	int n = M.rows();
	int m = M.cols();
	int i = 0, j = 0, k = 0;
	int col = 0;
	double p, tp;

	int maxmn = (M.cols()<M.rows()) ? M.rows() : M.cols();

	double pivtol = 2.2204e-16*maxmn*M.lpNorm<Eigen::Infinity>();

	b.clear();

	while ((i < m) && (j < n))
	{
		p = DBL_MIN;
		col = i;
		for (k = i; k < m; k++)
		{
			tp = std::abs(M(j, k));
			if (tp > p)
			{
				p = tp;
				col = k;
			}
		}

		if (p < pivtol)
		{
			M.block(j, i, 1, m - i).setZero();
			j++;
		}
		else
		{
			b.push_back(j);

			if (col != i)
				M.block(j, i, n - j, 1).swap(M.block(j, col, n - j, 1));

			M.block(j + 1, i, n - j - 1, 1) /= M(j, i);
			M(j, i) = 1.0;

			for (k = 0; k < m; k++)
			{
				if (k == i)
					continue;

				M.block(j, k, n - j, 1) -= M(j, k) * M.block(j, i, n - j, 1);
			}

			i++;
			j++;
		}
	}
}

void inputSwitchDirection(Eigen::MatrixXd &X, Eigen::MatrixXd &u){
	Eigen::VectorXd temp1 = X.row(0);
	Eigen::VectorXd temp2 = u.row(0);
	X.row(0) = X.row(1);
	X.row(1) = temp1;
	u.row(0) = u.row(1);
	u.row(1) = temp2; 
}

template<typename T>
T outputSwitchDirection(const T & in){
	T out;
	out.v(0) = - in.v(1);
	out.v(1) = - in.v(0);
	out.v(2) = - in.v(2);
	out.w(0) = - in.w(1);
	out.w(1) = - in.w(0);
	out.w(2) = - in.w(2);
	out.C(0) = in.C(1);
	out.C(1) = in.C(0);
	out.C(2) = in.C(2);
	out.t(0) = in.t(1);
	out.t(1) = in.t(0);
	out.t(2) = in.t(2);	
	out.f = in.f;
	out.rd = in.rd;

	return out;
}


