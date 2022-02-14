#pragma once
#include <Eigen/Eigen>
#include <iostream>
void charpoly_danilevsky(Eigen::MatrixXd A, double (&p)[65], double pivtol = 1e-14) {
	int nr = A.rows();
	int nc = A.cols();
	if (nr != nc) {
		std::cerr << "The matrix is not rectangular";
		return;
	}

	for (int i = nr-2; i >= 0; i--)
	{
		int pidx = i;

		if(std::abs(A(i+1,i))<pivtol){
			double pivval = 0;
			for(int j = 0; j<=i;j++){
				double aval = std::abs(A(i+1,i));
				if(aval > pivval){
					pidx = j;
					pivval = aval;
				}
			}
		}

		if(pidx!=i){
			Eigen::VectorXd tmp = A.row(pidx);
			A.row(pidx) = A.row(i);
			A.row(i) =tmp;

			Eigen::MatrixXd tmp2 = A.block(0,pidx,i+1,1);
			A.block(0,pidx,i+1,1) = A.block(0,i,i+1,1);
			A.block(0,i,i+1,1) = tmp2;
		}

		Eigen::VectorXd t = -A.row(i+1);
		t(i) = 1;
		t = t/A(i+1,i);

		A.row(i) = A.row(i+1)*A;

		Eigen::MatrixXd B = Eigen::MatrixXd::Zero(i+1,nr);

        for(int j =0; j<=i; j++){
            Eigen::VectorXd t2= A.row(j);
            t2(i) = 0;
            B.row(j) = A(j,i)*t+t2;
        }

        A.block(0,0,i+1,nr) = B;
        A.row(i+1) = Eigen::VectorXd::Zero(nr);
        A(i+1,i) = 1;
	}
p[0] = 1;
for(int i=1; i<nr+1; i++){
    p[i] = -A(0,i-1);
}
}