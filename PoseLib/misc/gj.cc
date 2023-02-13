// Copyright (c) 2018 James Pritts
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Modified for PoseLib to accept Eigen matrices by Marcus Valtonen Örnhag

#include <Eigen/Dense>
#include "gj.h"

namespace poselib {
    void gj(Eigen::MatrixXd *M) {
        int rcnt = M->rows();
        int ccnt = M->cols();

        M->transposeInPlace();
        double* A = M->data();

        double tol = 1e-15;
        int r = 0;      // row
        int c = 0;      // col
        int k;
        int l;
        int dstofs;
        int srcofs;
        int ofs = 0;
        int pofs = 0;
        double b;

        // gj
        ofs = 0;
        pofs = 0;
        while (r < rcnt && c < ccnt) {
            // find pivot
            double apivot = 0;
            double pivot = 0;
            int pivot_r = -1;

            pofs = ofs;
            for (k = r; k < rcnt; k++) {
                // pivot selection criteria here !
                if (fabs(*(A+pofs)) > apivot) {
                    pivot = *(A+pofs);
                    apivot = fabs(pivot);
                    pivot_r = k;
                }
                pofs += ccnt;
            }

            if (apivot < tol) {
                // empty col - shift to next col (or jump)
                c++;
                ofs++;

            } else {
                // process rows

                // exchange pivot and selected rows
                // + divide row
                if (pivot_r == r) {
                    srcofs = ofs;
                    for (l = c; l < ccnt; l++) {
                        *(A+srcofs) = *(A+srcofs)/pivot;
                        srcofs++;
                    }

                } else {
                    srcofs = ofs;
                    dstofs = ccnt*pivot_r+c;
                    for (l = c; l < ccnt; l++) {
                        b = *(A+srcofs);
                        *(A+srcofs) = *(A+dstofs)/pivot;
                        *(A+dstofs) = b;

                        srcofs++;
                        dstofs++;
                    }
                }

                // zero bottom
                pofs = ofs + ccnt;
                for (k = r + 1; k < rcnt; k++) {
                        // nonzero row
                        b = *(A+pofs);
                        dstofs = pofs + 1;
                        srcofs = ofs + 1;
                        for (l = c + 1; l < ccnt; l++) {
                            *(A+dstofs) = (*(A+dstofs) - *(A+srcofs) * b);
                            dstofs++;
                            srcofs++;
                        }
                        *(A+pofs) = 0;

                    pofs += ccnt;
                }

                // zero top
                pofs = c;
                for (k = 0; k < r; k++) {
                        // nonzero row
                        b = *(A+pofs);
                        dstofs = pofs + 1;
                        srcofs = ofs + 1;
                        for (l = c + 1; l < ccnt; l++) {
                            *(A+dstofs) = (*(A+dstofs) - *(A+srcofs) * b);
                            dstofs++;
                            srcofs++;
                        }
                        *(A+pofs) = 0;

                    pofs += ccnt;
                }

                r++;
                c++;
                ofs += ccnt + 1;
            }
        }

        (*M) << Eigen::Map<Eigen::MatrixXd>(A, ccnt, rcnt);
        M->transposeInPlace();
    }
}  // namespace HomLib
