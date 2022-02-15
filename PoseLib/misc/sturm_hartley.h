// Copyright Richard Hartley, 2010
// static const char *copyright = "Copyright Richard Hartley, 2010";

//--------------------------------------------------------------------------
// LICENSE INFORMATION
//
// 1.  For academic/research users:
//
// This program is free for academic/research purpose:   you can redistribute
// it and/or modify  it under the terms of the GNU General Public License as 
// published by the Free Software Foundation, either version 3 of the License,
// or (at your option) any later version.
//
// Under this academic/research condition,  this program is distributed in 
// the hope that it will be useful, but WITHOUT ANY WARRANTY; without even 
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along 
// with this program. If not, see <http://www.gnu.org/licenses/>.
//
// 2.  For commercial OEMs, ISVs and VARs:
// 
// For OEMs, ISVs, and VARs who distribute/modify/use this software 
// (binaries or source code) with their products, and do not license and 
// distribute their source code under the GPL, please contact NICTA 
// (www.nicta.com.au), and NICTA will provide a flexible OEM Commercial 
// License. 
//
//---------------------------------------------------------------------------

/*
 * sturm.c
 *
 * the functions to build and evaluate the Sturm sequence
 */

// #define RH_DEBUG
#pragma once
#include "hidden6.h"
#include <math.h>
#include <stdio.h>

#define RELERROR      1.0e-12   /* smallest relative error we want */
//#define MAXPOW        0        /* max power of 10 we wish to search to */
#define MAXIT         800       /* max number of iterations */
#define SMALL_ENOUGH  1.0e-12   /* a coefficient smaller than SMALL_ENOUGH 
                                 * is considered to be zero (0.0). */

/* structure type for representing a polynomial */
typedef struct p {
   int ord;
   double coef[Maxdegree+1];
   } poly;

/*---------------------------------------------------------------------------
 * evalpoly
 *
 * evaluate polynomial defined in coef returning its value.
 *--------------------------------------------------------------------------*/

double evalpoly (int ord, double *coef, double x)
   {
   double *fp = &coef[ord];
   double f = *fp;

   for (fp--; fp >= coef; fp--)
      f = x * f + *fp;

   return(f);
   }

int modrf_pos( int ord, double *coef, double a, double b, 
	double *val, int invert)
   {
   int  its;
   double fx, lfx;
   double *fp;
   double *scoef = coef;
   double *ecoef = &coef[ord];
   double fa, fb;

   // Invert the interval if required
   if (invert)
      {
      double temp = a;
      a = 1.0 / b;
      b = 1.0 / temp;
      }

   // Evaluate the polynomial at the end points
   if (invert)
      {
      fb = fa = *scoef;
      for (fp = scoef + 1; fp <= ecoef; fp++) 
         {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
         }
      }
   else
      {
      fb = fa = *ecoef;
      for (fp = ecoef - 1; fp >= scoef; fp--) 
         {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
         }
      }

   // if there is no sign difference the method won't work
   if (fa * fb > 0.0)
      return(0);

   // Return if the values are close to zero already
   if (fabs(fa) < RELERROR) 
      {
      *val = invert ? 1.0/a : a;
      return(1);
      }

   if (fabs(fb) < RELERROR) 
      {
      *val = invert ? 1.0/b : b;
      return(1);
      }

   lfx = fa;

   for (its = 0; its < MAXIT; its++) 
      {
      // Assuming straight line from a to b, find zero
      double x = (fb * a - fa * b) / (fb - fa);

      // Evaluate the polynomial at x
      if (invert)
         {
         fx = *scoef;
         for (fp = scoef + 1; fp <= ecoef; fp++)
            fx = x * fx + *fp;
	 }
      else
         {
         fx = *ecoef;
         for (fp = ecoef - 1; fp >= scoef; fp--)
            fx = x * fx + *fp;
         }

      // Evaluate two stopping conditions
      if (fabs(x) > RELERROR && fabs(fx/x) < RELERROR) 
         {
         *val = invert ? 1.0/x : x;
         return(1);
         }
      else if (fabs(fx) < RELERROR) 
         {
         *val = invert ? 1.0/x : x;
         return(1);
         }

      // Subdivide region, depending on whether fx has same sign as fa or fb
      if ((fa * fx) < 0) 
         {
         b = x;
         fb = fx;
         if ((lfx * fx) > 0)
            fa /= 2;
         } 
      else 
         {
         a = x;
         fa = fx;
         if ((lfx * fx) > 0)
            fb /= 2;
         }

   
      // Return if the difference between a and b is very small
      if (fabs(b-a) < fabs(RELERROR * a))
         {
         *val = invert ? 1.0/a : a;
         return(1);
         }

      lfx = fx;
      }

   //==================================================================
   // This is debugging in case something goes wrong.
   // If we reach here, we have not converged -- give some diagnostics
   //==================================================================

   fprintf(stderr, "modrf overflow on interval %f %f\n", a, b);
   fprintf(stderr, "\t b-a = %12.5e\n", b-a);
   fprintf(stderr, "\t fa  = %12.5e\n", fa);
   fprintf(stderr, "\t fb  = %12.5e\n", fb);
   fprintf(stderr, "\t fx  = %12.5e\n", fx);

   // Evaluate the true values at a and b
   if (invert)
      {
      fb = fa = *scoef;
      for (fp = scoef + 1; fp <= ecoef; fp++)
         {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
         }
      }
   else
      {
      fb = fa = *ecoef;
      for (fp = ecoef - 1; fp >= scoef; fp--) 
         {
         fa = a * fa + *fp;
         fb = b * fb + *fp;
         }
      }

   fprintf(stderr, "\t true fa = %12.5e\n", fa);
   fprintf(stderr, "\t true fb = %12.5e\n", fb);
   fprintf(stderr, "\t gradient= %12.5e\n", (fb-fa)/(b-a));

   // Print out the polynomial
   fprintf(stderr, "Polynomial coefficients\n");
   for (fp = ecoef; fp >= scoef; fp--) 
      fprintf (stderr, "\t%12.5e\n", *fp);

   return(0);
   }

/*---------------------------------------------------------------------------
 * modrf
 *
 * uses the modified regula-falsi method to evaluate the root
 * in interval [a,b] of the polynomial described in coef. The
 * root is returned is returned in *val. The routine returns zero
 * if it can't converge.
 *--------------------------------------------------------------------------*/

int modrf (int ord, double *coef, double a, double b, double *val)
   {
   // This is an interfact to modrf that takes account of different cases
   // The idea is that the basic routine works badly for polynomials on
   // intervals that extend well beyond [-1, 1], because numbers get too large

   double *fp;
   double *scoef = coef;
   double *ecoef = &coef[ord];
   const int invert = 1;

   double fp1= 0.0, fm1 = 0.0; // Values of function at 1 and -1
   double fa = 0.0, fb  = 0.0; // Values at end points
 
   // We assume that a < b
   if (a > b)
      {
      double temp = a;
      a = b;
      b = temp;
      }

   // The normal case, interval is inside [-1, 1]
   if (b <= 1.0 && a >= -1.0) return modrf_pos (ord, coef, a, b, val, !invert);

   // The case where the interval is outside [-1, 1]
   if (a >= 1.0 || b <= -1.0)
      return modrf_pos (ord, coef, a, b, val, invert);

   // If we have got here, then the interval includes the points 1 or -1.
   // In this case, we need to evaluate at these points

   // Evaluate the polynomial at the end points
   for (fp = ecoef - 1; fp >= scoef; fp--) 
      {
      fp1 = *fp + fp1;
      fm1 = *fp - fm1;
      fa = a * fa + *fp;
      fb = b * fb + *fp;
      }

   // Then there is the case where the interval contains -1 or 1
   if (a < -1.0 && b > 1.0)
      {
      // Interval crosses over 1.0, so cut
      if (fa * fm1 < 0.0)      // The solution is between a and -1
         return modrf_pos (ord, coef, a, -1.0, val, invert);
      else if (fb * fp1 < 0.0) // The solution is between 1 and b
         return modrf_pos (ord, coef, 1.0, b, val, invert);
      else                     // The solution is between -1 and 1
         return modrf_pos(ord, coef, -1.0, 1.0, val, !invert);
      }
   else if (a < -1.0)
      {
      // Interval crosses over 1.0, so cut
      if (fa * fm1 < 0.0)      // The solution is between a and -1
         return modrf_pos (ord, coef, a, -1.0, val, invert);
      else                     // The solution is between -1 and b
         return modrf_pos(ord, coef, -1.0, b, val, !invert); 
      }
   else  // b > 1.0
      {
      if (fb * fp1 < 0.0) // The solution is between 1 and b
         return modrf_pos (ord, coef, 1.0, b, val, invert);
      else                     // The solution is between a and 1
         return modrf_pos(ord, coef, a, 1.0, val, !invert);
      }
   }

/*---------------------------------------------------------------------------
 * modp
 *
 *  calculates the modulus of u(x) / v(x) leaving it in r, it
 *  returns 0 if r(x) is a constant.
 *  note: this function assumes the leading coefficient of v is 1 or -1
 *--------------------------------------------------------------------------*/

static int modp(poly *u, poly *v, poly *r)
   {
   int j, k;  /* Loop indices */

   double *nr = r->coef;
   double *end = &u->coef[u->ord];

   double *uc = u->coef;
   while (uc <= end)
      *nr++ = *uc++;

   if (v->coef[v->ord] < 0.0) {

      for (k = u->ord - v->ord - 1; k >= 0; k -= 2)
         r->coef[k] = -r->coef[k];

      for (k = u->ord - v->ord; k >= 0; k--)
         for (j = v->ord + k - 1; j >= k; j--)
            r->coef[j] = -r->coef[j] - r->coef[v->ord + k]
         * v->coef[j - k];
      } else {
         for (k = u->ord - v->ord; k >= 0; k--)
            for (j = v->ord + k - 1; j >= k; j--)
               r->coef[j] -= r->coef[v->ord + k] * v->coef[j - k];
      }

   k = v->ord - 1;
   while (k >= 0 && fabs(r->coef[k]) < SMALL_ENOUGH) {
      r->coef[k] = 0.0;
      k--;
      }

   r->ord = (k < 0) ? 0 : k;

   return(r->ord);
   }

/*---------------------------------------------------------------------------
 * buildsturm
 *
 * build up a sturm sequence for a polynomial in smat, returning
 * the number of polynomials in the sequence
 *--------------------------------------------------------------------------*/

int buildsturm(int ord, poly *sseq)
   {
   sseq[0].ord = ord;
   sseq[1].ord = ord - 1;

   /* calculate the derivative and normalise the leading coefficient */
      {
      int i;    // Loop index
      poly *sp;
      double f = fabs(sseq[0].coef[ord] * ord);
      double *fp = sseq[1].coef;
      double *fc = sseq[0].coef + 1;

      for (i=1; i<=ord; i++)
         *fp++ = *fc++ * i / f;

      /* construct the rest of the Sturm sequence */
      for (sp = sseq + 2; modp(sp - 2, sp - 1, sp); sp++) {

         /* reverse the sign and normalise */
         f = -fabs(sp->coef[sp->ord]);
         for (fp = &sp->coef[sp->ord]; fp >= sp->coef; fp--)
            *fp /= f;
         }

      sp->coef[0] = -sp->coef[0]; /* reverse the sign */

      return(sp - sseq);
      }
   }

/*---------------------------------------------------------------------------
 * numchanges
 *
 * return the number of sign changes in the Sturm sequence in
 * sseq at the value a.
 *--------------------------------------------------------------------------*/

int numchanges(int np, poly *sseq, double a)
   {
   int changes = 0;

   double lf = evalpoly(sseq[0].ord, sseq[0].coef, a);

   poly *s;
   for (s = sseq + 1; s <= sseq + np; s++) {
      double f = evalpoly(s->ord, s->coef, a);
      if (lf == 0.0 || lf * f < 0)
         changes++;
      lf = f;
      }

   return(changes);
   }

/*---------------------------------------------------------------------------
 * numroots
 *
 * return the number of distinct real roots of the polynomial described in sseq.
 *--------------------------------------------------------------------------*/

int numroots(int np, poly *sseq, int *atneg, int *atpos, bool non_neg)
   {
   int atposinf = 0;
   int atneginf = 0;

   /* changes at positive infinity */
   double f;
   double lf = sseq[0].coef[sseq[0].ord];

   poly *s;
   for (s = sseq + 1; s <= sseq + np; s++) {
      f = s->coef[s->ord];
      if (lf == 0.0 || lf * f < 0)
         atposinf++;
      lf = f;
      }

   // changes at negative infinity or zero
   if (non_neg)
      atneginf = numchanges(np, sseq, 0.0);

   else
      {
      if (sseq[0].ord & 1)
         lf = -sseq[0].coef[sseq[0].ord];
      else
         lf = sseq[0].coef[sseq[0].ord];

      for (s = sseq + 1; s <= sseq + np; s++) {
         if (s->ord & 1)
            f = -s->coef[s->ord];
         else
            f = s->coef[s->ord];
         if (lf == 0.0 || lf * f < 0)
            atneginf++;
         lf = f;
         }
      }

   *atneg = atneginf;
   *atpos = atposinf;

   return(atneginf - atposinf);
   }


/*---------------------------------------------------------------------------
 * sbisect
 *
 * uses a bisection based on the sturm sequence for the polynomial
 * described in sseq to isolate intervals in which roots occur,
 * the roots are returned in the roots array in order of magnitude.
 *--------------------------------------------------------------------------*/

int sbisect(int np, poly *sseq, 
            double min, double max, 
            int atmin, int atmax, 
            double *roots)
   {
   double mid;
   int atmid;
   int its;
   int  n1 = 0, n2 = 0;
   int nroot = atmin - atmax;

   if (nroot == 1) {

      /* first try a less expensive technique.  */
      if (modrf(sseq->ord, sseq->coef, min, max, &roots[0]))
         return 1;

      /*
       * if we get here we have to evaluate the root the hard
       * way by using the Sturm sequence.
       */
      for (its = 0; its < MAXIT; its++) {
         mid = (double) ((min + max) / 2);
         atmid = numchanges(np, sseq, mid);

         if (fabs(mid) > RELERROR) {
            if (fabs((max - min) / mid) < RELERROR) {
               roots[0] = mid;
               return 1;
               }
            } else if (fabs(max - min) < RELERROR) {
               roots[0] = mid;
               return 1;
            }

         if ((atmin - atmid) == 0)
            min = mid;
         else
            max = mid;
         }

      if (its == MAXIT) {
         fprintf(stderr, "sbisect: overflow min %f max %f\
                         diff %e nroot %d n1 %d n2 %d\n",
                         min, max, max - min, nroot, n1, n2);
         roots[0] = mid;
         }

      return 1;
      }

   /* more than one root in the interval, we have to bisect */
   for (its = 0; its < MAXIT; its++) {

      mid = (double) ((min + max) / 2);
      atmid = numchanges(np, sseq, mid);

      n1 = atmin - atmid;
      n2 = atmid - atmax;

      if (n1 != 0 && n2 != 0) {
         sbisect(np, sseq, min, mid, atmin, atmid, roots);
         sbisect(np, sseq, mid, max, atmid, atmax, &roots[n1]);
         break;
         }

      if (n1 == 0)
         min = mid;
      else
         max = mid;
      }

   if (its == MAXIT) {
      fprintf(stderr, "sbisect: roots too close together\n");
      fprintf(stderr, "sbisect: overflow min %f max %f diff %e\
                      nroot %d n1 %d n2 %d\n",
                      min, max, max - min, nroot, n1, n2);
      for (n1 = atmax; n1 < atmin; n1++)
         roots[n1 - atmax] = mid;
      }

   return 1; 
   }

int find_real_roots_sturm( 
	double *p, int order, double *roots, int *nroots, int maxpow, bool non_neg)
   {
   /*
    * finds the roots of the input polynomial.  They are returned in roots.
    * It is assumed that roots is already allocated with space for the roots.
    */

   poly sseq[Maxdegree+1];
   double  min, max;
   int  i, nchanges, np, atmin, atmax;

   // Copy the coefficients from the input p.  Normalize as we go
   double norm = 1.0 / p[order];
   for (i=0; i<=order; i++)
      sseq[0].coef[i] =  p[i] * norm;

   // Now, also normalize the other terms
   double val0 = fabs(sseq[0].coef[0]);
   double fac = 1.0; // This will be a factor for the roots
   if (val0 > 10.0)  // Do this in case there are zero roots
      {
      fac = pow(val0, -1.0/order);
      double mult = fac;
      for (int i=order-1; i>=0; i--)
         {
         sseq[0].coef[i] *= mult;
         mult = mult * fac; 
         }
      }

   /* build the Sturm sequence */
   np = buildsturm(order, sseq);

#ifdef RH_DEBUG
   {
   int i, j;

   printf("Sturm sequence for:\n");
   for (i=order; i>=0; i--)
      printf("%lf ", sseq[0].coef[i]);
   printf("\n\n");

   for (i = 0; i <= np; i++) {
      for (j = sseq[i].ord; j >= 0; j--)
         printf("%10f ", sseq[i].coef[j]);
      printf("\n");
      }

   printf("\n");
   }
#endif

   // get the number of real roots
   *nroots = numroots(np, sseq, &atmin, &atmax, non_neg);

   if (*nroots == 0) {
      // fprintf(stderr, "solve: no real roots\n");
      return 0 ;
      }

   /* calculate the bracket that the roots live in */
   if (non_neg) min = 0.0;
   else
      {
      min = -1.0;
      nchanges = numchanges(np, sseq, min);
      for (i = 0; nchanges != atmin && i != maxpow; i++) { 
         min *= 10.0;
         nchanges = numchanges(np, sseq, min);
         }

      if (nchanges != atmin) {
         //printf("solve: unable to bracket all negative roots\n");
         atmin = nchanges;
         }
      }

   max = 1.0;
   nchanges = numchanges(np, sseq, max);
   for (i = 0; nchanges != atmax && i != maxpow; i++) { 
      max *= 10.0;
      nchanges = numchanges(np, sseq, max);
      }

   if (nchanges != atmax) {
      //printf("solve: unable to bracket all positive roots\n");
      atmax = nchanges;
      }

   *nroots = atmin - atmax;

   /* perform the bisection */
   sbisect(np, sseq, min, max, atmin, atmax, roots);

   /* Finally, reorder the roots */
   for (i=0; i<*nroots; i++)
      roots[i] /= fac;

#ifdef RH_DEBUG

   /* write out the roots */
   printf("Number of roots = %d\n", *nroots);
   for (i=0; i<*nroots; i++)
      printf("%12.5e\n", roots[i]);

#endif

   return 1; 
   }
