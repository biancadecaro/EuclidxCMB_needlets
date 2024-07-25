/**
 * @file needlets_unit.h
 * @author Alessandro Renzi
 * @date 16 12 2015
 * @brief Module to manage all the operations related to the needlets transform.
 *
 * In contrast to my Fortran lib, this module provide all the functions for
 * needlets transformation and manipulation.
 *
 */

#ifndef __MYLIB_NEEDLETS_UNIT__
#define __MYLIB_NEEDLETS_UNIT__

#include <stdio.h>
#include <tgmath.h> //includes <math.h> and <complex.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_matrix.h>


#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

void mylib_print_needlets_parameters(void); //DEBUG

double mylib_jmax_lmax2B(int jmax,int lmax);
double mylib_jmax_xmax2B(int jmax,double xmax);
int mylib_x2j_lbound(double B, double x);
int mylib_x2j_ubound(double B, double x);
int mylib_B_xmin2jmin(double B, double xmin);
int mylib_B_xmax2jmax(double B, double xmax);

double mylib_f_standard_needlets(double t, void * params);
void mylib_c_qags(double a, double b, double epsabs, double epsrel,\
		double *result_out, double *abserr_out);
double mylib_phi(double t,double B,double *norm);
double mylib_needlets_std_normalization(void);
double mylib_b_of_x(double x,double B,double *norm);

void mylib_needlets_std_init_b_values_harmonic(gsl_matrix *b_values,
		double B,int jmax, int lmax);
void mylib_needlets_check_windows(int jmax,int lmax,gsl_matrix *b_values);

void mylib_needlets_std_init_b_values_even_square_uncentred_1d(gsl_matrix *b_values,
		double B,int jmax,int N, double L);
void mylib_needlets_f2betajk_omp_even_uncentred_1d(int N, gsl_vector *func,
		gsl_matrix *betajk,double B,int jmax, double L);
void mylib_needlets_betajk2f_omp_even_uncentred_1d(int N, gsl_matrix *betajk,
		gsl_vector *func, double B, int jmax, double L);

void mylib_needlets_std_init_b_values_even_square_uncentred_2d(gsl_matrix **b_values,
		double B,int jmax,int N, double L);
void mylib_needlets_f2betajk_omp_even_uncentred_2d(int N, gsl_matrix *square,
		gsl_matrix **betajk,double B,int jmax, double L);
void mylib_needlets_betajk2f_omp_even_uncentred_2d(int N, gsl_matrix **betajk,
		gsl_matrix *square, double B, int jmax, double L);

#ifdef __cplusplus
}
#endif

#endif //__MYLIB_NEEDLETS_UNIT__
