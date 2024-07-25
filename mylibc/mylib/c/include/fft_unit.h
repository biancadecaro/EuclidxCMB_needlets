/**
 * @file fft_unit.h
 * @author Alessandro Renzi
 * @date 18 12 2015
 * @brief Module to manage all the operations related to the FFT transform.
 *
 */

#ifndef __MYLIB_FFT_UNIT__
#define __MYLIB_FFT_UNIT__

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // Must be defined before fftw3.h if we want to use the standard
										 // library complex type.
#include <fftw3.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include "constants.h"

#ifdef __cplusplus
extern "C" {
#endif

void mylib_fftshift_even_1d_dp(int N, gsl_vector *v);
void mylib_fftshift_even_1d_dpc(int N, gsl_vector_complex *vk);

void mylib_fftshift_even_square_2d_dp(int N, gsl_matrix *square);
void mylib_fftshift_even_square_2d_dpc(int N, gsl_matrix_complex *squarek);

void mylib_fftfreq_even_uncentred_1d(int N, gsl_vector *v, double L);
void mylib_fftfreq_even_square_uncentred_2d(int N, gsl_matrix *square, double L);

void mylib_fft_r2c_omp_even_uncentred_1d(int N,gsl_vector *v,
		gsl_vector_complex	*vk, double L);
void mylib_fft_c2r_omp_even_uncentred_1d(int N,gsl_vector_complex	*vk,
		gsl_vector *v, double L);
void mylib_fft_r2c_omp_even_uncentred_2d(int N, gsl_matrix *patch,
		gsl_matrix_complex *patchk, double L);
void mylib_fft_c2r_omp_even_uncentred_2d(int N, gsl_matrix_complex *patchk,
		gsl_matrix *patch, double L);

#ifdef __cplusplus
}
#endif

#endif //__MYLIB_FFT_UNIT__
