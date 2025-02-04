#include "fft_unit.h"
#include "needlets_unit.h"

#define MYLIB_NEEDLETS_EPSABS MAX_TOL_DP*100./**< Needlets precision */
#define MYLIB_NEEDLETS_EPSREL MAX_TOL_DP*100. /**< Needlets precision */
#define MYLIB_NEEDLETS_LIMIT_LOWER 1.+MAX_TOL_DP*10. /**< Increase an epsilon*/
#define MYLIB_NEEDLETS_LIMIT_UPPER 1.-MAX_TOL_DP*10. /**< Decrease an epsilon*/
#define MYLIB_QAGS_LIMIT 1000

/**
 * @brief Print needlets precision parameters.
 * @return always return 0
 */
void mylib_print_needlets_parameters() {

	printf("needlets_epsabs=%e,needlets_epsrel=%e\n",
			MYLIB_NEEDLETS_EPSABS,MYLIB_NEEDLETS_EPSREL);
	printf("needlets_limit_lower=%30.20e,needlets_limit_upper=%30.20e\n\n",
			MYLIB_NEEDLETS_LIMIT_LOWER,MYLIB_NEEDLETS_LIMIT_UPPER);

} //mylib_print_needlets_parameters

/**
 * @brief Calculate the parameter B given lmax and jmax
 *
 * This function calculate the value of the needlets width parameter B that
 * center the *last* needlets windows function b(l/B**j) on lmax.
 *
 * @param jmax Maximum needlets frequency
 * @param lmax The experiments maximum harmonic value (integer value)
 * @return Needlets width parameter B
 */
double mylib_jmax_lmax2B(int jmax,int lmax) {
  return pow((lmax*MYLIB_NEEDLETS_LIMIT_LOWER),(1./jmax));
}

/**
 * @brief Calculate the parameter B given xmax and jmax
 *
 * This function calculate the value of the needlets width parameter B that
 * center the *last* needlets windows function b(x/B**j) on xmax. In contrast to
 * the case of mylib_jmax_lmax2B now xmax could be any double precision number
 * greater than zero that represent the maximum value in the needlets windows
 * function argument (for example freqmax in case of 1D fourier transform).
 *
 * @param jmax Maximum needlets frequency
 * @param xmax The needlets maximum argument value (double precision value)
 * @return Needlets width parameter B
 */
double mylib_jmax_xmax2B(int jmax,double xmax) {
  return pow((xmax*MYLIB_NEEDLETS_LIMIT_LOWER),(1./jmax));
}//mylib_jmax_xmax2B



/**
 * @brief Calculate minimum frequency j that contains the number x given B
 *
 * @param B Needlets width parameter
 * @param x (double precision value)
 * @return j Lower j that contains x
 */
int mylib_x2j_lbound(double B, double x) {
	return abs(ceil(-1.+log2(x*MYLIB_NEEDLETS_LIMIT_LOWER)/log2(B)));
}//mylib_x2j_lbound



/**
 * @brief Calculate maximum frequency j that contains the number x given B
 *
 * @param B Needlets width parameter
 * @param x (double precision value)
 * @return j Upper j that contains x
 */
int mylib_x2j_ubound(double B, double x) {
	return abs(floor(1.+log2(x*MYLIB_NEEDLETS_LIMIT_UPPER)/log2(B)));
} //mylib_x2j_ubound



/**
 * @brief Calculate the parameter jmin given B and xmin
 *
 * @param B Needlets width parameter
 * @param xmin The needlets minimum argument value (double precision value)
 * @return jmin Minimum needlets frequency
 */
int mylib_B_xmin2jmin(double B, double xmin) {
  return mylib_x2j_lbound(B,xmin);
}//mylib_B_xmax2jmax



/**
 * @brief Calculate the parameter jmax given B and xmax
 *
 * @param B Needlets width parameter
 * @param xmax The needlets maximum argument value (double precision value)
 * @return jmax Maximum needlets frequency
 */
int mylib_B_xmax2jmax(double B, double xmax) {
  return mylib_x2j_ubound(B,xmax);
}//mylib_B_xmax2jmax



/**
 * @brief Standard needlets function
 *
 * This function is used to construct the standard needlets.
 * See Section 2 step 1 on the cited paper below. This function can be
 * substituted by any C^inf function compactly supported in [-1,1], in that case
 * pay attention to the needlets normalization.
 *
 * @param t input parameter
 * @param params only one parameter: needlets relative error
 * @return Standard needlets f (Section 2 step 1)
 *
 * @see arXiv:0707.0844
 *
 */
double mylib_f_standard_needlets (double t, void * params) {
  double epsrel = *(double *) params;
  double f1=0.;
  if ((-1. < t) && (t < 1.)) {
    if (abs(abs(t)-1.) < epsrel) {
      f1=0.;
    } else {
      f1=exp(-1./(1.-(t*t)));
    }
  }
  return f1;
} //mylib_f_standard_needlets



/**
 * @brief gsl QAGS integrator wrapper
 *
 * @param a integration inf limit
 * @param b integration sup limit
 * @param epsabs needlets absolute error
 * @param epsrel needlets relative error
 * @param *result_out integral output
 * @param *abserr_out estimated absolute error
 * @return return 0 if no problems, 1 otherwise
 *
 */
void mylib_c_qags(double a, double b, double epsabs, double epsrel,
		double *result_out, double *abserr_out) {

	int ierr=0;

  gsl_integration_workspace * w =
  		gsl_integration_workspace_alloc(MYLIB_QAGS_LIMIT);

  double result, abserr;

  gsl_function F;
  F.function = &mylib_f_standard_needlets;
  F.params = &epsrel;

  gsl_integration_qags(&F, a, b, epsabs, epsrel, MYLIB_QAGS_LIMIT,
  		w, &result, &abserr);

  if (abserr > epsrel) {
    printf ("*** Possible problems with qags****\n");
    printf ("estimated error   = % .18f\n", abserr);
    printf ("intervals         =  %d\n", w->size);
    ierr=1;
  }

  gsl_integration_workspace_free(w);

  *result_out=result;
  *abserr_out=abserr;

} //mylib_c_qags



/**
 * @brief Standard needlets Phi function
 *
 * This function is used to construct the standard needlets.
 * See Section 2 step 3 on the cited paper below.
 *
 * @param t input parameter
 * @param B needlets width parameter
 * @param norm standard needlets normalization (see the denominator of
 * 						 Section 2 step 2)
 * @return Standard needlets Phi (Section 2 step 3)
 *
 * @see arXiv:0707.0844
 *
 */
double mylib_phi(double t,double B,double *norm) {

	double phi;

  if ((0.<=t) && (t<=1./B)) {
    phi=1.;
  } else if (t>=1.) {
    phi=0.;
  } else {
    double app_result;
    double abserr;
    double appt=1.-(2.*B*(t-(1./B))/(B-1.));
    mylib_c_qags(-1.,appt,MYLIB_NEEDLETS_EPSABS,MYLIB_NEEDLETS_EPSREL,
    		&app_result,&abserr);
    phi=app_result/(*norm);
  }

	return phi;

} //mylib_phi



/**
 * @brief Normalization for the Standard needlets function
 *
 * @return Standard needlets normalization reference (Section 2 step 1)
 *
 * @see arXiv:0707.0844
 *
 */
double mylib_needlets_std_normalization() {
  double abserr,res;
  mylib_c_qags(-1.,1.,MYLIB_NEEDLETS_EPSABS,MYLIB_NEEDLETS_EPSREL,&res,&abserr);
	return res;
}//mylib_needlets_std_normalization



/**
 * @brief Standard needlets windows function
 *
 * This function is used to construct the standard needlets window function.
 * See Section 2 Step 4 on the cited paper below.
 *
 * @param x input parameter (could be ell, k, freq, or any number greater than 0)
 * @param B needlets width parameter
 * @param *norm standard needlets normalization (see the denominator of
 * 						 Section 2 step 2)
 * @return Standard needlets Phi (Section 2 step 3)
 *
 * @see arXiv:0707.0844
 *
 */
double mylib_b_of_x(double x,double B,double *norm) {
  double phi1=mylib_phi(x/B,B,norm);
  double phi2=mylib_phi(x,B,norm);

	return sqrt(fabs(phi1-phi2));

}//mylib_b_of_x



/**
 * @brief Standard needlets windows functions 1D: gsl_matrix
 *
 * This function is used to construct the standard needlets window functions.
 * The output is a table with jmax row and lmax columns. This function works
 * only for harmonics coefficients.
 *
 * @param **b_values standard needlets window functions (0:jmax,0:lmax)
 * @param B needlets width parameter
 * @param jmax maximum frequency (j=0,...,jmax)
 * @param lmax maximum harmonic value (l=0,...,lmax)
 * @return always return 0
 *
 * @see arXiv:0707.0844
 *
 */
void mylib_needlets_std_init_b_values_harmonic(gsl_matrix *b_values,
		double B,int jmax,int lmax) {

  int l,j;
  double norm=mylib_needlets_std_normalization();

  for(j=0; j<=jmax; ++j)
    for(l=0; l<=lmax; ++l)
			gsl_matrix_set(b_values,j,l,mylib_b_of_x((double)l/pow(B,j),B,&norm));

} //mylib_needlets_std_init_b_values_harmonic



/**
 * @brief This function check if the standard needlets windows functions are
 * 				correctly generated.
 *
 * Standard needlets function must respect the condition in equation 2 of the
 * paper cited below.
 *
 * @param jmax maximum frequency (j=0,...,jmax)
 * @param lmax maximum harmonic value (l=0,...,lmax)
 * @param *b_values standard needlets window functions (0:jmax,0:lmax)
 * @return always return 0
 *
 * @see arXiv:0707.0844
 *
 */
void mylib_needlets_check_windows(int jmax,int lmax,gsl_matrix *b_values) {
  double acc;

  for(int l=1; l<=lmax; ++l) {
  	acc=0.;
  	for(int j=0; j<=jmax;++j)
  		acc+=pow(gsl_matrix_get(b_values,j,l),2.);
    if ((acc-1.)>MYLIB_NEEDLETS_EPSREL)
    	printf("Wrong needlets windows? SUM[b(l=%d,:)]-1=%e must be zero!\n",
    			l,acc-1.);
  }

} //mylib_needlets_check_windows



/**
 * @brief Standard needlets windows functions 1D: array of gsl_matrix [FFT version]
 *
 * This function return the 1D standard needlets functions, in form of gsl_matrix
 * with dimension (jmax+1,N) (N is even).
 *
 * @param *b_values standard needlets window functions (0:jmax,0:N-1)
 * @param B needlets width parameter
 * @param jmax maximum frequency (j=0,...,jmax)
 * @param N number of points (sampling)
 * @param L physical dimension of the 1D signal - Not needed?
 *
 * @see arXiv:0707.0844
 *
 */
void mylib_needlets_std_init_b_values_even_square_uncentred_1d(gsl_matrix *b_values,
		double B,int jmax,int N, double L) {

  //double Delta=L/(double)N; //Sample Spacing
  //double Deltak=1./(double)N/Delta; //Sampling Rate
  double norm=mylib_needlets_std_normalization();
  int halfN=N/2;
  int j,k;

	#pragma omp parallel
  {

  #pragma omp for collapse(2)
  for(j=0; j<=jmax; ++j) {
    for(k=0; k<halfN; ++k) {
    	//gsl_matrix_set(b_values,j,k,mylib_b_of_x((k*Deltak)/pow(B,j),B,&norm));
    	//gsl_matrix_set(b_values,j,N-1-k,mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm));
    	gsl_matrix_set(b_values,j,k,mylib_b_of_x(k/pow(B,j),B,&norm));
    	gsl_matrix_set(b_values,j,N-1-k,mylib_b_of_x((k+1)/pow(B,j),B,&norm));
    }
  }//#pragma omp for collapse(2)

  }//#pragma omp parallel

} //mylib_needlets_std_init_b_values_even_square_uncentred_1d



/**
 * @brief Standard needlets transform for functions 1D [FFT version]
 *
 * This function return the coefficients of the standard needlets transform of
 * a function func.

 * @param N function' sampling points
 * @param *func gsl_vector function, NOTE: mean value must be subtracted before this
 * 								function is called
 * @param *betajk gsl_matrix with needlets coefficients of the function (0:jmax,0:N-1)
 * @param B needlets width parameter
 * @param jmax maximum frequency (j=0,...,jmax)
 * @param L FFT function real length - Forse non è necessaria!!!
 *
 * @see arXiv:0707.0844
 *
 */
void mylib_needlets_f2betajk_omp_even_uncentred_1d(int N, gsl_vector *func,
		gsl_matrix *betajk, double B,int jmax, double L) {

  //double Delta=L/(double)N; //Sample Spacing
  //double Deltak=1./(double)N/Delta; //Sampling Rate
  double norm=mylib_needlets_std_normalization();
  int halfN=N/2;
  int j,k;

  gsl_vector_complex *ak=gsl_vector_complex_alloc(N);
  //mylib_fft_r2c_omp_even_uncentred_1d(N,func,ak,L);
  mylib_fft_r2c_omp_even_uncentred_1d(N,func,ak,N);

  gsl_vector *betajk_j=gsl_vector_alloc(N);
  gsl_vector_complex *ak_j=gsl_vector_complex_alloc(N);

  double app1,app2;

//	#pragma omp parallel private(funckapp)
//  {

//  #pragma omp for
  for(j=0; j<=jmax; ++j) {
    for(k=0; k<halfN; ++k) {
    	//app1=mylib_b_of_x((k*Deltak)/pow(B,j),B,&norm);
    	//app2=mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm);
    	app1=mylib_b_of_x(k/pow(B,j),B,&norm);
    	app2=mylib_b_of_x((k+1)/pow(B,j),B,&norm);
    		gsl_vector_complex_set(ak_j,k,
    				gsl_complex_mul_real(gsl_vector_complex_get(ak,k),
    						//mylib_b_of_x((k*Deltak)/pow(B,j),B,&norm)));
    						app1));
    		gsl_vector_complex_set(ak_j,N-1-k,
    				gsl_complex_mul_real(gsl_vector_complex_get(ak,N-1-k),
    						//mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm)));
    						app2));
    }
    //mylib_fft_c2r_omp_even_uncentred_1d(N,ak_j,betajk_j,L); //This function is not Thread safe!
    mylib_fft_c2r_omp_even_uncentred_1d(N,ak_j,betajk_j,N); //This function is not Thread safe!

    gsl_matrix_set_row(betajk,j,betajk_j);

  }//#pragma omp for

//  }//#pragma omp parallel

  gsl_vector_free(betajk_j);
  gsl_vector_complex_free(ak);
  gsl_vector_complex_free(ak_j);

} //mylib_needlets_f2betajk_omp_even_uncentred_1d



/**
 * @brief Standard needlets transform reconstruction for functions 1D [FFT version]
 *
 * This function return the reconstruction of a function given the coefficients
 * of the standard needlets transform of a function func.

 * @param N function' sampling points
 * @param *betajk gsl_matrix with needlets coefficients of the function (0:jmax,0:N-1)
 * @param *func gsl_vector function
 * @param B needlets width parameter
 * @param jmax maximum frequency (j=0,...,jmax)
 * @param L FFT function real length - Forse non è necessaria!!!
 *
 * @see arXiv:0707.0844
 *
 */
void mylib_needlets_betajk2f_omp_even_uncentred_1d(int N, gsl_matrix *betajk,
		gsl_vector *func, double B,int jmax, double L) {

  //double Delta=L/(double)N; //Sample Spacing
  //double Deltak=1./(double)N/Delta; //Sampling Rate
  double norm=mylib_needlets_std_normalization();
  int halfN=N/2;
  int j,k;

  gsl_vector_set_zero(func);

  gsl_vector *betajk_j=gsl_vector_alloc(N);
  gsl_vector_complex *ak_j=gsl_vector_complex_alloc(N);

//	#pragma omp parallel private(funckapp)
//  {

//  #pragma omp for
  for(j=0; j<=jmax; ++j) {

  	gsl_matrix_get_row(betajk_j,betajk,j);

  	//mylib_fft_r2c_omp_even_uncentred_1d(N,betajk_j,ak_j,L); //This function is not Thread safe!
  	mylib_fft_r2c_omp_even_uncentred_1d(N,betajk_j,ak_j,N); //This function is not Thread safe!

    for(k=0; k<halfN; ++k) {
    	gsl_vector_complex_set(ak_j,k,
    			gsl_complex_mul_real(gsl_vector_complex_get(ak_j,k),
    					//mylib_b_of_x((k*Deltak)/pow(B,j),B,&norm)));
    					mylib_b_of_x(k/pow(B,j),B,&norm)));
    	gsl_vector_complex_set(ak_j,N-1-k,
    			gsl_complex_mul_real(gsl_vector_complex_get(ak_j,N-1-k),
    					//mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm)));
    					mylib_b_of_x((k+1)/pow(B,j),B,&norm)));
    }
    //mylib_fft_c2r_omp_even_uncentred_1d(N,ak_j,betajk_j,L);
    mylib_fft_c2r_omp_even_uncentred_1d(N,ak_j,betajk_j,N);

    gsl_vector_add(func,betajk_j);

  }//#pragma omp for

//  }//#pragma omp parallel private(appval)

  gsl_vector_free(betajk_j);
  gsl_vector_complex_free(ak_j);

} //mylib_needlets_betajk2f_omp_even_uncentred_1d



/**
 * @brief Standard needlets windows functions 2D: array of gsl_matrix
 *
 * This function return the 2D standard needlets functions, in form of array
 * of gsl_matrix, any matrix is a (N,N) patch for a frequency j (N is even).
 *
 * @param **b_values standard needlets window functions (0:N-1,0:N-1)
 * @param B needlets width parameter
 * @param jmax maximum frequency (j=0,...,jmax)
 * @param N side number dimension of a square patch
 * @param L side physical dimension of a square patch (degree)
 *
 * @see arXiv:0707.0844
 *
 */
void mylib_needlets_std_init_b_values_even_square_uncentred_2d(gsl_matrix **b_values,
		double B,int jmax,int N, double L) {

  double Delta=L/(double)N; //Sample Spacing
  double Deltak=1./(double)N/Delta; //Sampling Rate
  double norm=mylib_needlets_std_normalization();
  int halfN=N/2;
  int j,k1,k2;

	#pragma omp parallel
  {

  #pragma omp for collapse(3)
  for(j=0; j<=jmax; ++j) {
    for(k1=0; k1<halfN; ++k1) {
    	for(k2=0; k2<halfN; ++k2) {
    		gsl_matrix_set(b_values[j],k1,k2,
    				mylib_b_of_x(sqrt((double)(k1*k1+k2*k2))/pow(B,j)
    						//*Deltak,B,&norm));
    						,B,&norm));
    		gsl_matrix_set(b_values[j],k1,N-1-k2,
    				mylib_b_of_x(sqrt((double)(k1*k1+(k2+1)*(k2+1)))/pow(B,j)
    						//*Deltak,B,&norm));
    						,B,&norm));
    		gsl_matrix_set(b_values[j],N-1-k1,k2,
    				mylib_b_of_x(sqrt((double)((k1+1)*(k1+1)+k2*k2))/pow(B,j)
    						//*Deltak,B,&norm));
    						,B,&norm));
    		gsl_matrix_set(b_values[j],N-1-k1,N-1-k2,
    				mylib_b_of_x(sqrt((double)((k1+1)*(k1+1)+(k2+1)*(k2+1)))/pow(B,j)
    						//*Deltak,B,&norm));
    						,B,&norm));
    	}
    }
  }//#pragma omp for

  }//#pragma omp parallel

}//mylib_needlets_std_init_b_values_even_square_uncentred_2d



/**
 * @brief Standard needlets transform for functions 2D [FFT version]
 *
 * This function return the coefficients of the standard needlets transform of
 * a function func.

 * @param N function' sampling points
 * @param *func gsl_matrix function, NOTE: mean value must be subtracted before this
 * 								function is called
 * @param **betajk array of gsl_matrix with needlets coefficients of the function (0:jmax,0:N-1,0:N-1)
 * @param B needlets width parameter
 * @param jmax maximum frequency (j=0,...,jmax)
 * @param L FFT function real length - Forse non è necessaria!!!
 *
 * @see arXiv:0707.0844
 *
 */
void mylib_needlets_f2betajk_omp_even_uncentred_2d(int N, gsl_matrix *square,
		gsl_matrix **betajk, double B,int jmax, double L) {

  //double Delta=L/(double)N; //Sample Spacing
  //double Deltak=1./(double)N/Delta; //Sampling Rate
  double norm=mylib_needlets_std_normalization();
  int halfN=N/2;
  int j,k1,k2;

  gsl_matrix_complex *ak1k2=gsl_matrix_complex_alloc(N,N);
  mylib_fft_r2c_omp_even_uncentred_2d(N,square,ak1k2,N);

  gsl_matrix *betajk_j=gsl_matrix_alloc(N,N);
  gsl_matrix_complex *ak1k2_j=gsl_matrix_complex_alloc(N,N);

//	#pragma omp parallel private(funckapp)
//  {

//  #pragma omp for
  for(j=0; j<=jmax; ++j) {
    for(k1=0; k1<halfN; ++k1)
    	for(k2=0; k2<halfN; ++k2) {
				gsl_matrix_complex_set(ak1k2_j,k1,k2,
						gsl_complex_mul_real(gsl_matrix_complex_get(ak1k2,k1,k2),
								//mylib_b_of_x((k*Deltak)/pow(B,j),B,&norm)));
								mylib_b_of_x(sqrt((double)(k1*k1+k2*k2))/pow(B,j),B,&norm)));
				gsl_matrix_complex_set(ak1k2_j,N-1-k1,k2,
						gsl_complex_mul_real(gsl_matrix_complex_get(ak1k2,N-1-k1,k2),
								//mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm)));
								mylib_b_of_x(sqrt((double)((k1+1)*(k1+1)+k2*k2))/pow(B,j),B,&norm)));
				gsl_matrix_complex_set(ak1k2_j,k1,N-1-k2,
						gsl_complex_mul_real(gsl_matrix_complex_get(ak1k2,k1,N-1-k2),
								//mylib_b_of_x((k*Deltak)/pow(B,j),B,&norm)));
								mylib_b_of_x(sqrt((double)(k1*k1+(k2+1)*(k2+1)))/pow(B,j),B,&norm)));
				gsl_matrix_complex_set(ak1k2_j,N-1-k1,N-1-k2,
						gsl_complex_mul_real(gsl_matrix_complex_get(ak1k2,N-1-k1,N-1-k2),
								//mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm)));
								mylib_b_of_x(sqrt((double)((k1+1)*(k1+1)+(k2+1)*(k2+1)))/pow(B,j),B,&norm)));
    	}
    //mylib_fft_c2r_omp_even_uncentred_1d(N,ak_j,betajk_j,L); //This function is not Thread safe!
    mylib_fft_c2r_omp_even_uncentred_2d(N,ak1k2_j,betajk_j,N); //This function is not Thread safe!

    //gsl_matrix_set_row(betajk,j,betajk_j);
    gsl_matrix_memcpy(betajk[j],betajk_j);

  }//#pragma omp for

//  }//#pragma omp parallel

  gsl_matrix_free(betajk_j);
  gsl_matrix_complex_free(ak1k2);
  gsl_matrix_complex_free(ak1k2_j);

} //mylib_needlets_f2betajk_omp_even_uncentred_2d



/**
 * @brief Standard needlets transform reconstruction for functions 2D [FFT version]
 *
 * This function return the reconstruction of a function given the coefficients
 * of the standard needlets transform of a function square.

 * @param N function' sampling points
 * @param **betajk vector of gsl_matrix with needlets coefficients of the function (0:jmax,0:N-1)
 * @param *square gsl_matrix function on grid
 * @param B needlets width parameter
 * @param jmax maximum frequency (j=0,...,jmax)
 * @param L FFT function real length - Forse non è necessaria!!!
 *
 * @see arXiv:0707.0844
 *
 */
void mylib_needlets_betajk2f_omp_even_uncentred_2d(int N, gsl_matrix **betajk,
		gsl_matrix *square, double B,int jmax, double L) {

  //double Delta=L/(double)N; //Sample Spacing
  //double Deltak=1./(double)N/Delta; //Sampling Rate
  double norm=mylib_needlets_std_normalization();
  int halfN=N/2;
  int j,k1,k2;

  gsl_matrix_set_zero(square);

  gsl_matrix *betajk_j=gsl_matrix_alloc(N,N);
  gsl_matrix_complex *ak1k2_j=gsl_matrix_complex_alloc(N,N);

//	#pragma omp parallel private(funckapp)
//  {

//  #pragma omp for
  for(j=0; j<=jmax; ++j) {

  	gsl_matrix_memcpy(betajk_j,betajk[j]);

  	//mylib_fft_r2c_omp_even_uncentred_1d(N,betajk_j,ak_j,L); //This function is not Thread safe!
  	mylib_fft_r2c_omp_even_uncentred_2d(N,betajk_j,ak1k2_j,N); //This function is not Thread safe!

  	for(k1=0; k1<halfN; ++k1) {
			for(k2=0; k2<halfN; ++k2) {
				gsl_matrix_complex_set(ak1k2_j,k1,k2,
						gsl_complex_mul_real(gsl_matrix_complex_get(ak1k2_j,k1,k2),
								//mylib_b_of_x((k*Deltak)/pow(B,j),B,&norm)));
								mylib_b_of_x(sqrt((double)(k1*k1+k2*k2))/pow(B,j),B,&norm)));
				gsl_matrix_complex_set(ak1k2_j,N-1-k1,k2,
						gsl_complex_mul_real(gsl_matrix_complex_get(ak1k2_j,N-1-k1,k2),
								//mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm)));
								mylib_b_of_x(sqrt((double)((k1+1)*(k1+1)+k2*k2))/pow(B,j),B,&norm)));
				gsl_matrix_complex_set(ak1k2_j,k1,N-1-k2,
						gsl_complex_mul_real(gsl_matrix_complex_get(ak1k2_j,k1,N-1-k2),
								//mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm)));
								mylib_b_of_x(sqrt((double)(k1*k1+(k2+1)*(k2+1)))/pow(B,j),B,&norm)));
				gsl_matrix_complex_set(ak1k2_j,N-1-k1,N-1-k2,
						gsl_complex_mul_real(gsl_matrix_complex_get(ak1k2_j,N-1-k1,N-1-k2),
								//mylib_b_of_x(((k+1)*Deltak)/pow(B,j),B,&norm)));
								mylib_b_of_x(sqrt((double)((k1+1)*(k1+1)+(k2+1)*(k2+1)))/pow(B,j),B,&norm)));
			}
  	}

    //mylib_fft_c2r_omp_even_uncentred_1d(N,ak_j,betajk_j,L);
    mylib_fft_c2r_omp_even_uncentred_2d(N,ak1k2_j,betajk_j,N);

    gsl_matrix_add(square,betajk_j);

  }//#pragma omp for

//  }//#pragma omp parallel private(appval)

  gsl_matrix_free(betajk_j);
  gsl_matrix_complex_free(ak1k2_j);

} //mylib_needlets_betajk2f_omp_even_uncentred_2d
