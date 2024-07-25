/*
 * fft_unit.c
 *
 *  Created on: 18 dic 2015
 *      Author: alexnino
 */

#include "fft_unit.h"

/**
 * @brief Shift the zero-frequency component to the centre of the spectrum.
 *
 * This function swaps half-spaces for x axis. Note that v[0] is the Nyquist
 * component only if dim(v)=N is even [which is the case].
 *
 * @param N vector dimension
 * @param v real gsl_vector of dimension N
 */
void mylib_fftshift_even_1d_dp(int N, gsl_vector *v) {

  double temp;
  int x,halfN=N/2;

	#pragma omp parallel private(temp)
  {

		#pragma omp for
		for(x=0;x<halfN;++x) {
			temp=gsl_vector_get(v,x);
			gsl_vector_set(v,x,gsl_vector_get(v,x+halfN));
			gsl_vector_set(v,x+halfN,temp);
		} //#pragma omp for

  } //#pragma omp parallel

} //mylib_fftshift_even_1d_dp



/**
 * @brief Shift the zero-frequency component to the centre of the spectrum.
 *
 * This function swaps half-spaces for x axis. Note that v[0] is the Nyquist
 * component only if dim(v)=N is even [which is the case].
 *
 * @param N vector dimension
 * @param v complex gsl_vector of dimension N
 */
void mylib_fftshift_even_1d_dpc(int N, gsl_vector_complex *vk) {

  gsl_complex temp;
  int x,halfN=N/2;

	#pragma omp parallel private(temp)
  {

		#pragma omp for
		for(x=0;x<halfN;++x) {
			temp=gsl_vector_complex_get(vk,x);
			gsl_vector_complex_set(vk,x,gsl_vector_complex_get(vk,x+halfN));
			gsl_vector_complex_set(vk,x+halfN,temp);
		} //#pragma omp for

  } //#pragma omp parallel

} //mylib_fftshift_even_1d_dpc



/**
 * @brief Return the Discrete Fourier Transform sample frequencies.
 *
 * The returned float array f contains the frequency bin centers in cycles per
 * unit of the sample spacing (with zero at the start). For instance, if the
 * sample spacing is in seconds, then the frequency unit is cycles/second.
 *
 * @param N vector dimension
 * @param v gsl_vector of dimension N
 * @param L Sample dimension (Sample Spacing = L/N)
 */
void mylib_fftfreq_even_uncentred_1d(int N, gsl_vector *v, double L) {

	double Delta=L/(double)N; //Sample Spacing
	double Deltak=1./(double)N/Delta; //Sampling Rate

	int j,halfN=N/2;

	//Populate output
	#pragma omp parallel
	{

		#pragma omp for
		for(j=0;j<halfN;++j) {
			gsl_vector_set(v,j,j*Deltak);
			gsl_vector_set(v,N-1-j,(-j-1)*Deltak);
		} //#pragma omp for

	} //#pragma omp parallel

}//mylib_fftfreq_even_uncentred_1d



/**
 * @brief Return the 2D Discrete Fourier Transform frequencies.
 *
 * NOTE: for this case we return a gsl_matrix with sqrt(k1**2+k2**2) values for
 * the grid points.
 *
 * @param N vector dimension
 * @param square gsl_matrix of dimension (N,N)
 * @param L Sample dimension (Sample Spacing = L/N)
 */
void mylib_fftfreq_even_square_uncentred_2d(int N, gsl_matrix *square, double L) {

	double Delta=L/(double)N; //Sample Spacing
	double Deltak=1./(double)N/Delta; //Sampling Rate

	int i,j,halfN=N/2;

	//Populate output
	#pragma omp parallel
	{

		#pragma omp for collapse(2)
		for(i=0;i<halfN;++i)
			for(j=0;j<halfN;++j) {
				gsl_matrix_set(square,i,j,sqrt((double)(i*i+j*j))*Deltak);
				gsl_matrix_set(square,i,N-1-j,sqrt((double)(i*i+(j+1)*(j+1)))*Deltak);
				gsl_matrix_set(square,N-1-i,j,sqrt((double)((i+1)*(i+1)+j*j))*Deltak);
				gsl_matrix_set(square,N-1-i,N-1-j,sqrt((double)((i+1)*(i+1)+(j+1)*(j+1)))*Deltak);
			} //#pragma omp for

	} //#pragma omp parallel

}//mylib_fftfreq_even_square_uncentred_2d

/**
 * @brief 1D FFT Real2Complex, openMP parallel for uncentred EVEN dimension data.
 *
 * Note: this function use 1 more vector memory space.
 * Normalisation can be included in the definition of Delta (The standard assume
 * that all normalisation goes to c2r [BACKWARD transform]).
 *
 * @param N vector dimension
 * @param *v double gsl_vector of dimension N
 * @param Delta real grid dimension, put 1 if not known or not used
 *
 * @param *vk OUTPUT complex gsl_vector of dimension N
 */
void mylib_fft_r2c_omp_even_uncentred_1d(int N,gsl_vector *v,
		gsl_vector_complex	*vk, double L) {

	double Delta=L/(double)N; //Sample Spacing
	//double Deltak=1./(double)N/Delta; //Sampling Rate
	double norm=Delta;
	int j;
	int hcdim=N/2+1; //half_complex dimension - EVEN
	int halfN=N/2;

  /**
   * FFTW requires aligned data stored in row major order (ALIGNED=flattened
   * multi-dimensional array).
   * Real and Imaginary parts of complex number stored as double[2].
   * Fortunately, GSL implements gsl_vector_complex in the required format.
   */

	double *in = v->data;
	fftw_complex *out = fftw_alloc_complex(hcdim);

	/**
	 * IMPORTANT NOTE: In general, use the plan *before* initialisation of arrays
	 * For better performance use FFTW_MEASURE, but then you must put
	 * initialisation after plans. [That means v(k) data must be copied].
	 */

	/**
	 * Threaded FFTW initialisation [before allocations can give errors!!!].
	 */
	int	iret=fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	//r2c is always FORWARD
	fftw_plan fp = fftw_plan_dft_r2c_1d(N,in,out,FFTW_ESTIMATE);
	//Calculate FFT
	fftw_execute(fp);
	fftw_destroy_plan(fp);

	//Populate output
	#pragma omp parallel
	{

		#pragma omp for
		for(j=0;j<halfN;++j) {
			gsl_vector_complex_set(vk,j,gsl_complex_mul_real(
				gsl_complex_rect(creal(out[j]),cimag(out[j])),norm));
			gsl_vector_complex_set(vk,j+halfN,gsl_complex_mul_real(
				gsl_complex_rect(creal(out[halfN-j]),-cimag(out[halfN-j])),norm));
		} //#pragma omp for

	} //#pragma omp parallel

	//Destroy plan
	//fftw_free(in); //PROBLEM with v->data?
	fftw_free(out);
	fftw_cleanup_threads();

} //mylib_fft_r2c_omp_even_uncentred_1d



/**
 * @brief 1D FFT Complex2Real, openMP parallel for uncentred EVEN dimension data.
 *
 * Note: this function use 1 more vector memory space.
 * Normalisation can be included in the definition of Deltak (The standard assume
 * that all normalisation goes to c2r [BACKWARD transform]).
 *
 * @param N vector dimension
 * @param *vk complex gsl_vector of dimension N
 * @param Deltak Fourier grid dimension, put 1 if not known or not used
 *
 * @param *v OUTPUT gsl_vector of dimension N
 */
void mylib_fft_c2r_omp_even_uncentred_1d(int N,gsl_vector_complex	*vk,
		gsl_vector *v, double L) {

	double Delta=L/(double)N; //Sample Spacing
	double Deltak=1./(double)N/Delta; //Sampling Rate
	double norm=Deltak; //REMEMBER: for FFTW we should normalise by N, so the real
	// normalisation is: norm=(N*Deltak)/N
	int j;
	int hcdim=N/2+1; //half_complex dimension - EVEN
	int halfN=N/2;

  /**
   * FFTW requires aligned data stored in row major order (ALIGNED=flattened
   * multi-dimensional array).
   * Real and Imaginary parts of complex number stored as double[2].
   * Fortunately, GSL implements gsl_vector_complex in the required format.
   */
	fftw_complex *in = fftw_alloc_complex(hcdim);

	//Initialise input vector (Half-Complex dimension)
	gsl_complex app;
	#pragma omp parallel private(app)
	{

		#pragma omp for
		for(j=0;j<hcdim;++j) {
			app=gsl_vector_complex_get(vk,j);
			in[j]=app.dat[0]+app.dat[1]*I; //C99 Complex
		} //#pragma omp for

	} //#pragma omp parallel

	/**
	 * IMPORTANT NOTE: In general, use the plan *before* initialisation of arrays
	 * For better performance use FFTW_MEASURE, but then you must put
	 * initialisation after plans. [That means v(k) data must be copied].
	 */

	/**
	 * Threaded FFTW initialisation [before allocations can give errors!!!].
	 */
	int	iret=fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	//c2r is always BACKWARD
	//fftw_plan fp = fftw_plan_dft_c2r_1d(N,in,out,FFTW_ESTIMATE);
	fftw_plan fp = fftw_plan_dft_c2r_1d(N,in,v->data,FFTW_ESTIMATE);
	//Calculate FFT
	fftw_execute(fp);
	fftw_destroy_plan(fp);

//	//Populate output
//	#pragma omp parallel
//	{
//
//		#pragma omp for
//		for(j=0;j<N;++j) {
//			gsl_vector_set(v,j,out[j]*norm);
//		} //#pragma omp for
//
//	} //#pragma omp parallel
	gsl_vector_scale(v,norm);

	//Destroy plan
	fftw_free(in);
	//fftw_free(out);
	fftw_cleanup_threads();

} //mylib_fft_c2r_omp_even_uncentred_1d



/**
 * @brief Shift the zero-frequency component to the centre of the image.
 *
 * This function swaps half-spaces for x-y axis. Note that v[0,0] is the Nyquist
 * component only if dim(v)=N is even [which is the case].
 *
 * @param N (square) matrix side dimension
 * @param v real square gsl_matrix of side N
 */
void mylib_fftshift_even_square_2d_dp(int N, gsl_matrix *square) {

  double temp;
  int x,y,halfN=N/2;

	#pragma omp parallel private(temp,x,y)
  {

		#pragma omp for collapse(2)
		for(x=0;x<halfN;++x) {
			for(y=0;y<halfN;++y) {
				temp=gsl_matrix_get(square,x,y);
				gsl_matrix_set(square,x,y,gsl_matrix_get(square,x+halfN,y+halfN));
				gsl_matrix_set(square,x+halfN,y+halfN,temp);
				temp=gsl_matrix_get(square,x,y+halfN);
				gsl_matrix_set(square,x,y+halfN,gsl_matrix_get(square,x+N/2,y));
				gsl_matrix_set(square,x+N/2,y,temp);
			}
		} //#pragma omp for

  } //#pragma omp parallel

} //mylib_fftshift_even_square_2d_dp



/**
 * @brief Shift the zero-frequency component to the centre of the complex image.
 *
 * This function swaps half-spaces for x-y axis. Note that v[0,0] is the Nyquist
 * component only if dim(v)=N is even [which is the case].
 *
 * @param N (square) matrix side dimension
 * @param v complex square gsl_matrix of side N
 */
void mylib_fftshift_even_square_2d_dpc(int N, gsl_matrix_complex *squarek) {

  gsl_complex temp;
  int x,y,halfN=N/2;

	#pragma omp parallel private(temp,x,y)
  {

		#pragma omp for collapse(2)
		for(x=0;x<halfN;++x) {
			for(y=0;y<halfN;++y) {
				temp=gsl_matrix_complex_get(squarek,x,y);
				gsl_matrix_complex_set(squarek,x,y,
						gsl_matrix_complex_get(squarek,x+halfN,y+halfN));
				gsl_matrix_complex_set(squarek,x+halfN,y+halfN,temp);
				temp=gsl_matrix_complex_get(squarek,x,y+halfN);
				gsl_matrix_complex_set(squarek,x,y+halfN,
						gsl_matrix_complex_get(squarek,x+N/2,y));
				gsl_matrix_complex_set(squarek,x+N/2,y,temp);
			}
		} //#pragma omp for

  } //#pragma omp parallel

} //mylib_fftshift_even_square_2d_dpc



/**
 * @brief 2D FFT Real2Complex, openMP parallel for uncentred EVEN dimension data.
 *
 * Note: this function use 1 more vector memory space.
 * Normalisation can be included in the definition of Delta (The standard assume
 * that all normalisation goes to c2r [BACKWARD transform]).
 *
 * @param N vector dimension
 * @param *patch double gsl_matrix of dimension N
 * @param L real grid dimension, put N if not known or not used
 *
 * @param *patchk OUTPUT complex gsl_matrix of dimension (N,N)
 */
void mylib_fft_r2c_omp_even_uncentred_2d(int N, gsl_matrix *patch,
		gsl_matrix_complex *patchk, double L) {

	double Delta=L/(double)N; //Sample Spacing
	//double Deltak=1./(double)N/Delta; //Sampling Rate
	double norm=Delta*Delta;
	int i,j;
	int hcdim=N/2+1; //half_complex dimension - EVEN
	int halfN=N/2;

  /**
   * FFTW requires aligned data stored in row major order (ALIGNED=flattened
   * multi-dimensional array).
   * Real and Imaginary parts of complex number stored as double[2].
   * Fortunately, GSL implements gsl_vector_complex in the required format.
   */

	double *in = patch->data;
	fftw_complex *out = fftw_alloc_complex(N*hcdim);

	/**
	 * IMPORTANT NOTE: In general, use the plan *before* initialisation of arrays
	 * For better performance use FFTW_MEASURE, but then you must put
	 * initialisation after plans. [That means v(k) data must be copied].
	 */

	/**
	 * Threaded FFTW initialisation [before allocations can give errors!!!].
	 */
	int	iret=fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	//r2c is always FORWARD
	fftw_plan fp = fftw_plan_dft_r2c_2d(N,N,in,out,FFTW_ESTIMATE);
	//Calculate FFT
	fftw_execute(fp);
	fftw_destroy_plan(fp);

	//Populate output
	#pragma omp parallel
	{

		#pragma omp for collapse(2)
		for(i=0;i<N;++i)
			for(j=0;j<halfN;++j) {
				gsl_matrix_complex_set(patchk,i,j,gsl_complex_mul_real(
					gsl_complex_rect(creal(out[MY2DIDX(i,j,hcdim)]),cimag(out[MY2DIDX(i,j,hcdim)])),norm));
				gsl_matrix_complex_set(patchk,i,j+halfN,gsl_complex_mul_real(
					gsl_complex_rect(creal(out[MY2DIDX(i,halfN-j,hcdim)]),-cimag(out[MY2DIDX(i,halfN-j,hcdim)])),norm));
			} //#pragma omp for

	} //#pragma omp parallel

	//Destroy plan
	//fftw_free(in); //PROBLEM with v->data?
	fftw_free(out);
	fftw_cleanup_threads();

}//mylib_fft_r2c_omp_even_uncentred_2d



/**
 * @brief 2D FFT Complex2Real, openMP parallel for uncentred EVEN dimension data.
 *
 * Note: this function use 1 more vector memory space.
 * Normalisation can be included in the definition of Delta (The standard assume
 * that all normalisation goes to c2r [BACKWARD transform]).
 *
 * @param N vector dimension
 * @param *patchk complex gsl_matrix of dimension (N,N)
 * @param L real grid dimension, put N if not known or not used
 *
 * @param *patch OUTPUT double gsl_matrix of dimension (N,N)
 */
void mylib_fft_c2r_omp_even_uncentred_2d(int N, gsl_matrix_complex *patchk,
		gsl_matrix *patch, double L) {

	double Delta=L/(double)N; //Sample Spacing
	double Deltak=1./(double)N/Delta; //Sampling Rate
	double norm=Deltak*Deltak; //REMEMBER: for FFTW we should normalise by N, so the real
	// normalisation is: norm=(N*Deltak)/N
	int i,j;
	int hcdim=N/2+1; //half_complex dimension - EVEN
	int halfN=N/2;

  /**
   * FFTW requires aligned data stored in row major order (ALIGNED=flattened
   * multi-dimensional array).
   * Real and Imaginary parts of complex number stored as double[2].
   * Fortunately, GSL implements gsl_vector_complex in the required format.
   */
	fftw_complex *in = fftw_alloc_complex(N*hcdim);

	//Initialise input vector (Half-Complex dimension)
	gsl_complex app;
	#pragma omp parallel private(app)
	{

		#pragma omp for collapse(2)
		for(i=0;i<N;++i)
			for(j=0;j<hcdim;++j) {
				app=gsl_matrix_complex_get(patchk,i,j);
				in[MY2DIDX(i,j,hcdim)]=app.dat[0]+app.dat[1]*I; //C99 Complex
			} //#pragma omp for

	} //#pragma omp parallel

	/**
	 * IMPORTANT NOTE: In general, use the plan *before* initialisation of arrays
	 * For better performance use FFTW_MEASURE, but then you must put
	 * initialisation after plans. [That means v(k) data must be copied].
	 */

	/**
	 * Threaded FFTW initialisation [before allocations can give errors!!!].
	 */
	int	iret=fftw_init_threads();
	fftw_plan_with_nthreads(omp_get_max_threads());
	//c2r is always BACKWARD
	//fftw_plan fp = fftw_plan_dft_c2r_1d(N,in,out,FFTW_ESTIMATE);
	fftw_plan fp = fftw_plan_dft_c2r_2d(N,N,in,patch->data,FFTW_ESTIMATE);
	//Calculate FFT
	fftw_execute(fp);
	fftw_destroy_plan(fp);

	//Populate output
	#pragma omp parallel
	{

		#pragma omp for collapse(2)
		for(i=0;i<N;++i)
			for(j=0;j<N;++j) {
				gsl_matrix_set(patch,i,j,gsl_matrix_get(patch,i,j)*norm);
			} //#pragma omp for

	} //#pragma omp parallel
	//gsl_matrix_scale(patch,norm);

	//Destroy plan
	fftw_free(in);
	//fftw_free(out);
	fftw_cleanup_threads();

}//mylib_fft_c2r_omp_even_uncentred_2d
