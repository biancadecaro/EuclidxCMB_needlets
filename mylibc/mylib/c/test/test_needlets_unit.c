#include <stdio.h>
#include <stdlib.h> //strtol

#include <gsl/gsl_statistics.h>

#include "mylibc.h"

int main(int nargs, const char* argv[]) {

  //int N = strtol(argv[1],NULL,10);
	mylib_print_needlets_parameters();

	FILE *f;
	char filename_j[MYFILENAMELEN];

	int N,jmin,jmax,lmax;
	double L,B,Delta,Deltak,freqmax,app;

	gsl_matrix *b_values;
  gsl_vector *v;
  gsl_vector_complex *vk;
  gsl_vector *vapp;

	N=8;
	jmax=6;
	L=0.5;
	Delta=L/(double)N;
	Deltak=1./(double)N/Delta;
	freqmax=N/2;//*Deltak;

	//printf("Max frequency value=%14.6e, Delta=%14.6e, Deltak=%14.6e\n",
	//		freqmax,Delta,Deltak);
	//B=mylib_jmax_xmax2B(jmax,freqmax);
	B=2.;
	//jmax=mylib_B_xmax2jmax(B,freqmax);
	//jmax=mylib_B_xmax2jmax(B,(double)N/2.);
	jmin=mylib_B_xmin2jmin(B,1.); //Always
	printf("jmin=%d,jmax=%d,freqmax=%e,B=%f\n",jmin,jmax,freqmax,B);

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/*BEGIN Standard needlets window functions - Harmonic/Fullsky version
*******************************************************************************/
/*
 	jmax=10;
 	lmax=50;
  B=mylib_jmax_lmax2B(jmax,lmax);
	printf("jmax=%d,lmax=%d,B=%f\n",jmax,lmax,B);

	b_values=gsl_matrix_calloc(jmax+1,lmax+1);
	mylib_needlets_std_init_b_values(b_values,B,jmax,lmax);

	f = fopen("b_values_matrix.txt", "w");
	//gsl_matrix_fprintf(stdout,b_values,"%14.6e");
  for(int j=0; j<=jmax; ++j) {
    for(int l=0; l<=lmax; ++l)
    	fprintf(f,"%14.6e ",gsl_matrix_get(b_values,j,l));
    fprintf(f,"\n");
  }
	fclose(f);
	mylib_needlets_check_windows(jmax,lmax,b_values);
	gsl_matrix_free(b_values);
*/
/*END Standard needlets window functions - Harmonic/Fullsky version
*******************************************************************************/

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/*BEGIN Standard needlets window functions - 1D-2D Flatsky version
*******************************************************************************/
/*
	b_values=gsl_matrix_calloc(jmax+1,N);
	mylib_needlets_std_init_b_values_even_square_uncentred_1d(b_values,B,jmax,N,N);
	f = fopen("b_values_matrix_1d_uncentred.txt", "w");
	//gsl_matrix_fprintf(stdout,b_values,"%14.6e");
	for(int j=0; j<=jmax; ++j) {
		for(int l=0; l<N; ++l)
			fprintf(f,"%14.6e ",gsl_matrix_get(b_values,j,l));
	  fprintf(f,"\n");
	}
	fclose(f);
	mylib_needlets_check_windows(jmax,N-1,b_values);
	gsl_matrix_free(b_values);

*/

/*
  gsl_matrix **b_values_2D=malloc((jmax+1)*sizeof(gsl_matrix *));
  for(int j=0;j<=jmax;++j) {
  	b_values_2D[j]=gsl_matrix_calloc(N,N);
  }

  mylib_needlets_std_init_b_values_even_square_uncentred_2d(b_values_2D,B,jmax,N,N);

  for(int j=0;j<=jmax;++j) {
  	char filename_j[MYFILENAMELEN];
  	snprintf(filename_j,MYFILENAMELEN,"b_values_2D_matrix_j%02d.txt",j);
  	printf("filename=%s\n",filename_j);
  	f = fopen(filename_j, "w");
  	//gsl_matrix_fprintf(stdout,b_values_2D[j],"%14.6e");
  	for(int p=0;p<N;++p) {
  		for(int q=0;q<N;++q)
  			fprintf(f,"%14.6e",gsl_matrix_get(b_values_2D[j],p,q));
  		fprintf(f,"\n");
  	}
   fclose(f);
  }

  for(int j=0;j<=jmax;++j) {
  	gsl_matrix_free(b_values_2D[j]);
  }
  free(b_values_2D);
*/
/*END Standard needlets window functions - 1D-2D Flatsky version
*******************************************************************************/

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/*BEGIN kvalues and FFTSHIFT, 1D and 2D version
*******************************************************************************/
/*
  gsl_vector *kvec=gsl_vector_alloc(N);
  gsl_vector *kvec_ord=gsl_vector_alloc(N);
  mylib_fftfreq_even_uncentred_1d(N,kvec,L);
  gsl_vector_memcpy(kvec_ord,kvec);
  mylib_fftshift_even_1d_dp(N,kvec_ord);

//  for(int i=0;i<N;++i) {
//  	printf("kvec=%e,kvec_ordered=%e\n",gsl_vector_get(kvec,i),
//  			gsl_vector_get(kvec_ord,i));
//  }
*/

/*
	gsl_matrix *kmat=gsl_matrix_alloc(N,N);
  gsl_matrix *kmat_ord=gsl_matrix_alloc(N,N);
  mylib_fftfreq_even_square_uncentred_2d(N,kmat,L);
  gsl_matrix_memcpy(kmat_ord,kmat);
  mylib_fftshift_even_square_2d_dp(N,kmat_ord);

  for(int i=0;i<N;++i) {
  	for(int j=0;j<N;++j) {
  		printf("i=%d,j=%d,kmat=%e,kmat_ordered=%e\n",i,j,gsl_matrix_get(kmat,i,j),
  				gsl_matrix_get(kmat_ord,i,j));
  	}
  }

*/

/*END kvalues and FFTSHIFT, 1D and 2D version
*******************************************************************************/

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/*BEGIN Back and Fort FFT test
*******************************************************************************/
/*

  v=gsl_vector_calloc(N);
  vk=gsl_vector_complex_calloc(N);

  for(int i=0;i<N;++i) {
  	gsl_vector_set(v,i,(double)i+1.);
  }

	gsl_vector_fprintf(stdout,v,"%14.6e");
	mylib_fft_r2c_omp_even_uncentred_1d(N,v,vk,L);

  printf("OUTPUT\n");
  gsl_vector_complex_fprintf(stdout,vk,"%14.6e");
  mylib_fftshift_even_1d_dpc(N,vk);
  printf("ORDERED OUTPUT\n");
  gsl_vector_complex_fprintf(stdout,vk,"%14.6e");

  //Test FFT forward and backward transform
  vapp=gsl_vector_alloc(N);
  for(int i=0;i<N;++i) {
  	gsl_vector_set(v,i,(double)i+1.);
  	gsl_vector_set(vapp,i,(double)i+1.);
  }
  mylib_fft_r2c_omp_even_uncentred_1d(N,v,vk,L);
  mylib_fft_c2r_omp_even_uncentred_1d(N,vk,v,L);
	gsl_vector_sub(vapp,v);
	printf("Deve fare zero!\n");
  gsl_vector_fprintf(stdout,vapp,"%14.6e");

  gsl_vector_free(v);
  gsl_vector_complex_free(vk);
  gsl_vector_free(vapp);

*/

/*
	gsl_matrix *m=gsl_matrix_calloc(N,N);
  gsl_matrix_complex *ak=gsl_matrix_complex_alloc(N,N);

	double count=1.;
	for(int i=0;i<N;++i)
		for(int j=0;j<N;++j) {
			gsl_matrix_set(m,i,j,count);
			++count;
		}

	gsl_matrix_fprintf(stdout,m,"%14.6e");
	mylib_fft_r2c_omp_even_uncentred_2d(N,m,ak,L);

  printf("OUTPUT\n");
  gsl_matrix_complex_fprintf(stdout,ak,"%14.6e");
  mylib_fftshift_even_square_2d_dpc(N,ak);
  printf("ORDERED OUTPUT\n");
  gsl_matrix_complex_fprintf(stdout,ak,"%14.6e");

  //Test FFT forward and backward transform
  gsl_matrix *mapp=gsl_matrix_alloc(N,N);

  count=1.;
  for(int i=0;i<N;++i)
  	for(int j=0;j<N;++j) {
  		gsl_matrix_set(m,i,j,count);
  		gsl_matrix_set(mapp,i,j,count);
  	}

  mylib_fft_r2c_omp_even_uncentred_2d(N,m,ak,L);
  mylib_fft_c2r_omp_even_uncentred_2d(N,ak,m,L);
	gsl_matrix_sub(mapp,m);
	printf("Deve fare zero!\n");
  gsl_matrix_fprintf(stdout,mapp,"%14.6e");

  gsl_matrix_free(m);
  gsl_matrix_complex_free(ak);
  gsl_matrix_free(mapp);

*/

/*END Back and Fort FFT test
*******************************************************************************/

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/*BEGIN 1D Needlets Reconstruction test
*******************************************************************************/
/*
//	B=mylib_jmax_xmax2B(jmax,freqmax);
	//jmax=5;
	//jmax=mylib_B_xmax2jmax(B,N/2.);
	//printf("Max frequency value=%14.6e, jmax=%d, B=%f\n",freqmax,jmax,B);

  v=gsl_vector_calloc(N);
  gsl_matrix *betajk=gsl_matrix_calloc(jmax+1,N);
  gsl_vector *betajk_j=gsl_vector_alloc(N);

  for(int i=0;i<N;++i) {
  	gsl_vector_set(v,i,(double)i+1.);
  }

  mylib_needlets_f2betajk_omp_even_uncentred_1d(N,v,betajk,B,jmax,N);

  f = fopen("betajk_matrix.txt", "w");
  for(int j=0; j<=jmax; ++j) {
  	gsl_matrix_get_row(betajk_j,betajk,j);
  	gsl_vector_fprintf(f,betajk_j,"%14.6e");
    //for(int l=0; l<N; ++l)
    //	fprintf(f,"%14.6e ",gsl_matrix_get(betajk,j,l));
    fprintf(f,"\n");
  }
	fclose(f);

	vapp=gsl_vector_calloc(N);
	mylib_needlets_betajk2f_omp_even_uncentred_1d(N,betajk,vapp,B,jmax,N);

	printf("v:\n");
	gsl_vector_fprintf(stdout,v,"%14.6e");
	printf("\nvapp:\n");
	gsl_vector_fprintf(stdout,vapp,"%14.6e");
	printf("\n");
	gsl_vector_sub(vapp,v);
	printf("Deve fare zero!\n");
	gsl_vector_fprintf(stdout,vapp,"%14.6e");
*/
/*END 1D Needlets Reconstruction test
*******************************************************************************/

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/*BEGIN 2D Needlets Reconstruction test
*******************************************************************************/
/*
//	B=mylib_jmax_xmax2B(jmax,freqmax);
	//jmax=5;
	//jmax=mylib_B_xmax2jmax(B,N/2.);
	//printf("Max frequency value=%14.6e, jmax=%d, B=%f\n",freqmax,jmax,B);

	gsl_matrix *m=gsl_matrix_calloc(N,N);
  gsl_matrix **betajk=malloc((jmax+1)*sizeof(gsl_matrix *));
  for(int j=0;j<=jmax;++j) {
  	betajk[j]=gsl_matrix_calloc(N,N);
  }
	gsl_matrix *betajk_j=gsl_matrix_alloc(N,N);

	double count=1.;
	for(int i=0;i<N;++i)
		for(int j=0;j<N;++j) {
			gsl_matrix_set(m,i,j,count);
			++count;
		}

	mylib_needlets_f2betajk_omp_even_uncentred_2d(N,m,betajk,B,jmax,N);

	f = fopen("betajk_matrix.txt", "w");
	for(int j=0; j<=jmax; ++j) {
		snprintf(filename_j,MYFILENAMELEN,"betajk_values_2D_matrix_j%02d.txt",j);
		//printf("filename=%s\n",filename_j);
		f = fopen(filename_j, "w");
		//gsl_matrix_get_row(betajk_j,betajk,j);
		gsl_matrix_memcpy(betajk_j,betajk[j]);
		//gsl_matrix_fprintf(f,betajk_j,"%14.6e");
		for(int x=0;x<N;++x) {
			for(int y=0;y<N;++y) {
				fprintf(f,"%14.6e ",gsl_matrix_get(betajk_j,x,y));
			}
			fprintf(f,"\n");
		}
		//for(int l=0; l<N; ++l)
		//	fprintf(f,"%14.6e ",gsl_matrix_get(betajk,j,l));
		fprintf(f,"\n");
		fclose(f);
	}

	gsl_matrix *mapp=gsl_matrix_calloc(N,N);
	mylib_needlets_betajk2f_omp_even_uncentred_2d(N,betajk,mapp,B,jmax,N);

	printf("m:\n");
	gsl_matrix_fprintf(stdout,m,"%14.6e");
	printf("\nmapp:\n");
	gsl_matrix_fprintf(stdout,mapp,"%14.6e");
	printf("\n");
	gsl_matrix_sub(mapp,m);
	printf("Deve fare zero!\n");
	gsl_matrix_fprintf(stdout,mapp,"%14.6e");
*/
/*END 2D Needlets Reconstruction test
*******************************************************************************/

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
/*BEGIN 2D Needlets Reconstruction test
*******************************************************************************/

//	B=mylib_jmax_xmax2B(jmax,freqmax);
	//jmax=5;
	//jmax=mylib_B_xmax2jmax(B,N/2.);
	//printf("Max frequency value=%14.6e, jmax=%d, B=%f\n",freqmax,jmax,B);

	N=1024;
	B=2.;
	jmax=10;

	gsl_matrix *m=gsl_matrix_calloc(N,N);
	gsl_matrix **betajk=malloc((jmax+1)*sizeof(gsl_matrix *));
	for(int j=0;j<=jmax;++j) {
		betajk[j]=gsl_matrix_calloc(N,N);
	}
	gsl_matrix *betajk_j=gsl_matrix_alloc(N,N);

	f=fopen("fisica_solare.txt","r");
	for(int i=0;i<N;++i) {
		for(int j=0;j<N;++j) {
			if (!fscanf(f, "%lf", &app))
				break;
			else
				gsl_matrix_set(m,i,j,app);
			//printf("%lf\n",app);
		}
	}
	fclose(f);

	double mean=gsl_stats_mean(m->data,1,N*N);
	printf("Mean Value=%e\n",mean);
	gsl_matrix_add_constant(m,-mean);

	mylib_needlets_f2betajk_omp_even_uncentred_2d(N,m,betajk,B,jmax,N);

	f = fopen("betajk_matrix.txt", "w");
	for(int j=0; j<=jmax; ++j) {
		snprintf(filename_j,MYFILENAMELEN,"betajk_values_2D_matrix_j%02d.txt",j);
		//printf("filename=%s\n",filename_j);
		f = fopen(filename_j, "w");
		//gsl_matrix_get_row(betajk_j,betajk,j);
		gsl_matrix_memcpy(betajk_j,betajk[j]);
		//gsl_matrix_fprintf(f,betajk_j,"%14.6e");
		for(int x=0;x<N;++x) {
			for(int y=0;y<N;++y) {
				fprintf(f,"%14.6e ",gsl_matrix_get(betajk_j,x,y));
			}
			fprintf(f,"\n");
		}
		//for(int l=0; l<N; ++l)
		//	fprintf(f,"%14.6e ",gsl_matrix_get(betajk,j,l));
		fprintf(f,"\n");
		fclose(f);
	}

	gsl_matrix *mapp=gsl_matrix_calloc(N,N);
	mylib_needlets_betajk2f_omp_even_uncentred_2d(N,betajk,mapp,B,jmax,N);

	printf("m:\n");
	//gsl_matrix_fprintf(stdout,m,"%14.6e");
	printf("\nmapp:\n");
	//gsl_matrix_fprintf(stdout,mapp,"%14.6e");
	printf("\n");
	gsl_matrix_sub(mapp,m);

	printf("Deve fare zero!\n");
	//gsl_matrix_fprintf(stdout,mapp,"%14.6e");
	for(int x=0;x<5;++x) {
		for(int y=0;y<5;++y) {
			printf("%14.6e\n",gsl_matrix_get(mapp,x,y));
		}
	}

/*END 2D Needlets Reconstruction test
*******************************************************************************/

  return 0;

}
