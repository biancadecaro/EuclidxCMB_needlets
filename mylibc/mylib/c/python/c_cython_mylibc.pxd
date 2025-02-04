cdef extern from "gsl/gsl_block.h":

    ctypedef struct gsl_block:
        size_t size
        double * data

cdef extern from "gsl/gsl_matrix.h" nogil:

    ctypedef struct gsl_matrix:
        size_t size1
        size_t size2
        size_t tda
        double * data
        gsl_block * block
        int owner

    gsl_matrix *gsl_matrix_alloc (int n1, int n2)
    gsl_matrix *gsl_matrix_calloc (int n1, int n2)
    void gsl_matrix_set_zero (gsl_matrix * m)
    void gsl_matrix_free (gsl_matrix * m)
    int gsl_matrix_add_constant (gsl_matrix * a, const double x)

cdef inline double gsl_matrix_get(gsl_matrix *m, int i, int j) nogil:
    return m.data[i*m.tda+j]
    
cdef inline void gsl_matrix_set(gsl_matrix *m, int i, int j, double x) nogil:
    m.data[i*m.tda+j] = x

cdef extern from "gsl/gsl_vector.h" nogil:

    ctypedef struct gsl_vector:
        size_t size;
        size_t stride;
        double * data;
        gsl_block * block;
        int owner;

    gsl_vector *gsl_vector_alloc (int n)
    gsl_vector *gsl_vector_calloc (int n)
    void gsl_vector_set_zero (gsl_vector * v)
    void gsl_vector_free (gsl_vector * v)

cdef inline double gsl_vector_get(gsl_vector *v, int i) nogil:
    return v.data[i]
    
cdef inline void gsl_vector_set(gsl_vector *v, int i, double x) nogil:
    v.data[i] = x
    
cdef extern from "mylibc.h" nogil:
    void mylib_print_needlets_parameters();
    double mylib_jmax_lmax2B(int jmax,int lmax);
    double mylib_jmax_xmax2B(int jmax,double xmax);
#    void mylib_fftshift_even_square_2d_dp(int N, gsl_matrix *square);
    void mylib_needlets_std_init_b_values_harmonic(gsl_matrix *b_values,
            double B,int jmax,int lmax);
    void mylib_needlets_check_windows(int jmax,int lmax,gsl_matrix *b_values);		
#    void mylib_needlets_std_init_b_values_even_square_uncentred_1d(gsl_matrix *b_values,double B,int jmax,int N, double L);
#    void mylib_needlets_f2betajk_omp_even_uncentred_1d(int N, gsl_vector *func,gsl_matrix *betajk,double B,int jmax, double L);
#    void mylib_needlets_betajk2f_omp_even_uncentred_1d(int N, gsl_matrix *betajk,gsl_vector *func, double B, int jmax, double L);
#    void mylib_needlets_std_init_b_values_even_square_uncentred_2d(gsl_matrix **b_values,double B,int jmax,int N, double L);
#    void mylib_needlets_f2betajk_omp_even_uncentred_2d(int N, gsl_matrix *square,gsl_matrix **betajk,double B,int jmax, double L);
#    void mylib_needlets_betajk2f_omp_even_uncentred_2d(int N, gsl_matrix **betajk,gsl_matrix *square, double B, int jmax, double L);

