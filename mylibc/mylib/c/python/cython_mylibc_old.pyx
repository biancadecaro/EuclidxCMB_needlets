#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from math import exp,log
import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport *
cimport cython

ctypedef np.double_t DTYPE_t
ctypedef np.int_t DTYPE_i

# Import the .pxd containing definitions
from c_cython_mylibc cimport *

#TEST:parallel openmp cython
from cython.parallel import prange

#TEST: faster cython
#boundscheck(False)

cdef gsl_matrix* pyarray2gslmatrix(DTYPE_t[:,:] a):
#cdef gsl_matrix* pyarray2gslmatrix(np.ndarray[DTYPE_t,ndim=2] a):
#cdef gsl_matrix* pyarray2gslmatrix(object[DTYPE_t,ndim=2] a):
    cdef Py_ssize_t i1,i2
    cdef gsl_matrix* m
    cdef Py_ssize_t n1 = a.shape[0]
    cdef Py_ssize_t n2 = a.shape[1]
    m = gsl_matrix_alloc(n1,n2)
    for i1 from 0<=i1<n1:
        for i2 from 0<=i2<n2: 
            gsl_matrix_set(m,i1,i2,a[i1,i2])
    return m
    
cdef gsl_matrix** pyarray2carrayofgslmatrix(DTYPE_t[:,:,:] a):
#cdef gsl_matrix** pyarray2carrayofgslmatrix(np.ndarray[DTYPE_t,ndim=3] a):
#cdef gsl_matrix** pyarray2carrayofgslmatrix(object[DTYPE_t,ndim=3] a):
    cdef Py_ssize_t i1,i2,i3
    cdef gsl_matrix** m
    cdef Py_ssize_t n1 = a.shape[0]
    cdef Py_ssize_t n2 = a.shape[1]
    cdef Py_ssize_t n3 = a.shape[2]

    m=<gsl_matrix **>malloc(n1*sizeof(gsl_matrix *))
    for i from 0<=i<n1:
        m[i] = gsl_matrix_alloc(n2,n3)

    for i1 from 0<=i1<n1:
        for i2 from 0<=i2<n2:
            for i3 from 0<=i3<n3: 
                gsl_matrix_set(m[i1],i2,i3,a[i1,i2,i3])
    return m

cdef DTYPE_t[:,:] gslmatrix2pyarray(gsl_matrix* m):
#cdef np.ndarray[DTYPE_t, ndim=2] gslmatrix2pyarray(gsl_matrix* m):
#cdef object[DTYPE_t, ndim=2] gslmatrix2pyarray(gsl_matrix* m):
    """ copies contents of m into numeric array """
    cdef Py_ssize_t i1, i2
    cdef DTYPE_t[:,:] ans 
    #cdef np.ndarray[DTYPE_t, ndim=2] ans 
    #cdef object[DTYPE_t, ndim=2] ans 
    ans = np.empty((m.size1, m.size2), np.double)
    for i1 in prange(m.size1,nogil=True): 
        for i2 in prange(m.size2,nogil=False):
            ans[i1, i2] = gsl_matrix_get(m, i1, i2)
    return ans

cdef DTYPE_t[:,:,:] carrayofgslmatrix2pyarray(int jmax, gsl_matrix** m):
#cdef np.ndarray[DTYPE_t, ndim=3] carrayofgslmatrix2pyarray(int jmax, gsl_matrix** m):
#cdef object[DTYPE_t, ndim=3] carrayofgslmatrix2pyarray(int jmax, gsl_matrix** m):
    cdef Py_ssize_t i1,i2
    cdef DTYPE_t[:,:,:] ans 
    #cdef np.ndarray[DTYPE_t, ndim=3] ans 
    #cdef object[DTYPE_t, ndim=3] ans 
    ans = np.empty((jmax+1, m[0].size1, m[0].size2), np.double)
    for j in np.arange(jmax+1):
        for i1 in np.arange(m[0].size1):
            for i2 in np.arange(m[0].size2):
                ans[j,i1, i2] = gsl_matrix_get(m[j], i1, i2)
    return ans

cdef gsl_vector* pyarray2gslvector(DTYPE_t[:] a):
#cdef gsl_vector* pyarray2gslvector(np.ndarray[DTYPE_t,ndim=1] a):
#cdef gsl_vector* pyarray2gslvector(object[DTYPE_t,ndim=1] a):
    cdef Py_ssize_t i
    cdef gsl_vector* v
    cdef Py_ssize_t n = a.shape[0]
    v = gsl_vector_alloc(n)
    for i from 0<=i<n:
        gsl_vector_set(v,i,a[i])
    return v
    
cdef DTYPE_t[:] gslvector2pyarray(gsl_vector* v):
#cdef np.ndarray[DTYPE_t, ndim=1] gslvector2pyarray(gsl_vector* v):
#cdef object[DTYPE_t, ndim=1] gslvector2pyarray(gsl_vector* v):
    """ copies contents of v into numeric array """
    cdef Py_ssize_t i
    cdef DTYPE_t[:] ans 
    #cdef np.ndarray[DTYPE_t, ndim=1] ans
    #cdef object[DTYPE_t, ndim=1] ans
    ans = np.empty((v.size), np.double)
    for i in prange(v.size,nogil=True):
        ans[i] = gsl_vector_get(v, i)
    return ans

cdef class mylibc:
    def debug_needlets(self):
        """
        debug_needlets()

        Print needlets precision parameters.

        Parameters
        ----------

        Returns
        -------
        """
        mylib_print_needlets_parameters()
        return

    def mylib_jmax_lmax2B(self,jmax,lmax):
        """
        mylib_jmax_lmax2B(jmax,lmax)

        Calculate the parameter B given lmax and jmax

        This function calculate the value of the needlets width parameter B that 
        center the *last* needlets windows function b(l/B**j) on lmax.

        Parameters
        ----------
        jmax: int 
            Maximum needlets frequency
        lmax: int
            The experiments maximum harmonic value

        Returns
        -------
        B: float
            Needlet width parameter
        """
        return mylib_jmax_lmax2B(jmax,lmax)


    def mylib_jmax_xmax2B(self,jmax,xmax):
        """
        mylib_jmax_lmax2B(jmax,xmax)
        
        Calculate the parameter B given xmax and jmax

        This function calculate the value of the needlets width parameter B that
        center the *last* needlets windows function b(x/B**j) on xmax. In contrast to
        the case of mylib_jmax_lmax2B now xmax could be any double precision number
        greater than zero that represent the maximum value in the needlets windows
        function argument (for example freqmax in case of 1D fourier transform).

        Parameters
        ----------
        jmax: int 
            Maximum needlets frequency
        xmax: float
            The needlets maximum argument value

        Returns
        -------
        B: float
            Needlets width parameter
        """
        return mylib_jmax_xmax2B(jmax,xmax)

    def mylib_needlets_std_init_b_values(self,B,jmax,lmax):
        """
        mylib_needlets_check_windows(jmax,lmax,b_values)
        
        This function check if the 1D needlets window functions are correct. [It checks if \sum_j b(l/B**j)**2=1 for l>0]

        Parameters
        ----------
        B: double
            Needlets width parameter
        jmax: int
            Max frequency (j=0,...,jmax)
        lmax: int
            Max harmonic parameter

        Returns
        ----------
        b_values: [0:jmax,0:lmax]
            Needlets window functions
        """
        #c_b_values=array2matrix(py_b_values)
        cdef gsl_matrix* m
        cdef Py_ssize_t n1 = jmax+1
        cdef Py_ssize_t n2 = lmax+1
        m = gsl_matrix_calloc(n1,n2)

        mylib_needlets_std_init_b_values_harmonic(m,B,jmax,lmax)
        py_b_values=gslmatrix2pyarray(m)
        gsl_matrix_free(m)
        return py_b_values

#    def mylib_needlets_check_windows(self,jmax,lmax,DTYPE_t[:,:] py_b_values):
#    def mylib_needlets_check_windows(self,jmax,lmax,np.ndarray[DTYPE_t, ndim=2] py_b_values):
    def mylib_needlets_check_windows(self,jmax,lmax,object[DTYPE_t, ndim=2] py_b_values):
        """
        mylib_needlets_check_windows(jmax,lmax,b_values)
        
        This function check if the 1D needlets window functions are correct. [It checks if \sum_j b(l/B**j)**2=1 for l>0]

        Parameters
        ----------
        jmax: int
            Max frequency (j=0,...,jmax)
        lmax: int
            Max harmonic parameter

        Returns
        ----------
        """
        c_b_values=pyarray2gslmatrix(py_b_values)
        mylib_needlets_check_windows(jmax,lmax,c_b_values)
        gsl_matrix_free(c_b_values)

    def mylib_needlets_std_init_b_values_even_square_uncentred_1d(self,B,jmax,N,L):
        """
        mylib_needlets_std_init_b_values_even_square_uncentred_1d(B,jmax,N,L)
        
        Standard needlets windows functions 1D: array of gsl_matrix [FFT version] 

        This function return the 1D standard needlets functions, in form of ndarray matrix
        with dimension (jmax+1,N) (N is even).
 
        Parameters
        ----------
        B: double
            Needlets width parameter
        jmax: int
            Max frequency (j=0,...,jmax)
        N: int
            Number of points (sampling)
        L: double
            physical dimension of the 1D signal - Not needed? 

        Returns
        ----------
        b_values: [0:jmax,0:N-1]
            Needlets window functions
        """
        cdef gsl_matrix* m
        cdef Py_ssize_t n1 = jmax+1
        cdef Py_ssize_t n2 = N
        m = gsl_matrix_calloc(n1,n2)

        mylib_needlets_std_init_b_values_even_square_uncentred_1d(m,B,jmax,N,L)

        py_b_values=gslmatrix2pyarray(m)
        gsl_matrix_free(m)
        return py_b_values

#    def mylib_needlets_f2betajk_omp_even_uncentred_1d(self,DTYPE_t[:] func,B,jmax,L):
#    def mylib_needlets_f2betajk_omp_even_uncentred_1d(self,np.ndarray[DTYPE_t, ndim=1] func,B,jmax,L):
    def mylib_needlets_f2betajk_omp_even_uncentred_1d(self,object[DTYPE_t, ndim=1] func,B,jmax,L):
        """
        mylib_needlets_f2betajk_omp_even_uncentred_1d(func,B,jmax,L)
        
        Standard needlets transform for functions 1D [FFT version]

        This function return the coefficients of the standard needlets transform of a function func.
 
        Parameters
        ----------
        func: [0:N-1]
            Sampled signal array
        B: double
            Needlets width parameter
        jmax: int
            Max frequency (j=0,...,jmax)
        L: double
            physical dimension of the 1D signal - Not needed? 

        Returns
        ----------
        betajk: [0:jmax,0:N-1]
            Needlets coefficients of the input signal
        """

        N=len(func)
        v = pyarray2gslvector(func)

        cdef gsl_matrix* m
        cdef Py_ssize_t n1 = jmax+1
        cdef Py_ssize_t n2 = N
        m = gsl_matrix_alloc(n1,n2)

        mylib_needlets_f2betajk_omp_even_uncentred_1d(N,v,m,B,jmax,L)

        py_betajk=gslmatrix2pyarray(m)
        gsl_matrix_free(m)
        gsl_vector_free(v)
        return py_betajk

#    def mylib_needlets_betajk2f_omp_even_uncentred_1d(self,DTYPE_t[:,:] betajk,B,L):
#    def mylib_needlets_betajk2f_omp_even_uncentred_1d(self,np.ndarray[DTYPE_t, ndim=2] betajk,B,L):
    def mylib_needlets_betajk2f_omp_even_uncentred_1d(self,object[DTYPE_t, ndim=2] betajk,B,L):
        """
        mylib_needlets_f2betajk_omp_even_uncentred_1d(func,B,L)
        
        Standard needlets transform for functions 1D [FFT version]

        This function return the coefficients of the standard needlets transform of a function func.
 
        Parameters
        ----------
        betajk: [0:jmax,0:N-1]
            Needlets coefficients of the input signal
        B: double
            Needlets width parameter
        L: double
            physical dimension of the 1D signal - Not needed? 

        Returns
        ----------
        func: [0:N-1]
            Sampled signal array
        """

        N=len(betajk[0,:])
        jmax=len(betajk[:,0])-1
        #print("N={:d}, jmax={:d}".format(N,jmax))
        m = pyarray2gslmatrix(betajk)

        cdef gsl_vector* v
        cdef Py_ssize_t n = N
        v = gsl_vector_alloc(n)

        mylib_needlets_betajk2f_omp_even_uncentred_1d(N,m,v,B,jmax,L)

        py_func=gslvector2pyarray(v)
        gsl_matrix_free(m)
        gsl_vector_free(v)
        return py_func

#    def mylib_fftshift_even_square_2d_dp(self,N,DTYPE_t[:,:] pysquare):
#Faster but numpy array only
#    def mylib_fftshift_even_square_2d_dp(self,N,np.ndarray[DTYPE_t, ndim=2] pysquare):
#Slower but generic, python3 only
    def mylib_fftshift_even_square_2d_dp(self,N,object[DTYPE_t, ndim=2] pysquare):
        """
        mylib_fftshift_even_2d_dp(N,square)
        
        This function shift (FFT compatible) the (real valued) square patch

        Parameters
        ----------
        N: int (even)
            Square side number dimension
        square: float (NxN)
            A real valued square patch

        Returns
        -------
        square: float (NxN)
            The square patch is shifted (FFT compatible)
        """
        #To be tested!!!:
        #mylib_fftshift_even_2d_dp(N,<gsl_matrix *> pysquare.data);
        csquare=pyarray2gslmatrix(pysquare)
        mylib_fftshift_even_square_2d_dp(N, csquare)
        pysquare=gslmatrix2pyarray(csquare)
        gsl_matrix_free(csquare)
        return pysquare

    def mylib_needlets_std_init_b_values_even_square_uncentred_2d(self,B,jmax,N,L):
        """
        mylib_needlets_std_init_b_values_even_square_uncentred_1d(B,jmax,N,L)
        
        Standard needlets windows functions 2D: array of gsl_matrix
 
        This function return the 2D standard needlets functions, in form of array of gsl_matrix,
        any matrix is a (N,N) patch for a frequency j (N is even).

        Parameters
        ----------
        B: double
            Needlets width parameter
        jmax: int
            Max frequency (j=0,...,jmax)
        N: int
            Number of grid side points
        L: double
            physical dimension of the grid - Not needed? 

        Returns
        ----------
        b_values: [0:jmax,0:N-1,0:N-1]
            Needlets window functions
        """
        cdef gsl_matrix** m
        cdef Py_ssize_t n1 = jmax+1
        cdef Py_ssize_t n2 = N
        m=<gsl_matrix **>malloc(n1*sizeof(gsl_matrix *))
        for i from 0<=i<=jmax:
            m[i] = gsl_matrix_calloc(n2,n2)

        mylib_needlets_std_init_b_values_even_square_uncentred_2d(m,B,jmax,N,L)

        py_b_values=carrayofgslmatrix2pyarray(jmax,m)
        for i from 0<=i<=jmax:
            gsl_matrix_free(m[i])
        free(m)
        return py_b_values

#    def mylib_needlets_f2betajk_omp_even_uncentred_2d(self,DTYPE_t[:,:] func,B,jmax,L):
#    def mylib_needlets_f2betajk_omp_even_uncentred_2d(self,np.ndarray[DTYPE_t, ndim=2] func,B,jmax,L):
    def mylib_needlets_f2betajk_omp_even_uncentred_2d(self,object[DTYPE_t, ndim=2] func,B,jmax,L):
        """
        mylib_needlets_f2betajk_omp_even_uncentred_2d(func,B,jmax,L)
        
        Standard needlets transform for functions 2D [FFT version]

        This function return the coefficients of the standard needlets transform of a function func.
 
        Parameters
        ----------
        func: [0:N-1,0:N-1]
            Sampled signal array
        B: double
            Needlets width parameter
        jmax: int
            Max frequency (j=0,...,jmax)
        L: double
            physical dimension of the grid - Not needed? 

        Returns
        ----------
        betajk: [0:jmax,0:N-1,0:N-1]
            Needlets coefficients of the input signal
        """

        N=len(func[0,:])
        square = pyarray2gslmatrix(func)

        cdef gsl_matrix** betajk
        cdef Py_ssize_t n1 = jmax+1
        cdef Py_ssize_t n2 = N
        betajk=<gsl_matrix **>malloc(n1*sizeof(gsl_matrix *))
        for i from 0<=i<=jmax:
            betajk[i] = gsl_matrix_calloc(n2,n2)

        mylib_needlets_f2betajk_omp_even_uncentred_2d(N,square,betajk,B,jmax,L)

        py_betajk=carrayofgslmatrix2pyarray(jmax,betajk)
        for i from 0<=i<=jmax:
            gsl_matrix_free(betajk[i])
        free(betajk)
        gsl_matrix_free(square)
        return py_betajk

#    def mylib_needlets_betajk2f_omp_even_uncentred_2d(self,DTYPE_t[:,:,:] betajk,B,L):
#    def mylib_needlets_betajk2f_omp_even_uncentred_2d(self,np.ndarray[DTYPE_t, ndim=3] betajk,B,L):
    def mylib_needlets_betajk2f_omp_even_uncentred_2d(self,object[DTYPE_t, ndim=3] betajk,B,L):
        """
        mylib_needlets_f2betajk_omp_even_uncentred_2d(func,B,L)
        
        Standard needlets transform for functions 2D [FFT version]

        This function return the coefficients of the standard needlets transform of a function func.
 
        Parameters
        ----------
        betajk: [0:jmax,0:N-1,0:N-1]
            Needlets coefficients of the input signal
        B: double
            Needlets width parameter
        L: double
            physical dimension of the grid - Not needed? 

        Returns
        ----------
        func: [0:N-1,0:N-1]
            Sampled signal array
        """

        N=len(betajk[0,0,:])
        jmax=len(betajk[:,0,0])-1
        m = pyarray2carrayofgslmatrix(betajk)
        print("N={:d}, jmax={:d}".format(N,jmax))

        cdef gsl_matrix* square
        cdef Py_ssize_t n = N
        square = gsl_matrix_alloc(N,N)

        mylib_needlets_betajk2f_omp_even_uncentred_2d(N,m,square,B,jmax,L)

        py_square=gslmatrix2pyarray(square)
        for i from 0<=i<=jmax:
            gsl_matrix_free(m[i])
        free(m)
        gsl_matrix_free(square)
        return py_square


