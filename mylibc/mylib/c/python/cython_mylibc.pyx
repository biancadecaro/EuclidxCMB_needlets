from math import exp,log
import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport *
cimport cython

ctypedef np.double_t cDOUBLE
DOUBLE = np.float64

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
# ATTENTION: if I use this line than anything below will not be showed by the module in the python program!
#np.import_array()


# Import the .pxd containing definitions
from c_cython_mylibc cimport *

#TEST:parallel openmp cython
from cython.parallel import parallel,prange

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef gsl_matrix* pyarray2gslmatrix(cDOUBLE[:,:] a):
    cdef gsl_matrix* m
    cdef int i1,i2,n1,n2
    n1=a.shape[0]
    n2=a.shape[1]
    m = gsl_matrix_alloc(n1,n2)
    for i1 in prange(n1,nogil=True):
        for i2 in prange(n2):
            gsl_matrix_set(m,i1,i2,a[i1,i2])
    return m
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef gsl_matrix** pyarray2carrayofgslmatrix(cDOUBLE[:,:,:] a):
    cdef gsl_matrix** m
    cdef int i1,i2,i3,n1,n2,n3
    n1=a.shape[0]
    n2=a.shape[1]
    n3=a.shape[2]

    m=<gsl_matrix **>malloc(n1*sizeof(gsl_matrix *))

    for i1 in prange(n1,nogil=True):
        m[i1] = gsl_matrix_alloc(n2,n3)
        for i2 in prange(n2):
            for i3 in prange(n3):
                gsl_matrix_set(m[i1],i2,i3,a[i1,i2,i3])
    return m

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef cDOUBLE[:,:] gslmatrix2pyarray(gsl_matrix* m):
    """ copies contents of m into numeric array """
    cdef int i1,i2,n1,n2
    n1=m.size1
    n2=m.size2
    cdef cDOUBLE[:,:] ans
    ans = np.empty((n1,n2), dtype=DOUBLE)
    for i1 in prange(n1,nogil=True): 
        for i2 in prange(n2):
            ans[i1, i2] = gsl_matrix_get(m, i1, i2)
    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef cDOUBLE[:,:,:] carrayofgslmatrix2pyarray(int n1, gsl_matrix** m):
    cdef int i,i1,i2,n2,n3
    n2=m[0].size1
    n3=m[0].size1
    cdef cDOUBLE[:,:,:] ans 
    ans = np.empty((n1,n2,n3),dtype=DOUBLE)
    for i in prange(n1,nogil=True):
        for i1 in prange(n2):
            for i2 in prange(n3):
                ans[i,i1, i2] = gsl_matrix_get(m[i], i1, i2)
    return ans

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef gsl_vector* pyarray2gslvector(cDOUBLE[:] a):
    cdef int i,n
    cdef gsl_vector* v
    n = a.shape[0]
    v = gsl_vector_alloc(n)
    for i in prange(n,nogil=True):
        gsl_vector_set(v,i,a[i])
    return v
    
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cdef cDOUBLE[:] gslvector2pyarray(gsl_vector* v):
    """ copies contents of v into numeric array """
    cdef int i,n
    n=v.size
    cdef cDOUBLE[:] ans 
    ans = np.empty((n), np.double)
    for i in prange(n,nogil=True):
        ans[i] = gsl_vector_get(v, i)
    return ans

def debug_needlets():
    """
    debug_needlets()

    Print needlets precision parameters.

    Parameters
    ----------

    Returns
    -------
    """
    mylib_print_needlets_parameters()

def mylibpy_jmax_lmax2B(pyjmax,pylmax):
    """
    mylibpy_jmax_lmax2B(jmax,lmax)

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
    return mylib_jmax_lmax2B(pyjmax,pylmax)

def mylibpy_jmax_xmax2B(pyjmax,pyxmax):
    """
    mylibpy_jmax_lmax2B(jmax,xmax)
    
    Calculate the parameter B given xmax and jmax

    This function calculate the value of the needlets width parameter B that
    center the *last* needlets windows function b(x/B**j) on xmax. In contrast to
    the case of mylibpy_jmax_lmax2B now xmax could be any double precision number
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
    return mylib_jmax_xmax2B(pyjmax,pyxmax)

def mylibpy_needlets_std_init_b_values(pyB,pyjmax,pylmax):
    """
    mylibpy_needlets_check_windows(jmax,lmax,b_values)
    
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
    cdef gsl_matrix* m
    cdef int jmax=pyjmax
    cdef int lmax=pylmax
    cdef double B=pyB
    cdef Py_ssize_t n1 = jmax+1
    cdef Py_ssize_t n2 = lmax+1

    with nogil:
        m = gsl_matrix_calloc(n1,n2)
        mylib_needlets_std_init_b_values_harmonic(m,B,jmax,lmax)
    
    py_b_values=gslmatrix2pyarray(m)

    with nogil:    
        gsl_matrix_free(m)

    return np.asarray(py_b_values)

def mylibpy_needlets_std_init_b_values_mergej(pyB,pyjmax,pylmax,mergej):
    """
    mylibpy_needlets_check_windows(jmax,lmax,b_values)
    
    This function check if the 1D needlets window functions are correct. [It checks if \sum_j b(l/B**j)**2=1 for l>0]

    Parameters
    ----------
    B: double
        Needlets width parameter
    jmax: int
        Max frequency (j=0,...,jmax)
    lmax: int
        Max harmonic parameter
    mergej:
        list of j bins to merge
    Returns
    ----------
    b_values: [0:jmax,0:lmax]
        Needlets window functions
    """

    b_values=mylibpy_needlets_std_init_b_values(pyB,pyjmax,pylmax)
    b_values_new = np.copy(b_values)

    jvec = np.arange(pyjmax+1)
    jmin_merge = mergej[0]
    jmax_merge = mergej[-1]
    jvec_new = np.arange(len(jvec) -len(mergej)+1)
    temp = np.zeros(pylmax+1)
    for l in range(b_values.shape[1]):
        for j in mergej:
            temp[l] += b_values[j][l]**2
        temp[l] = np.sqrt(temp[l])
    b_values_new[jmin_merge] =  temp
   
    idx_left=np.where(jvec<=jmin_merge)
    idx_right=np.where(jvec>jmax_merge)
    b_values_left = b_values_new[idx_left] 
    b_values_right = b_values_new[idx_right] 
    b_values_new_2 = np.concatenate([b_values_left, b_values_right], axis=0)
    print(b_values[jmin_merge], b_values[jmax_merge], b_values_new_2[1])
    return b_values_new_2

def j_B2nside(j,B):
    """
    Given the needlet frequency j and the B value return the minimum nside
    that must be used to analyze the frequency.
    """
    from numpy import log2
    
    # Frequency xmax
    xmax = B**(j + 1)
    
    # We use the approximate rule that xmax >= 2nside 
    #nside = int( round( 2**( log2(xmax) - 1 ) ) )
    #nside = 2**(xmax-1).bit_length()
    nside = 2**int( log2(xmax) )
    return nside


# Return the lmax of a specific frequency j
def j_B2l_upperbound(B, j):
  from numpy import floor
  needlet_machine_limit_upper = 1. - 8.e-15
  return int( np.floor( B**(j + 1)*needlet_machine_limit_upper ) )


#    def mylibpy_needlets_check_windows(self,jmax,lmax,cDOUBLE[:,:] py_b_values):
#    def mylibpy_needlets_check_windows(self,jmax,lmax,np.ndarray[cDOUBLE, ndim=2] py_b_values):
def mylibpy_needlets_check_windows(pyjmax,pylmax,object[cDOUBLE, ndim=2] py_b_values):
    """
    mylibpy_needlets_check_windows(jmax,lmax,b_values)
    
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
    cdef int jmax=pyjmax, lmax=pylmax
    with nogil:
        mylib_needlets_check_windows(jmax,lmax,c_b_values)
        gsl_matrix_free(c_b_values)



def mylibpy_needlets_f2betajk_j_healpix_harmonic(object[cDOUBLE, ndim=1] Map,object[cDOUBLE, ndim=2] py_b_values,pyj):
    """
    mylibpy_needlets_f2betajk_j_healpix_harmonic(Map,b_values,j)
    
    Standard needlets transform for harmonic functions. Healpy is required.

    This function return the coefficients of the standard needlets transform of a Healpix map for a choosen j.

    Parameters
    ----------
    Map: [0:Npix-1]
        Healpix Map
    b_values: [0:jmax,0:lmax]
        Needlets window functions
    jmax: int
        Choosen j (j=0,...,jmax)

    Returns
    ----------
    betajk_j: [0:Npix-1]
        Needlets coefficients of the input signal for a choosen j
    """
    from healpy import almxfl,alm2map,map2alm,get_nside
    nside=get_nside(Map)
    #jmax=len(py_b_values[:,0])+1
    #lmax=py_b_values.shape[0]+1
    lmax=len(py_b_values[0,:])-1
    #Npix=12*nside**2
    #print("f2betajk_j: lmax={:d},nside={:d},j={:d}".format(lmax,nside,pyj))
    return alm2map(almxfl(map2alm(Map,lmax=lmax),py_b_values[pyj,:]),lmax=lmax,nside=nside, verbose=False) #BIANCA verbose = false



def mylibpy_needlets_f2betajk_healpix_harmonic(object[cDOUBLE, ndim=1] Map,pyB,pyjmax,pylmax):
    """
    mylibpy_needlets_f2betajk_healpix_harmonic(Map,B,jmax,lmax)
    
    Standard needlets transform for harmonic functions. Healpy is required.

    This function return the coefficients of the standard needlets transform of a Healpix map.

    Parameters
    ----------
    Map: [0:Npix-1]
        Healpix Map
    B: double
        Needlets width parameter
    jmax: int
        Max frequency (j=0,...,jmax)
    lmax: int
        maximum harmonic

    Returns
    ----------
    betajk: [0:jmax,0:Npix-1]
        Needlets coefficients of the input signal
    """
    from healpy import almxfl,alm2map,map2alm,get_nside
    nside=get_nside(Map)
    Npix=12*nside**2
    b_values=mylibpy_needlets_std_init_b_values(pyB,pyjmax,pylmax)
    alm=map2alm(Map,lmax=pylmax)
    #print("lmax={:d},jmax={:d},nside={:d},B={:e}".format(pylmax,pyjmax,nside,pyB)) 
    py_betajk=np.empty((pyjmax+1,Npix))
    for j in np.arange(pyjmax+1):
        py_betajk[j,:]=alm2map(almxfl(alm,b_values[j,:]),lmax=pylmax,nside=nside, verbose = False) #BIANCA verbose = False
    return py_betajk


def mylibpy_needlets_f2betajk_healpix_harmonic_mergej(object[cDOUBLE, ndim=1] Map,pyB,pyjmax,pylmax, mergej):
    """
    mylibpy_needlets_f2betajk_healpix_harmonic_mergej(Map,B,jmax,lmax, mergej)
    
    Standard needlets transform for harmonic functions. Healpy is required.

    This function return the coefficients of the standard needlets transform of a Healpix map.

    Parameters
    ----------
    Map: [0:Npix-1]
        Healpix Map
    B: double
        Needlets width parameter
    jmax: int
        Max frequency (j=0,...,jmax)
    lmax: int
        maximum harmonic
    mergej:
        j to merge

    Returns
    ----------
    betajk: [0:jmax,0:Npix-1]
        Needlets coefficients of the input signal
    """
    from healpy import almxfl,alm2map,map2alm,get_nside
    nside=get_nside(Map)
    Npix=12*nside**2
    b_values=mylibpy_needlets_std_init_b_values_mergej(pyB,pyjmax,pylmax, mergej)
    alm=map2alm(Map,lmax=pylmax)
    #print("lmax={:d},jmax={:d},nside={:d},B={:e}".format(pylmax,pyjmax,nside,pyB)) 
    py_betajk=np.empty((b_values.shape[0],Npix))
    for j in np.arange(b_values.shape[0]):
        py_betajk[j,:]=alm2map(almxfl(alm,b_values[j,:]),lmax=pylmax,nside=nside, verbose = False) #BIANCA verbose = False
    return py_betajk



def mylibpy_needlets_f2betajk_accres_healpix_harmonic(object[cDOUBLE, ndim=1] Map,pyB,pyjmax,pylmax):
    """
    mylibpy_needlets_f2betajk_accres_healpix_harmonic(Map,B,jmax,lmax)
    
    Standard needlets transform for harmonic functions. Healpy is required.

    This function return the coefficients of the standard needlets transform of a Healpix map.
    
    In particular this function use the correct resolution for all the selected resolutions pyjmax

    Parameters
    ----------
    Map: [0:Npix-1]
        Healpix Map
    B: double
        Needlets width parameter
    jmax: int
        Max frequency (j=0,...,jmax)
    lmax: int
        maximum harmonic

    Returns
    ----------
    betajk: [0:jmax,0:Npix-1]
        Needlets coefficients of the input signal
    """
    from healpy import almxfl,alm2map,map2alm,get_nside
    nside = get_nside(Map)
    Npix = 12*nside**2
    b_values = mylibpy_needlets_std_init_b_values(pyB,pyjmax,pylmax)
    #alm = map2alm(Map,lmax=pylmax)
    #print("lmax={:d},jmax={:d},nside={:d},B={:e}".format(pylmax, pyjmax, nside, pyB))
    #py_betajk=np.empty((pyjmax+1,Npix))
    py_betajk = [ np.zeros(12) ] # j = 0 - Monopole must be subtracted!
    for j in np.arange(1,pyjmax + 1):
        jnside = min( j_B2nside(j,pyB), nside )
        jlmax  = min( j_B2l_upperbound(pyB, j), pylmax )
        print( "jnside = {:d}, jlmax = {:d}".format(jnside, jlmax) )
        py_betajk.append( np.empty(12*jnside**2) )
        alm = map2alm(Map,lmax=jlmax, verbose=False) #BIANCA verbose=False
        py_betajk[j][:] = alm2map( almxfl(alm, b_values[j, :jlmax + 1 ]), lmax = jlmax, nside = jnside, verbose=False ) #BIANCA verbose=False
    return py_betajk




def mylibpy_needlets_betajk2f_healpix_harmonic(object[cDOUBLE, ndim=2] betajk,pyB,pylmax):
    """
    mylibpy_needlets_f2betajk_healpix_harmonic(betajk,B,lmax)
    
    Standard needlets transform for harmonic functions. Healpy is required.

    This function return the reconstructed Map from the its standard needlets coefficients.

    Parameters
    ----------
    betajk: [0:jmax,0:Npix-1]
        Needlets coefficients of the input signal
    B: double
        Needlets width parameter
    lmax: double
        maximum multipole

    Returns
    ----------
    Map: [0:Npix-1]
        Reconstructed Map
    """
    from healpy import get_nside
    jmax=len(betajk[:,0])-1
    nside=get_nside(betajk[0,:])
    Npix=12*nside**2
    b_values=mylibpy_needlets_std_init_b_values(pyB,jmax,pylmax)
    #print("lmax={:d},jmax={:d},nside={:d},B={:e}".format(pylmax,jmax,nside,pyB))
    Map=np.zeros(Npix)
    for j in np.arange(jmax+1):
        print("j=",j)
        Map+=mylibpy_needlets_f2betajk_j_healpix_harmonic(betajk[j,:],b_values,j)
    return Map


#def needlets_betajk2f_accres_healpix_harmonic(object[cDOUBLE, ndim=2] betajk,pyB,pylmax):
def mylibpy_needlets_betajk2f_accres_healpix_harmonic( betajk, pyB, pylmax):
    """
    mylibpy_needlets_f2betajk_healpix_harmonic(betajk,B,lmax)
    
    Standard needlets transform for harmonic functions. Healpy is required.

    This function return the reconstructed Map from the its standard needlets coefficients.

    Parameters
    ----------
    betajk: [0:jmax,0:Npix-1]
        Needlets coefficients of the input signal
    B: double
        Needlets width parameter
    lmax: double
        maximum multipole

    Returns
    ----------
    Map: [0:Npix-1]
        Reconstructed Map
    """
    from healpy import get_nside, ud_grade
    jmax = len(betajk[:])-1
    nside = get_nside( betajk[-1][:] )
    Npix = 12*nside**2
    b_values = mylibpy_needlets_std_init_b_values(pyB, jmax, pylmax)
    #print("lmax={:d},jmax={:d},nside={:d},B={:e}".format(pylmax,jmax,nside,pyB))
    Map=np.zeros(Npix)
    for j in np.arange(1,jmax + 1):
        print("j=",j)
        #jnside = min( j_B2nside(j,pyB), nside )
        jlmax  = min( j_B2l_upperbound(pyB, j), pylmax )
        Map += ud_grade( mylibpy_needlets_f2betajk_j_healpix_harmonic(betajk[j][:],b_values[:, :jlmax + 1 ], j ), nside )
    return Map


#def mylibpy_needlets_std_init_b_values_even_square_uncentred_1d(pyB,pyjmax,pyN,pyL):
#    """
#    mylibpy_needlets_std_init_b_values_even_square_uncentred_1d(B,jmax,N,L)
#    
#    Standard needlets windows functions 1D: array of gsl_matrix [FFT version] 
#
#    This function return the 1D standard needlets functions, in form of ndarray matrix
#    with dimension (jmax+1,N) (N is even).
#
#    Parameters
#    ----------
#    B: double
#        Needlets width parameter
#    jmax: int
#        Max frequency (j=0,...,jmax)
#    N: int
#        Number of points (sampling)
#    L: double
#        physical dimension of the 1D signal - Not needed? 
#
#    Returns
#    ----------
#    b_values: [0:jmax,0:N-1]
#        Needlets window functions
#    """
#    cdef gsl_matrix* m
#    cdef int jmax=pyjmax, N=pyN
#    cdef double B=pyB, L=pyL
#    cdef int n1 = jmax+1
#    cdef int n2 = N
#    
#    with nogil:
#        m = gsl_matrix_calloc(n1,n2)
#        mylibpy_needlets_std_init_b_values_even_square_uncentred_1d(m,B,jmax,N,L)
#
#    py_b_values=gslmatrix2pyarray(m)
#
#    with nogil:
#        gsl_matrix_free(m)
#
#    return np.asarray(py_b_values)
#
#
#
##    def mylibpy_needlets_f2betajk_omp_even_uncentred_1d(self,cDOUBLE[:] func,B,jmax,L):
##    def mylibpy_needlets_f2betajk_omp_even_uncentred_1d(self,np.ndarray[cDOUBLE, ndim=1] func,B,jmax,L):
#def mylibpy_needlets_f2betajk_omp_even_uncentred_1d(object[cDOUBLE, ndim=1] func,pyB,pyjmax,pyL):
#    """
#    mylibpy_needlets_f2betajk_omp_even_uncentred_1d(func,B,jmax,L)
#    
#    Standard needlets transform for functions 1D [FFT version]
#
#    This function return the coefficients of the standard needlets transform of a function func.
#
#    Parameters
#    ----------
#    func: [0:N-1]
#        Sampled signal array
#    B: double
#        Needlets width parameter
#    jmax: int
#        Max frequency (j=0,...,jmax)
#    L: double
#        physical dimension of the 1D signal - Not needed? 
#
#    Returns
#    ----------
#    betajk: [0:jmax,0:N-1]
#        Needlets coefficients of the input signal
#    """
#
#    cdef int N=len(func)
#    v = pyarray2gslvector(func)
#
#    cdef gsl_matrix* m
#    cdef int jmax=pyjmax
#    cdef double B=pyB, L=pyL
#    cdef int n1 = jmax+1
#    cdef int n2 = N
#
#    with nogil:
#        m = gsl_matrix_alloc(n1,n2)
#        mylibpy_needlets_f2betajk_omp_even_uncentred_1d(N,v,m,B,jmax,L)
#
#    py_betajk=gslmatrix2pyarray(m)
#
#    with nogil:
#        gsl_matrix_free(m)
#        gsl_vector_free(v)
#
#    return py_betajk
#
#
#
#
##    def mylibpy_needlets_betajk2f_omp_even_uncentred_1d(self,cDOUBLE[:,:] betajk,B,L):
##    def mylibpy_needlets_betajk2f_omp_even_uncentred_1d(self,np.ndarray[cDOUBLE, ndim=2] betajk,B,L):
#def mylibpy_needlets_betajk2f_omp_even_uncentred_1d(object[cDOUBLE, ndim=2] betajk,pyB,pyL):
#    """
#    mylibpy_needlets_f2betajk_omp_even_uncentred_1d(func,B,L)
#    
#    Standard needlets transform for functions 1D [FFT version]
#
#    This function return the coefficients of the standard needlets transform of a function func.
#
#    Parameters
#    ----------
#    betajk: [0:jmax,0:N-1]
#        Needlets coefficients of the input signal
#    B: double
#        Needlets width parameter
#    L: double
#        physical dimension of the 1D signal - Not needed? 
#
#    Returns
#    ----------
#    func: [0:N-1]
#        Sampled signal array
#    """
#
#    cdef int N=len(betajk[0,:])
#    cdef int jmax=len(betajk[:,0])-1
#    cdef double B=pyB, L=pyL
#    m = pyarray2gslmatrix(betajk)
#
#    cdef gsl_vector* v
#    cdef int n = N
#
#    with nogil:
#        v = gsl_vector_alloc(n)
#        mylibpy_needlets_betajk2f_omp_even_uncentred_1d(N,m,v,B,jmax,L)
#
#    py_func=gslvector2pyarray(v)
#
#    with nogil:
#        gsl_matrix_free(m)
#        gsl_vector_free(v)
#
#    return py_func
#
#
#
#
##    def mylibpy_fftshift_even_square_2d_dp(self,N,cDOUBLE[:,:] pysquare):
##Faster but numpy array only
##    def mylibpy_fftshift_even_square_2d_dp(self,N,np.ndarray[cDOUBLE, ndim=2] pysquare):
##Slower but generic, python3 only
#def fftshift_even_square_2d_dp(pyN,object[cDOUBLE, ndim=2] pysquare):
#    """
#    mylibpy_fftshift_even_2d_dp(N,square)
#    
#    This function shift (FFT compatible) the (real valued) square patch
#
#    Parameters
#    ----------
#    N: int (even)
#        Square side number dimension
#    square: float (NxN)
#        A real valued square patch
#
#    Returns
#    -------
#    square: float (NxN)
#        The square patch is shifted (FFT compatible)
#    """
#    #To be tested!!!:
#    #mylibpy_fftshift_even_2d_dp(N,<gsl_matrix *> pysquare.data);
#    cdef int N=pyN
#    csquare=pyarray2gslmatrix(pysquare)
#    with nogil:
#        mylibpy_fftshift_even_square_2d_dp(N, csquare)
#    pysquare=gslmatrix2pyarray(csquare)
#    with nogil:
#        gsl_matrix_free(csquare)
#    return pysquare
#
#
#
#def mylibpy_needlets_std_init_b_values_even_square_uncentred_2d(pyB,pyjmax,pyN,pyL):
#    """
#    mylibpy_needlets_std_init_b_values_even_square_uncentred_1d(B,jmax,N,L)
#    
#    Standard needlets windows functions 2D: array of gsl_matrix
#
#    This function return the 2D standard needlets functions, in form of array of gsl_matrix,
#    any matrix is a (N,N) patch for a frequency j (N is even).
#
#    Parameters
#    ----------
#    B: double
#        Needlets width parameter
#    jmax: int
#        Max frequency (j=0,...,jmax)
#    N: int
#        Number of grid side points
#    L: double
#        physical dimension of the grid - Not needed? 
#
#    Returns
#    ----------
#    b_values: [0:jmax,0:N-1,0:N-1]
#        Needlets window functions
#    """
#    cdef gsl_matrix** m
#    cdef int jmax=pyjmax,N=pyN
#    cdef double B=pyB, L=pyL
#    cdef int n1 = jmax+1
#    cdef int n2 = N,i
#
#    m=<gsl_matrix **>malloc(n1*sizeof(gsl_matrix *))
#
#    with nogil:
#        #for i from 0<=i<=jmax:
#        for i in prange(jmax+1):
#            m[i] = gsl_matrix_calloc(n2,n2)
#
#        mylibpy_needlets_std_init_b_values_even_square_uncentred_2d(m,B,jmax,N,L)
#
#    py_b_values=carrayofgslmatrix2pyarray(jmax+1,m)
#
#    #for i from 0<=i<=jmax:
#    with nogil:
#        for i in prange(jmax+1):
#            gsl_matrix_free(m[i])
#    free(m)
#    return py_b_values
#
#
#
##    def mylibpy_needlets_f2betajk_omp_even_uncentred_2d(self,cDOUBLE[:,:] func,B,jmax,L):
##    def mylibpy_needlets_f2betajk_omp_even_uncentred_2d(self,np.ndarray[cDOUBLE, ndim=2] func,B,jmax,L):
#def mylibpy_needlets_f2betajk_omp_even_uncentred_2d(object[cDOUBLE, ndim=2] func,pyB,pyjmax,pyL):
#    """
#    mylibpy_needlets_f2betajk_omp_even_uncentred_2d(func,B,jmax,L)
#    
#    Standard needlets transform for functions 2D [FFT version]
#
#    This function return the coefficients of the standard needlets transform of a function func.
#
#    Parameters
#    ----------
#    func: [0:N-1,0:N-1]
#        Sampled signal array
#    B: double
#        Needlets width parameter
#    jmax: int
#        Max frequency (j=0,...,jmax)
#    L: double
#        physical dimension of the grid - Not needed? 
#
#    Returns
#    ----------
#    betajk: [0:jmax,0:N-1,0:N-1]
#        Needlets coefficients of the input signal
#    """
#
#    cdef int N=len(func[0,:])
#    square = pyarray2gslmatrix(func)
#
#    cdef gsl_matrix** betajk
#    cdef int jmax=pyjmax
#    cdef double B=pyB, L=pyL
#    cdef int n1 = jmax+1
#    cdef int n2 = N, i
#    betajk=<gsl_matrix **>malloc(n1*sizeof(gsl_matrix *))
#
#    with nogil:
#        for i in prange(jmax+1):
#        #for i from 0<=i<=jmax:
#            betajk[i] = gsl_matrix_calloc(n2,n2)
#
#        mylibpy_needlets_f2betajk_omp_even_uncentred_2d(N,square,betajk,B,jmax,L)
#
#    py_betajk=carrayofgslmatrix2pyarray(jmax+1,betajk)
#
#    with nogil:
#        for i in prange(jmax+1):
#        #for i from 0<=i<=jmax:
#            gsl_matrix_free(betajk[i])
#        free(betajk)
#    gsl_matrix_free(square)
#    return py_betajk
#
#
#
##    def mylibpy_needlets_betajk2f_omp_even_uncentred_2d(self,cDOUBLE[:,:,:] betajk,B,L):
##    def mylibpy_needlets_betajk2f_omp_even_uncentred_2d(self,np.ndarray[cDOUBLE, ndim=3] betajk,B,L):
#def mylibpy_needlets_betajk2f_omp_even_uncentred_2d(object[cDOUBLE, ndim=3] betajk,pyB,pyL):
#    """
#    mylibpy_needlets_f2betajk_omp_even_uncentred_2d(func,B,L)
#    
#    Standard needlets transform for functions 2D [FFT version]
#
#    This function return the coefficients of the standard needlets transform of a function func.
#
#    Parameters
#    ----------
#    betajk: [0:jmax,0:N-1,0:N-1]
#        Needlets coefficients of the input signal
#    B: double
#        Needlets width parameter
#    L: double
#        physical dimension of the grid - Not needed? 
#
#    Returns
#    ----------
#    func: [0:N-1,0:N-1]
#        Sampled signal array
#    """
#
#    cdef int N=len(betajk[0,0,:])
#    cdef int jmax=len(betajk[:,0,0])-1
#    m = pyarray2carrayofgslmatrix(betajk)
#    #print("N={:d}, jmax={:d}".format(N,jmax))
#
#    cdef gsl_matrix* square
#    cdef double B=pyB, L=pyL
#    cdef int n = N,i
#
#    with nogil:
#        square = gsl_matrix_alloc(N,N)
#        mylibpy_needlets_betajk2f_omp_even_uncentred_2d(N,m,square,B,jmax,L)
#
#    py_square=gslmatrix2pyarray(square)
#
#    with nogil:
#        for i in prange(jmax+1):
#        #for i from 0<=i<=jmax:
#            gsl_matrix_free(m[i])
#        free(m)
#
#    gsl_matrix_free(square)
#
#    return py_square


