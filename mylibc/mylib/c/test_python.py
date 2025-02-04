from numpy import *
import matplotlib.pylab as plt

import cython_mylibc
pippo=cython_mylibc.mylibc()

#NOTE: Use only power of 2:
N=64
m=zeros((N,N))
mu, sigma = 0., 1.
for i in arange(N):
  m[i,:]=random.normal(mu, sigma, N)

import copy
mapp=copy.deepcopy(m)

#Needlets trasform works only when the mean value is subtracted
m=m-m.mean()

#plt.imshow(mapp,interpolation='none')
#plt.title("Original")
#plt.colorbar()
#plt.show()

#Up to now works only for B=2.
B=2.
jmax=int(log(N)/log(B)) #B**jmax=N
print("N={:d},B={:e},jmax={:d}".format(N,B,jmax))

b_values=pippo.mylib_needlets_std_init_b_values_even_square_uncentred_2d(B,jmax,N,N)
#for i in arange(jmax+1):
#  plt.imshow(b_values[i,:,:],interpolation='none')
#  plt.title("j={:d}".format(i))
#  plt.colorbar()
#  plt.show()

should_be_one=zeros((N,N))
for x in arange(N):
  for y in arange(N):
    for j in arange(jmax+1):
      app=b_values[j,x,y]
      should_be_one[x,y]+=app**2
#plt.imshow(should_be_one,interpolation='none')
#plt.title("Needlets closure".format(i))
#plt.colorbar()
#plt.show()

betajk=pippo.mylib_needlets_f2betajk_omp_even_uncentred_2d(m,B,jmax,N)
#for i in arange(jmax+1):
  #plt.imshow(betajk[i,:,:],interpolation='none')
  #plt.title("j={:d}".format(i))
  #plt.colorbar()
  #plt.show()

newm=pippo.mylib_needlets_betajk2f_omp_even_uncentred_2d(betajk,B,N)
#plt.imshow(newm,interpolation='none')
#plt.title("Reconstructed")
#plt.colorbar()
#plt.show()

#plt.imshow(abs(mapp-newm),interpolation='none')
#plt.title("Subtraction")
#plt.colorbar()
#plt.show()

