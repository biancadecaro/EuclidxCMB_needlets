#! /bin/bash

wget http://fftw.org/fftw-3.3.10.tar.gz
tar -zxf fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure CC=gcc F77=gfortran --enable-shared --enable-threads --enable-openmp --enable-mpi MPICC=mpicc
make -j 8
make install
