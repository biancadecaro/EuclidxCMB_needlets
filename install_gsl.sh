#! /bin/bash

wget ftp://ftp.gnu.org/gnu/gsl/gsl-2.8.tar.gz
tar -zxvf gsl-2.8.tar.gz
cd gsl-2.8
./configure
make -j 8
make check
make install
