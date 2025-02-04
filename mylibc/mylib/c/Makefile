#Modified Makefile from CLASS.

MDIR := $(shell pwd)
WRKDIR = $(MDIR)/build

.base:
	if ! [ -e $(WRKDIR) ]; then mkdir $(WRKDIR) ; mkdir $(WRKDIR)/lib; fi;
	touch build/.base

vpath %.c source:test
vpath %.o build
vpath .base build

########################################################
###### LINES TO ADAPT TO YOUR PLATEFORM ################
########################################################

# your C compiler:
CC       = gcc
#CC       = icc
#CC       = pgcc

# your tool for creating static libraries:
AR        = ar rv

# Your python interpreter
PYTHON = python

# your optimization flag
DEBUGFLAG=-v#-fbounds-check
OPTFLAG = -O3 -ffast-math -march=native
#OPTFLAG = -Ofast -ffast-math #-march=native
#OPTFLAG = -fast

# your openmp flag (comment for compiling without openmp)
OMPFLAG   = -fopenmp
#OMPFLAG   = -mp -mp=nonuma -mp=allcores -g
#OMPFLAG   = -openmp

# all other compilation flags
CCFLAG = -g -fPIC -std=c99 -DHAVE_INLINE $(DEBUGFLAG)
LDFLAG = -g -fPIC -std=c99 -L/usr/local/lib -lfftw3_threads -lfftw3_omp -lfftw3 -L/usr/local/lib -lgsl -lgslcblas#-L$(HOME)/fftw3/fftw-3.3.10-double/lib -lfftw3_threads -lfftw3_omp -lfftw3 -L$(HOME)/gsl/gsl-2.7.1/lib -lgsl -lgslcblas

########################################################
###### IN PRINCIPLE THE REST SHOULD BE LEFT UNCHANGED ##
########################################################

# pass current working directory to the code
#CCFLAG += -D__CLASSDIR__='"$(MDIR)"'

# where to find include files *.h
INCLUDES = -I../include -I/usr/local/include#-I../include -I$(HOME)/fftw3/fftw-3.3.10-double/include -I$(HOME)/gsl/gsl-2.7.1/include

# automatically add external programs if needed. First, initialize to blank.
EXTERNAL =

PYTHONFLAGS=--user

%.o:  %.c .base
	cd $(WRKDIR);$(CC) $(OPTFLAG) $(OMPFLAG) $(CCFLAG) $(INCLUDES) -c ../$< -o $*.o
SOURCES = fft_unit.o needlets_unit.o

MAIN = test_needlets_unit.o

all: testlib libmylibc.a cythonwrap

libmylibc.a: $(SOURCES) $(EXTERNAL)
	$(AR)  $@ $(addprefix build/, $(SOURCES) $(EXTERNAL))

testlib: $(SOURCES) $(EXTERNAL) $(MAIN)
	$(CC) $(OPTFLAG) $(OMPFLAG) -o testlib $(addprefix build/,$(notdir $^)) $(LDFLAG) -lm

#test_sigma: $(TOOLS) $(SOURCES) $(EXTERNAL) $(OUTPUT) $(TEST_SIGMA)
#	$(CC) $(OPTFLAG) $(OMPFLAG) $(LDFLAG) -o test_sigma $(addprefix build/,$(notdir $^)) -lm

cythonwrap: libmylibc.a python/cython_mylibc.pyx python/c_cython_mylibc.pxd
	cd python; $(PYTHON) setup.py install $(PYTHONFLAGS)
#export CC=$(CC);
clean: .base
	rm -rf $(WRKDIR);
	rm -f libmylibc.a
	rm -f testlib
	#rm -f $(MDIR)/python/needlets_module.c
	rm -rf $(MDIR)/python/build
	rm -rf python/cython_mylibc.c
