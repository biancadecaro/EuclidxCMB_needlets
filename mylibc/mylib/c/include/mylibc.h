/**
 * @file mylibc.h
 * @author Alessandro Renzi
 * @date 17 12 2015
 * @brief Library for scientific numerical computing.
 *
 */

#ifndef __MYLIBC__
#define __MYLIBC__

/* standard libraries */
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "string.h"
#include "float.h"
#ifdef _OPENMP
#include "omp.h"
#endif

#include "constants.h"
#include "fft_unit.h"
#include "needlets_unit.h"

#endif //__MYLIBC__
