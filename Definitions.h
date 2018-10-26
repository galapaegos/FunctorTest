#pragma once

#include <thrust/complex.h>

typedef double fptype;
typedef thrust::complex<fptype> fpcomplex;

__device__ fptype d_parameters[100];
__device__ fptype d_constants[100];
__device__ fptype d_observables[100];
__device__ fptype d_normalizations[100];
