#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include <iostream>

//cuRand
#include <curand_kernel.h>
#include <curand.h>

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);

__global__ 
void init_rand_state(curandState* d_rand_state, int width, int height);

#endif