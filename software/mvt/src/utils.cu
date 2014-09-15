#include <iostream>

#include <cuda.h>
#if CUDA_VERSION>=5000
#include <helper_cuda.h>
#define cutilSafeCall checkCudaErrors
#else
#include <cutil_inline.h>
#endif

#include "matrix.h" 
#include "utils.h"

__global__ void dot_cuda ( CUVEC vec );

void dot( CUVEC vec, int blockPerGrid )
{   	
	dot_cuda <<< blockPerGrid , ThreadPerBlock >>> (vec);
	cudaThreadSynchronize();
}

__global__ void dot_cuda ( CUVEC vec )
{
	__shared__ float chache[ThreadPerBlock] ;
	unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x ;
	unsigned int chacheIndex = threadIdx.x ;

	float* V1 = vec.v1;
	float* V2 = vec.v2;
	float* V3 = vec.v_ret;
	int length = vec.length;
	
	float temp=0;
	while ( tid < length )
	{
		temp += V1[tid] * V2[tid] ;	
		tid += blockDim.x * gridDim.x ;
	}

	chache[chacheIndex] = temp ;
	__syncthreads();

	int i  = blockDim.x / 2 ;

	while ( i!=0 )
	{
		if ( chacheIndex < i )
		{
			chache[chacheIndex] += chache [chacheIndex + i] ;
		}

		__syncthreads();
		i/=2 ;
	}

	if ( chacheIndex == 0 )
	{
		V3[blockIdx.x] = chache [0] ;
	}
}