#include <stdio.h>

#include <cuda.h>
#if CUDA_VERSION>=5000
#include <helper_cuda.h>
#define cutilSafeCall checkCudaErrors
#else
#include <cutil_inline.h>
#endif

#include "haarftr.h"

__global__ void HaarFtr_compute(CUHaarFtr haarftr_device, CUSampleSet samples);

void HaarFtr_classifySetF(CUHaarFtr haarftr, CUSampleSet samples, float* ret)
{	
	int n_samples = samples.n_samples;
			
	HaarFtr_compute<<< (n_samples+255)/256, 256 >>>(haarftr, samples);
	cudaThreadSynchronize();
	
	cutilSafeCall(cudaMemcpy(ret, samples.ret, sizeof(float)*n_samples, cudaMemcpyDeviceToHost));
}

__global__ void HaarFtr_compute(CUHaarFtr haarftr_device, CUSampleSet samples)
{	
	int n_samples = samples.n_samples;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	
	if( i < n_samples )
	{	
		CURect rect_sample = samples.rects[i];
		int n_rows = samples._ii_imgs_n_rows[haarftr_device._channel];
		int n_cols = samples._ii_imgs_n_cols[haarftr_device._channel];
		
		CURect r;
		float sum = 0.0f;
		
		for (int k = 0; k < haarftr_device._rects_size; k++)
		{
			r = haarftr_device._rects[k];
			r.x += rect_sample.x;
			r.y += rect_sample.y;
			
			if( !(
					(r.x < 0) || (r.y < 0) || ((r.x + r.width) >= n_cols) || ((r.y + r.height) >= n_rows) 
				 )
			)
			{
			
				sum +=
						haarftr_device._weights[k] * (
										samples._ii_imgs[haarftr_device._channel *n_rows*n_cols + (r.y + r.height)*n_cols + (r.x + r.width)]
									  + samples._ii_imgs[haarftr_device._channel *n_rows*n_cols + (r.y           )*n_cols + (r.x          )]
									  - samples._ii_imgs[haarftr_device._channel *n_rows*n_cols + (r.y + r.height)*n_cols + (r.x          )]
									  - samples._ii_imgs[haarftr_device._channel *n_rows*n_cols + (r.y           )*n_cols + (r.x + r.width)]  );
				
			}
		}
		
		samples.ret[i] = sum;
	}
}

