/*
 * utils_cuda.cpp *
 *  Created on: Jul 30, 2013
 *      Author: changkyu
 */

#include <cuda.h>
#include <cuda_runtime_api.h>
#if CUDA_VERSION>=5000
#include <helper_cuda.h>
#define cutilSafeCall checkCudaErrors
#else
#include <cutil_inline.h>
#endif
//#include <cutil_gl_inline.h>
//#include <cuda_gl_interop.h>

extern "C"
{
#include "../3rdparty/ALM/svm_struct_cuda_mpi/matrix.h"
}

#include "cv_onlinemil.h"
#include "haarftr.h"

#include "utils.h"
#include "utils_cuda.h"

using namespace cv::mil;

#if 0
float compute_hog_similarity( CUMATRIX hog1, CUMATRIX hog2 )
{
	int length = hog1.length;
	int blockPerGrid = imin(32, (length+ThreadPerBlock-1) / ThreadPerBlock );
	CUVEC vec;
tic_count(TIMER_ALM_DOT_ALLOC);
	cutilSafeCall(cudaMalloc((void **)&(vec.v1), length*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&(vec.v2), length*sizeof(float)));
	cutilSafeCall(cudaMalloc((void **)&(vec.v_ret), blockPerGrid*sizeof(float)));

	cutilSafeCall(cudaMemcpy(vec.v1,hog1.data,length*sizeof(float),cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(vec.v2,hog2.data,length*sizeof(float),cudaMemcpyHostToDevice));
	vec.length = length;
toc_count(TIMER_ALM_DOT_ALLOC);
tic_count(TIMER_ALM_DOT_CAL);
	dot(vec, blockPerGrid);
toc_count(TIMER_ALM_DOT_CAL);
tic_count(TIMER_ALM_DOT_FREE);
	float ret[length];
	cutilSafeCall(cudaMemcpy(ret,vec.v_ret,blockPerGrid*sizeof(float),cudaMemcpyDeviceToHost));
	float sum=0;
	for( int i = 0 ; i<blockPerGrid ; i++ )
	{
		sum+=ret[i];
	}

	cutilSafeCall(cudaFree(vec.v1));
	cutilSafeCall(cudaFree(vec.v2));
	cutilSafeCall(cudaFree(vec.v_ret));
toc_count(TIMER_ALM_DOT_FREE);
	return sum;
}
#endif

CUHaarFtr Alloc_CUHaarFtr(unsigned int n_rects, unsigned int n_weights)
{
	CUHaarFtr cuhaarftr;

	cutilSafeCall(cudaMalloc((void**)&(cuhaarftr._rects),sizeof(CURect)*n_rects));
	cutilSafeCall(cudaMalloc((void**)&(cuhaarftr._weights),sizeof(float)*n_weights));

	return cuhaarftr;
}

void HaarFtr_to_CUHaarFtr(HaarFtr* haarftr, CUHaarFtr* cuhaarftr)
{
	cuhaarftr->_channel = (int)haarftr->_channel;

	int n_rects = (int)haarftr->_rects.size();
	cuhaarftr->_rects_size = n_rects;
	CURect rects[n_rects];
	for( int r=0; r<n_rects; r++ )
	{
		rects[r].x      = haarftr->_rects[r].x;
		rects[r].y      = haarftr->_rects[r].y;
		rects[r].width  = haarftr->_rects[r].width;
		rects[r].height = haarftr->_rects[r].height;
	}

	cutilSafeCall(cudaMemcpy(cuhaarftr->_rects,rects,sizeof(CURect)*n_rects, cudaMemcpyHostToDevice));

	int n_weights = haarftr->_weights.size();
	float weights[n_weights];
	for( int w=0; w<n_weights; w++ )
	{
		weights[w] = haarftr->_weights[w];
	}
	cutilSafeCall(cudaMemcpy(cuhaarftr->_weights,weights,sizeof(float)*n_weights, cudaMemcpyHostToDevice));
}

void Free_CUHaarFtr(CUHaarFtr ftr)
{
	cutilSafeCall(cudaFree(ftr._rects));
	cutilSafeCall(cudaFree(ftr._weights));
}

CUSampleSet SampleSet_to_CUSamples(SampleSet sampleset)
{
	// Assume that ii_imgs are same for all samples
	int n_ch   = sampleset[0]._ii_imgs.size();
	int n_rows = sampleset[0]._ii_imgs[0].rows;
	int n_cols = sampleset[0]._ii_imgs[0].cols;

	CUSampleSet css;

	css._ii_imgs_n_channels = n_ch;
	int cols[n_ch], rows[n_ch];
	for( int ch=0; ch<n_ch; ch++ )
	{
		cols[ch] = n_cols;
		rows[ch] = n_rows;
	}
	cutilSafeCall(cudaMalloc((void**)&(css._ii_imgs_n_cols),sizeof(int)*n_ch));
	cutilSafeCall(cudaMalloc((void**)&(css._ii_imgs_n_rows),sizeof(int)*n_ch));
	cutilSafeCall(cudaMemcpy(css._ii_imgs_n_cols, cols, sizeof(int)*n_ch, cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(css._ii_imgs_n_rows, rows, sizeof(int)*n_ch, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMalloc((void**)&(css._ii_imgs),sizeof(float)*n_rows*n_cols*n_ch));
	float data[n_ch*n_rows*n_cols];
	for( int ch=0; ch<n_ch; ch++ )
	{
		memcpy(&data[ch*n_rows*n_cols],sampleset[0]._ii_imgs[ch].data,sizeof(float)*n_rows*n_cols);
	}
	cutilSafeCall(cudaMemcpy(css._ii_imgs, data, sizeof(float)*n_ch*n_rows*n_cols, cudaMemcpyHostToDevice));

	int n_samples = sampleset.size();
	css.n_samples = n_samples;
	CURect rects[n_samples];
	for( int s=0; s<n_samples; s++)
	{
		rects[s].x      = sampleset[s]._col;
		rects[s].y      = sampleset[s]._row;
		rects[s].width  = sampleset[s]._width;
		rects[s].height = sampleset[s]._height;
	}
	cutilSafeCall(cudaMalloc((void**)&(css.rects),sizeof(CURect)*n_samples));
	cutilSafeCall(cudaMemcpy(css.rects, rects, sizeof(CURect)*n_samples, cudaMemcpyHostToDevice));

	cutilSafeCall(cudaMalloc((void**)&(css.ret),sizeof(float)*n_samples));

	return css;
}

void Free_CUSampleSet(CUSampleSet css)
{
	cutilSafeCall(cudaFree(css._ii_imgs));
	cutilSafeCall(cudaFree(css._ii_imgs_n_cols));
	cutilSafeCall(cudaFree(css._ii_imgs_n_rows));
	cutilSafeCall(cudaFree(css.rects));
	cutilSafeCall(cudaFree(css.ret));
}


