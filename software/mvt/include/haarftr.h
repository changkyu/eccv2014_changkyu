#ifndef __HAARFTR_H__
#define __HAARFTR_H__

typedef struct CURect
{
	int x;
	int y;
	int width;
	int height;
}CURect;

typedef struct CUHaarFtr
{
	CURect*        _rects;
	int           _rects_size;
	float*        _weights;
	int           _channel;
}CUHaarFtr;

typedef struct CUSampleSet
{
	float*        _ii_imgs;
	int _ii_imgs_n_channels;
	int* _ii_imgs_n_rows;
	int* _ii_imgs_n_cols;

	int n_samples;
	CURect*       rects;
	float*       ret;
}CUSampleSet;

void HaarFtr_classifySetF(CUHaarFtr haarftr, CUSampleSet samples, float* ret);

#endif
