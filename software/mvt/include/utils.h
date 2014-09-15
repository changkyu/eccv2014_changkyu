#ifndef __MVT_UTILS_H__
#define __MVT_UTILS_H__

#define imin(a,b) (a<b?a:b)
#define ThreadPerBlock 256

typedef struct CUVEC
{
	float* v1;
	float* v2;
	float* v_ret;

	int length;
}CUVEC;

void dot( CUVEC vec , int blockPerGrid);

#endif
