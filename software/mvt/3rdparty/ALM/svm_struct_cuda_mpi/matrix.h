/*
  float matrix type for cuda
  author: Yu Xiang
  Date: 04/10/2011
*/

#ifndef CUMATRIX_H
#define CUMATRIX_H

#include <stdio.h>
typedef struct cumatrix
{
  int dims_num;
  int *dims;
  int length;
  /* column first storage */
  float *data;
}CUMATRIX;

CUMATRIX read_cumatrix(FILE *fp);
void write_cumatrix(CUMATRIX *pmat, FILE *fp);
void print_cumatrix(CUMATRIX *pmat);
void free_cumatrix(CUMATRIX *pmat);
CUMATRIX alloc_device_cumatrix(CUMATRIX mat_host);
void free_device_cumatrix(CUMATRIX *pmat);
CUMATRIX pad_3d_maxtrix(CUMATRIX A, int padx, int pady);
CUMATRIX copy_cumatrix(CUMATRIX A);

#endif
