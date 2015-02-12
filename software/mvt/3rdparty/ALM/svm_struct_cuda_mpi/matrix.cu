/*
  float matrix implementation
  author: Yu Xiang
  Date: 04/10/2011
*/

extern "C"
{
#include "matrix.h"
}
#include <cuda.h>
#if CUDA_VERSION>=5000
#include <helper_cuda.h>
#define cutilSafeCall checkCudaErrors
#else
#include "cutil_inline.h"
#endif

CUMATRIX read_cumatrix(FILE *fp)
{
  int i;
  CUMATRIX matrix;

  /* initialization */
  matrix.dims_num = 0;
  matrix.dims = NULL;
  matrix.length = 0;
  matrix.data = NULL;

  /* read dimension */
  if(fscanf(fp, "%d", &matrix.dims_num) != 1)
  {
    printf("fscanf failed\n");
    return matrix;
  }
  /* allocate dims */
  matrix.dims = (int*)malloc(sizeof(int)*matrix.dims_num);
  matrix.length = 1;
  for(i = 0; i < matrix.dims_num; i++)
  {
    if(fscanf(fp, "%d", matrix.dims+i) != 1)
    {
      printf("fscanf failed\n");
      return matrix;
    }
    matrix.length *= matrix.dims[i];
  }
  /* allocate data */
  matrix.data = (float*)malloc(sizeof(float)*matrix.length);
  for(i = 0; i < matrix.length; i++)
  {
    if(fscanf(fp, "%f", matrix.data+i) != 1)
    {
      printf("fscanf failed\n");
      return matrix;
    }
  }

  return matrix;
}

void write_cumatrix(CUMATRIX *pmat, FILE *fp)
{
  int i;

  fprintf(fp, "%d\n", pmat->dims_num);
  for(i = 0; i < pmat->dims_num; i++)
    fprintf(fp, "%d ", pmat->dims[i]);
  fprintf(fp, "\n");
  for(i = 0; i < pmat->length; i++)
    fprintf(fp, "%.12f ", pmat->data[i]);
}

void print_cumatrix(CUMATRIX *pmat)
{
  int i;

  printf("dims_num = %d\n", pmat->dims_num);
  for(i = 0; i < pmat->dims_num; i++)
    printf("%d ", pmat->dims[i]);
  printf("\n");
  for(i = 0; i < pmat->length; i++)
    printf("%f ", pmat->data[i]);
  printf("\n");
}

void free_cumatrix(CUMATRIX *pmat)
{
  if(pmat->dims != NULL)
    free(pmat->dims);
  if(pmat->data != NULL)
    free(pmat->data);
}

CUMATRIX alloc_device_cumatrix(CUMATRIX mat_host)
{
  CUMATRIX mat_device;
  int dims_num = mat_host.dims_num;
  int length = mat_host.length;

  /* allocate device memory */
  mat_device.dims_num = dims_num;
  cutilSafeCall(cudaMalloc((void**)&(mat_device.dims), sizeof(int)*dims_num));
  mat_device.length = length;
  cutilSafeCall(cudaMalloc((void**)&(mat_device.data), sizeof(float)*length));

  /* copy host memory to device */
  cutilSafeCall(cudaMemcpy(mat_device.dims, mat_host.dims, sizeof(int)*dims_num, cudaMemcpyHostToDevice) );
  cutilSafeCall(cudaMemcpy(mat_device.data, mat_host.data, sizeof(float)*length, cudaMemcpyHostToDevice) );

  return mat_device;
}

void free_device_cumatrix(CUMATRIX *pmat)
{
  cutilSafeCall(cudaFree(pmat->dims));
  cutilSafeCall(cudaFree(pmat->data));
}

/* pad 3d matrix */
CUMATRIX pad_3d_maxtrix(CUMATRIX A, int padx, int pady)
{
  int x, y, z, nx, ny, nz;
  CUMATRIX B;

  nx = A.dims[1];
  ny = A.dims[0];
  nz = A.dims[2];
  
  B.dims_num = 3;
  B.dims = (int*)malloc(sizeof(int)*3);
  B.dims[0] = ny+2*pady;
  B.dims[1] = nx+2*padx;
  B.dims[2] = nz;
  B.length = B.dims[0] * B.dims[1] * B.dims[2];
  B.data = (float*)malloc(sizeof(float)*B.length);
  memset(B.data, 0, sizeof(float)*B.length);

  for(z = 0; z < B.dims[2]; z++)
  {
    for(x = padx; x < B.dims[1]-padx; x++) 
    {
      for(y = pady; y < B.dims[0]-pady; y++) 
        B.data[z*B.dims[0]*B.dims[1] + x*B.dims[0] + y] = A.data[z*nx*ny + (x-padx)*ny + y-pady];
    }
  }
  return B;  
}

/* copy matrix */
CUMATRIX copy_cumatrix(CUMATRIX A)
{
  CUMATRIX B;

  B.dims_num = A.dims_num;
  B.dims = (int*)malloc(sizeof(int)*B.dims_num);
  memcpy(B.dims, A.dims, sizeof(int)*B.dims_num);
  B.length = A.length;
  B.data = (float*)malloc(sizeof(float)*B.length);
  memcpy(B.data, A.data, sizeof(float)*B.length);

  return B;
}
