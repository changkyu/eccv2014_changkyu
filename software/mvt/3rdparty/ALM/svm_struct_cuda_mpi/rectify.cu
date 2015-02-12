/*
  rectify image according to a homography
  author: Yu Xiang
  Date: 04/10/2011
*/

extern "C"
{
#include "rectify.h"
#include "matrix.h"
}
#include <cuda.h>
#if CUDA_VERSION>=5000
#include <helper_cuda.h>
#define cutilSafeCall checkCudaErrors
#else
#include "cutil_inline.h"
#endif

#define BLOCK_SIZE 16

__constant__ float X_device[4];
__constant__ float Y_device[4];

__global__ void rectify(CUMATRIX B, CUMATRIX A);

CUMATRIX rectify_image(CUMATRIX A, float *H, float *T)
{
  int i;
  CUMATRIX A_device;
  CUMATRIX B, B_device;
  float X[4], Y[4], min_val, max_val;

  A_device = alloc_device_cumatrix(A);

  // allocate output image B
  B.dims_num = A.dims_num;
  B.dims = (int*)malloc(sizeof(int)*B.dims_num);
  memcpy(B.dims, A.dims, sizeof(int)*B.dims_num);

  // coordinates transformed by H
  X[0] = 0;
  Y[0] = 0;
  X[1] = A.dims[1]*H[0];
  Y[1] = A.dims[1]*H[1];
  X[2] = A.dims[0]*H[3];
  Y[2] = A.dims[0]*H[4];
  X[3] = X[1]+X[2];
  Y[3] = Y[1]+Y[2];

  // construct the transformation matrix
  T[0] = H[0];
  T[1] = H[1];
  T[3] = H[3];
  T[4] = H[4];
  min_val = X[0];
  if(X[1] < min_val)
    min_val = X[1];
  if(X[2] < min_val)
    min_val = X[2];
  if(X[3] < min_val)
    min_val = X[3];
  T[6] = -min_val;

  min_val = Y[0];
  if(Y[1] < min_val)
    min_val = Y[1];
  if(Y[2] < min_val)
    min_val = Y[2];
  if(Y[3] < min_val)
    min_val = Y[3];
  T[7] = -min_val;

  T[2] = T[5] = 0;
  T[8] = 1;

  // translate the points
  X[0] += T[6];
  X[1] += T[6];
  X[2] += T[6];
  X[3] += T[6];
  Y[0] += T[7];
  Y[1] += T[7];
  Y[2] += T[7];
  Y[3] += T[7];
  
  // out vector image
  min_val = X[0];
  max_val = X[0];
  if(X[1] > max_val)
      max_val = X[1];
  if(X[2] > max_val)
      max_val = X[2];
  if(X[3] > max_val)
      max_val = X[3];
  if(X[1] < min_val)
      min_val = X[1];
  if(X[2] < min_val)
      min_val = X[2];
  if(X[3] < min_val)
      min_val = X[3];
  B.dims[1] = int(max_val-min_val);
  
  min_val = Y[0];
  max_val = Y[0];
  if(Y[1] > max_val)
      max_val = Y[1];
  if(Y[2] > max_val)
      max_val = Y[2];
  if(Y[3] > max_val)
      max_val = Y[3];
  if(Y[1] < min_val)
      min_val = Y[1];
  if(Y[2] < min_val)
      min_val = Y[2];
  if(Y[3] < min_val)
      min_val = Y[3];
  B.dims[0] = int(max_val-min_val);

  B.length = 1;
  for(i = 0; i < B.dims_num; i++)
    B.length *= B.dims[i];
  B.data = (float*)malloc(sizeof(float)*B.length);
  
  B_device = alloc_device_cumatrix(B);

  // copy to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(X_device, X, sizeof(float)*4));
  cutilSafeCall(cudaMemcpyToSymbol(Y_device, Y, sizeof(float)*4));

  // setup execution parameters
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((B.dims[1]+BLOCK_SIZE-1) / BLOCK_SIZE, (B.dims[0]+BLOCK_SIZE-1) / BLOCK_SIZE);

  rectify<<< grid, threads >>>(B_device, A_device);
  cudaThreadSynchronize();

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(B.data, B_device.data, sizeof(float)*B.length, cudaMemcpyDeviceToHost) );
  
  free_device_cumatrix(&A_device);
  free_device_cumatrix(&B_device);
  return B;
}

__global__ void rectify(CUMATRIX B, CUMATRIX A)
{
  float x12, y12, x13, y13, xx, yy, a, b, d, xp, yp, cx[2], cy[2], ux, uy, val;
  int x, y, xi, yi, dx, dy;
  int nx, ny, i;

  nx = A.dims[1];
  ny = A.dims[0];
  
  x12 = (X_device[1]-X_device[0]) / (float)nx;
  y12 = (Y_device[1]-Y_device[0]) / (float)nx;
  x13 = (X_device[2]-X_device[0]) / (float)ny;
  y13 = (Y_device[2]-Y_device[0]) / (float)ny;
  xx = ((X_device[3]-X_device[0])*(Y_device[2]-Y_device[0])-(Y_device[3]-Y_device[0])*(X_device[2]-X_device[0]))/((X_device[1]-X_device[0])*(Y_device[2]-Y_device[0])-(Y_device[1]-Y_device[0])*(X_device[2]-X_device[0]));
  yy = ((X_device[3]-X_device[0])*(Y_device[1]-Y_device[0])-(Y_device[3]-Y_device[0])*(X_device[1]-X_device[0]))/((X_device[2]-X_device[0])*(Y_device[1]-Y_device[0])-(Y_device[2]-Y_device[0])*(X_device[1]-X_device[0]));
  a = (yy-1.0) / (1.0-xx-yy);
  b = (xx-1.0) / (1.0-xx-yy);

  x = blockIdx.x * blockDim.x + threadIdx.x;
  y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x < B.dims[1] && y < B.dims[0])
  {
    xx = 0.5 + (((float)x-X_device[0])*y13-((float)y-Y_device[0])*x13) / (x12*y13-y12*x13);
    yy = 0.5 - (((float)x-X_device[0])*y12-((float)y-Y_device[0])*x12) / (x12*y13-y12*x13);
    d = 1.0 - (a/(a+1.0))*xx/(float)nx - (b/(b+1.0))*yy/(float)ny;
    xp = xx / ((a+1.0)*d);
    yp = yy / ((b+1.0)*d);

    /* bilinear interpolation */
    if(xp < 0 || xp > (float)nx || yp < 0 || yp > (float)ny)
    {
      for(i = 0; i < B.dims[2]; i++)
        B.data[i*B.dims[0]*B.dims[1]+x*B.dims[0]+y] = 0;
    }
    else
    {
      xp -= 0.5; 
      yp -= 0.5;
      xi = (int)floorf((float)xp); 
      yi = (int)floorf((float)yp);
      ux = xp - (float)xi;
      uy = yp - (float)yi;
      cx[0] = ux;
      cx[1] = 1 - ux;
      cy[0] = uy;
      cy[1] = 1 - uy;

      for(i = 0; i < B.dims[2]; i++)
      {
        val = 0;
        for(dx = 0; dx <= 1; dx++)
        {
          for(dy = 0; dy <= 1; dy++)
            if(xi+dx >= 0 && xi+dx < nx && yi+dy >= 0 && yi+dy < ny)
              val += cx[1-dx] * cy[1-dy] * A.data[i*nx*ny+(xi+dx)*ny+yi+dy];
        }
        B.data[i*B.dims[0]*B.dims[1]+x*B.dims[0]+y] = val;
      }
    }
  }
}

/*
int main(int argc, char** argv)
{
  int i;
  FILE *fp;
  MATRIX A, A_device;
  MATRIX H;
  MATRIX B, B_device;
  float X[4], Y[4], min_val, max_val;
  float T[9];
  float *X_device, *Y_device;

  // load image A
  fp = fopen(argv[1], "r");
  if(fp == NULL)
  {
    printf("can not open file %s\n", argv[1]);
    return 1;
  }
  A = read_matrix(fp);
  fclose(fp);
  A_device = alloc_device_matrix(A);

  // load homography H
  fp = fopen(argv[2], "r");
  if(fp == NULL)
  {
    printf("can not open file %s\n", argv[2]);
    return 1;
  }
  H = read_matrix(fp);
  fclose(fp);

  // allocate output image B
  B.dims_num = A.dims_num;
  B.dims = (int*)malloc(sizeof(int)*B.dims_num);
  memcpy(B.dims, A.dims, sizeof(int)*B.dims_num);

  // coordinates transformed by H
  X[0] = 0;
  Y[0] = 0;
  X[1] = A.dims[1]*H.data[0];
  Y[1] = A.dims[1]*H.data[1];
  X[2] = A.dims[0]*H.data[3];
  Y[2] = A.dims[0]*H.data[4];
  X[3] = X[1]+X[2];
  Y[3] = Y[1]+Y[2];

  // construct the transformation matrix
  T[0] = H.data[0];
  T[1] = H.data[1];
  T[3] = H.data[3];
  T[4] = H.data[4];
  min_val = X[0];
  if(X[1] < min_val)
    min_val = X[1];
  if(X[2] < min_val)
    min_val = X[2];
  if(X[3] < min_val)
    min_val = X[3];
  T[6] = -min_val;

  min_val = Y[0];
  if(Y[1] < min_val)
    min_val = Y[1];
  if(Y[2] < min_val)
    min_val = Y[2];
  if(Y[3] < min_val)
    min_val = Y[3];
  T[7] = -min_val;

  T[2] = T[5] = 0;
  T[8] = 1;

  // translate the points
  X[0] += T[6];
  X[1] += T[6];
  X[2] += T[6];
  X[3] += T[6];
  Y[0] += T[7];
  Y[1] += T[7];
  Y[2] += T[7];
  Y[3] += T[7];
  
  // out vector image
  min_val = X[0];
  max_val = X[0];
  if(X[1] > max_val)
      max_val = X[1];
  if(X[2] > max_val)
      max_val = X[2];
  if(X[3] > max_val)
      max_val = X[3];
  if(X[1] < min_val)
      min_val = X[1];
  if(X[2] < min_val)
      min_val = X[2];
  if(X[3] < min_val)
      min_val = X[3];
  B.dims[1] = int(max_val-min_val);
  
  min_val = Y[0];
  max_val = Y[0];
  if(Y[1] > max_val)
      max_val = Y[1];
  if(Y[2] > max_val)
      max_val = Y[2];
  if(Y[3] > max_val)
      max_val = Y[3];
  if(Y[1] < min_val)
      min_val = Y[1];
  if(Y[2] < min_val)
      min_val = Y[2];
  if(Y[3] < min_val)
      min_val = Y[3];
  B.dims[0] = int(max_val-min_val);

  B.length = 1;
  for(i = 0; i < B.dims_num; i++)
    B.length *= B.dims[i];
  B.data = (float*)malloc(sizeof(float)*B.length);
  B_device = alloc_device_matrix(B);

  // allocate device X and Y
  cutilSafeCall(cudaMalloc((void**)&X_device, sizeof(float)*4));
  cutilSafeCall(cudaMemcpy(X_device, X, sizeof(float)*4, cudaMemcpyHostToDevice) );
  cutilSafeCall(cudaMalloc((void**)&Y_device, sizeof(float)*4));
  cutilSafeCall(cudaMemcpy(Y_device, Y, sizeof(float)*4, cudaMemcpyHostToDevice) );

  // setup execution parameters
  dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 grid((B.dims[1]+BLOCK_SIZE-1) / BLOCK_SIZE, (B.dims[0]+BLOCK_SIZE-1) / BLOCK_SIZE);

  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  for(i = 0; i < 1000; i++)
    rectify<<< grid, threads >>>(B_device, A_device, X_device, Y_device);
  cudaThreadSynchronize();

  // stop and destroy timer
  cutilCheckError(cutStopTimer(timer));
  float dSeconds = cutGetTimerValue(timer)/1000.0;
  cutilCheckError(cutDeleteTimer(timer));
  printf("time = %f\n", dSeconds);

  // copy result from device to host
  cutilSafeCall(cudaMemcpy(B.data, B_device.data, sizeof(float)*B.length, cudaMemcpyDeviceToHost) );

  fp = fopen("B.dat", "w");
  write_matrix(&B, fp);
  fclose(fp);
  
  free_device_matrix(&A_device);
  free_device_matrix(&B_device);
  free_matrix(&A);
  free_matrix(&H);
  free_matrix(&B);
  cutilSafeCall(cudaFree(X_device));
  cutilSafeCall(cudaFree(Y_device));
  return 0;
}
*/
