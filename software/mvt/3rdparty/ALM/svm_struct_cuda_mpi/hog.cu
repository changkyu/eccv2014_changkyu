/*
  hog feature extraction
  author: Yu Xiang
  Date: 04/10/2011
*/

extern "C"
{
#include "matrix.h"
#include "hog.h"
}
#include <cuda.h>
#if CUDA_VERSION>=5000
#include <helper_cuda.h>
#define cutilSafeCall checkCudaErrors
#else
#include "cutil_inline.h"
#endif

// unit vectors used to compute gradient orientation
float uu[9] = {1.0000, 
		0.9397, 
		0.7660, 
		0.500, 
		0.1736, 
		-0.1736, 
		-0.5000, 
		-0.7660, 
		-0.9397};
float vv[9] = {0.0000, 
		0.3420, 
		0.6428, 
		0.8660, 
		0.9848, 
		0.9848, 
		0.8660, 
		0.6428, 
		0.3420};

__constant__ float uu_dev[9];
__constant__ float vv_dev[9];

__global__ void compute_histogram_norm(CUMATRIX H, CUMATRIX N, CUMATRIX A, int sbin);
__global__ void compute_hog(CUMATRIX HOG, CUMATRIX H, CUMATRIX N);

/* compute gradient image */
CUMATRIX compute_gradient_image(CUMATRIX A)
{
  int x, y, nx, ny;
  CUMATRIX B;

  nx = A.dims[1];
  ny = A.dims[0];
  B.dims_num = 3;
  B.dims = (int*)malloc(sizeof(int)*3);
  B.dims[0] = ny;
  B.dims[1] = nx;
  B.dims[2] = 2;
  B.length = nx*ny*2;
  B.data = (float*)malloc(sizeof(float)*B.length);
  memset(B.data, 0, sizeof(float)*B.length);

  for(x = 1; x < nx-1; x++) 
  {
    for(y = 1; y < ny-1; y++) 
    {
      // first color channel
      float *s = A.data + x*ny + y;
      float dy = *(s+1) - *(s-1);
      float dx = *(s+ny) - *(s-ny);
      float v = dx*dx + dy*dy;

      // second color channel
      s += nx*ny;
      float dy2 = *(s+1) - *(s-1);
      float dx2 = *(s+ny) - *(s-ny);
      float v2 = dx2*dx2 + dy2*dy2;

      // third color channel
      s += nx*ny;
      float dy3 = *(s+1) - *(s-1);
      float dx3 = *(s+ny) - *(s-ny);
      float v3 = dx3*dx3 + dy3*dy3;

      // pick channel with strongest gradient
      if (v2 > v) {
	v = v2;
	dx = dx2;
	dy = dy2;
      } 
      if (v3 > v) {
	v = v3;
	dx = dx3;
	dy = dy3;
      }
      B.data[x*ny+y] = dx;
      B.data[nx*ny+x*ny+y] = dy;
    }
  }
  return B;  
}

/* compute hog feature */
CUMATRIX compute_hog_features(CUMATRIX B, int sbin)
{
  CUMATRIX B_device;
  CUMATRIX H_device, N_device;
  CUMATRIX HOG, HOG_device;

  B_device = alloc_device_cumatrix(B);

  // block dimension
  int blocks[2];
  blocks[0] = (int)round((float)B.dims[0]/(float)sbin);
  blocks[1] = (int)round((float)B.dims[1]/(float)sbin);

  // allocate device memory for histograms and their norm
  int dims[3];
  dims[0] = blocks[0];
  dims[1] = blocks[1];
  dims[2] = 18;
  H_device.dims_num = 3;
  cutilSafeCall(cudaMalloc((void**)&(H_device.dims), sizeof(int)*3));
  cutilSafeCall(cudaMemcpy(H_device.dims, dims, sizeof(int)*3, cudaMemcpyHostToDevice) );
  H_device.length = dims[0]*dims[1]*dims[2];
  cutilSafeCall(cudaMalloc((void**)&(H_device.data), sizeof(float)*H_device.length));
  cutilSafeCall(cudaMemset(H_device.data, 0, sizeof(float)*H_device.length));

  N_device.dims_num = 2;
  cutilSafeCall(cudaMalloc((void**)&(N_device.dims), sizeof(int)*2));
  cutilSafeCall(cudaMemcpy(N_device.dims, blocks, sizeof(int)*2, cudaMemcpyHostToDevice) );
  N_device.length = blocks[0]*blocks[1];
  cutilSafeCall(cudaMalloc((void**)&(N_device.data), sizeof(float)*N_device.length));
  cutilSafeCall(cudaMemset(N_device.data, 0, sizeof(float)*N_device.length));

  // allocate memory for HOG features
  HOG.dims_num = 3;
  HOG.dims = (int*)malloc(sizeof(int)*3);
  HOG.dims[0] = blocks[0];
  HOG.dims[1] = blocks[1];
  HOG.dims[2] = 32;
  HOG.length = blocks[0]*blocks[1]*32;
  HOG.data = (float*)malloc(sizeof(float)*HOG.length);
  HOG_device = alloc_device_cumatrix(HOG);
  cutilSafeCall(cudaMemset(HOG_device.data, 0, sizeof(float)*HOG_device.length));

  // copy to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(uu_dev, uu, sizeof(float)*9));
  cutilSafeCall(cudaMemcpyToSymbol(vv_dev, vv, sizeof(float)*9));

  // launch kernel
  compute_histogram_norm<<< blocks[1], blocks[0] >>>(H_device, N_device, B_device, sbin);
  cudaThreadSynchronize();
  compute_hog<<< blocks[1], blocks[0] >>>(HOG_device, H_device, N_device);
  cudaThreadSynchronize();

  cutilSafeCall(cudaMemcpy(HOG.data, HOG_device.data, sizeof(float)*HOG.length, cudaMemcpyDeviceToHost) );

  free_device_cumatrix(&HOG_device);
  free_device_cumatrix(&B_device);
  free_device_cumatrix(&H_device);
  free_device_cumatrix(&N_device);
  return HOG;
}

__global__ void compute_histogram_norm(CUMATRIX H, CUMATRIX N, CUMATRIX A, int sbin)
{
  // image dimension
  int nx = A.dims[1];
  int ny = A.dims[0];

  int blocks[2];
  blocks[0] = blockDim.x;
  blocks[1] = gridDim.x;

  // hog block index
  int bx = blockIdx.x;
  int by = threadIdx.x;

  // start and end indexes of pixels contribute this hog block
  int xstart, xend, ystart, yend;
  if(bx*sbin-sbin/2 <= 0)
    xstart = 1;
  else
    xstart = bx*sbin-sbin/2;
  
  if(bx*sbin+3*sbin/2 >= nx - 1)
    xend = nx - 2;
  else
    xend = bx*sbin+3*sbin/2-1;

  if(by*sbin-sbin/2 <= 0)
    ystart = 1;
  else
    ystart = by*sbin-sbin/2;
  
  if(by*sbin+3*sbin/2 >= ny - 1)
    yend = ny - 2;
  else
    yend = by*sbin+3*sbin/2-1;

  // compute histogram of gradients
  for(int x = xstart; x <= xend; x++)
  {
    for(int y = ystart; y <= yend; y++)
    {
      // first color channel
      float *s = A.data + min(x, nx-2)*ny + min(y, ny-2);
      float dx = *s;
      // second color channel
      s += nx*ny;
      float dy = *s;
      float v = dx*dx + dy*dy;

      // snap to one of 18 orientations
      float best_dot = 0;
      int best_o = 0;
      for (int o = 0; o < 9; o++) 
      {
        float dot = uu_dev[o]*dx + vv_dev[o]*dy;
        if (dot > best_dot) 
        {
          best_dot = dot;
          best_o = o;
        }
        else if (-dot > best_dot) 
        {
	  best_dot = -dot;
	  best_o = o+9;
        }
      }

      // add to 4 histograms around pixel using linear interpolation
      float xp = ((float)x+0.5)/(float)sbin - 0.5;
      float yp = ((float)y+0.5)/(float)sbin - 0.5;
      int ixp = (int)floorf(xp);
      int iyp = (int)floorf(yp);
      float vx0 = xp-ixp;
      float vy0 = yp-iyp;
      float vx1 = 1.0-vx0;
      float vy1 = 1.0-vy0;
      v = sqrtf(v);

      if(ixp == bx && iyp == by)
        *(H.data + bx*blocks[0] + by + best_o*blocks[0]*blocks[1]) += vx1*vy1*v;
      else if(ixp+1 == bx && iyp == by)
        *(H.data + bx*blocks[0] + by + best_o*blocks[0]*blocks[1]) += vx0*vy1*v;
      else if(ixp == bx && iyp+1 == by)
        *(H.data + bx*blocks[0] + by + best_o*blocks[0]*blocks[1]) += vx1*vy0*v;
      else if(ixp+1 == bx && iyp+1 == by)
        *(H.data + bx*blocks[0] + by + best_o*blocks[0]*blocks[1]) += vx0*vy0*v;
    }
  }

  // compute norm of histogram
  for (int o = 0; o < 9; o++)
  {
    float *src1 = H.data + bx*blocks[0] + by + o*blocks[0]*blocks[1];
    float *src2 = H.data + bx*blocks[0] + by + (o+9)*blocks[0]*blocks[1];
    *(N.data + bx*blocks[0] + by) += (*src1 + *src2) * (*src1 + *src2);
  }
}

__global__ void compute_hog(CUMATRIX HOG, CUMATRIX H, CUMATRIX N)
{
  float f[32];

  int blocks[2];
  blocks[0] = blockDim.x;
  blocks[1] = gridDim.x;

  // hog block index
  int bx = blockIdx.x;
  int by = threadIdx.x;

  // compute hog features
  if(bx == 0 || bx == blocks[1]-1 || by == 0 || by == blocks[0]-1)
      *(HOG.data + bx*blocks[0] + by + 31*blocks[0]*blocks[1]) = 1.0;
  else
  {
    float *dst = f;      
    float *src, *p, n1, n2, n3, n4;
    float eps = 0.0001;

    p = N.data + bx*blocks[0] + by;
    n1 = 1.0 / sqrtf(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
    p = N.data + bx*blocks[0] + by-1;
    n2 = 1.0 / sqrtf(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
    p = N.data + (bx-1)*blocks[0] + by;
    n3 = 1.0 / sqrtf(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);
    p = N.data + (bx-1)*blocks[0] + by-1;
    n4 = 1.0 / sqrtf(*p + *(p+1) + *(p+blocks[0]) + *(p+blocks[0]+1) + eps);

    float t1 = 0;
    float t2 = 0;
    float t3 = 0;
    float t4 = 0;

    // contrast-sensitive features
    src = H.data + bx*blocks[0] + by;
    for (int o = 0; o < 18; o++) 
    {
      float h1 = min(*src * n1, 0.2);
      float h2 = min(*src * n2, 0.2);
      float h3 = min(*src * n3, 0.2);
      float h4 = min(*src * n4, 0.2);
      *dst = 0.5 * (h1 + h2 + h3 + h4);
      t1 += h1;
      t2 += h2;
      t3 += h3;
      t4 += h4;
      dst++;
      src += blocks[0]*blocks[1];
    }

    // contrast-insensitive features
    src = H.data + bx*blocks[0] + by;
    for (int o = 0; o < 9; o++)
    {
      float sum = *src + *(src + 9*blocks[0]*blocks[1]);
      float h1 = min(sum * n1, 0.2);
      float h2 = min(sum * n2, 0.2);
      float h3 = min(sum * n3, 0.2);
      float h4 = min(sum * n4, 0.2);
      *dst = 0.5 * (h1 + h2 + h3 + h4);
      dst++;
      src += blocks[0]*blocks[1];
    }

    // texture features
    *dst = 0.2357 * t1;
    dst++;
    *dst = 0.2357 * t2;
    dst++;
    *dst = 0.2357 * t3;
    dst++;
    *dst = 0.2357 * t4;

    // truncation feature
    dst++;
    *dst = 1;

    /* assign features */
    dst = HOG.data + bx*blocks[0] + by;
    for(int i = 0; i < 32; i++)
    {
      *dst = f[i];
       dst += blocks[0]*blocks[1];
    }
  }
}

/*
int main(int argc, char** argv)
{
  FILE *fp;
  CUMATRIX A;
  CUMATRIX B, B_device;
  CUMATRIX H_device, N_device;
  CUMATRIX HOG, HOG_device;

  fp = fopen(argv[1], "r");
  if(fp == NULL)
  {
    printf("can not open file %s\n", argv[1]);
    return 1;
  }
  A = read_cumatrix(fp);
  fclose(fp);

  // compute gradient image
  B = compute_gradient_image(A);
  B_device = alloc_device_cumatrix(B);

  // block dimension
  int sbin = 6;
  int blocks[2];
  blocks[0] = (int)round((float)B.dims[0]/(float)sbin);
  blocks[1] = (int)round((float)B.dims[1]/(float)sbin);

  // allocate device memory for histograms and their norm
  int dims[3];
  dims[0] = blocks[0];
  dims[1] = blocks[1];
  dims[2] = 18;
  H_device.dims_num = 3;
  cutilSafeCall(cudaMalloc((void**)&(H_device.dims), sizeof(int)*3));
  cutilSafeCall(cudaMemcpy(H_device.dims, dims, sizeof(int)*3, cudaMemcpyHostToDevice) );
  H_device.length = dims[0]*dims[1]*dims[2];
  cutilSafeCall(cudaMalloc((void**)&(H_device.data), sizeof(float)*H_device.length));
  cutilSafeCall(cudaMemset(H_device.data, 0, sizeof(float)*H_device.length));

  N_device.dims_num = 2;
  cutilSafeCall(cudaMalloc((void**)&(N_device.dims), sizeof(int)*2));
  cutilSafeCall(cudaMemcpy(N_device.dims, blocks, sizeof(int)*2, cudaMemcpyHostToDevice) );
  N_device.length = blocks[0]*blocks[1];
  cutilSafeCall(cudaMalloc((void**)&(N_device.data), sizeof(float)*N_device.length));
  cutilSafeCall(cudaMemset(N_device.data, 0, sizeof(float)*N_device.length));

  // allocate memory for HOG features
  HOG.dims_num = 3;
  HOG.dims = (int*)malloc(sizeof(int)*3);
  HOG.dims[0] = blocks[0];
  HOG.dims[1] = blocks[1];
  HOG.dims[2] = 32;
  HOG.length = blocks[0]*blocks[1]*32;
  HOG.data = (float*)malloc(sizeof(float)*HOG.length);
  HOG_device = alloc_device_cumatrix(HOG);
  cutilSafeCall(cudaMemset(HOG_device.data, 0, sizeof(float)*HOG_device.length));

  // copy to constant memory
  cutilSafeCall(cudaMemcpyToSymbol(uu_dev, uu, sizeof(float)*9));
  cutilSafeCall(cudaMemcpyToSymbol(vv_dev, vv, sizeof(float)*9));

  unsigned int timer = 0;
  cutilCheckError(cutCreateTimer(&timer));
  cutilCheckError(cutStartTimer(timer));

  // launch kernel

  compute_histogram_norm<<< blocks[1], blocks[0] >>>(H_device, N_device, B_device, sbin);
  cudaThreadSynchronize();
  compute_hog<<< blocks[1], blocks[0] >>>(HOG_device, H_device, N_device);
  cudaThreadSynchronize();

  cutilCheckError(cutStopTimer(timer));
  float dSeconds = cutGetTimerValue(timer)/1000.0;
  cutilCheckError(cutDeleteTimer(timer));
  printf("time = %f\n", dSeconds);

  cutilSafeCall(cudaMemcpy(HOG.data, HOG_device.data, sizeof(float)*HOG.length, cudaMemcpyDeviceToHost) );

  fp = fopen(argv[2], "w");
  write_cumatrix(&HOG, fp);
  fclose(fp);

  free_device_cumatrix(&HOG_device);
  free_device_cumatrix(&B_device);
  free_device_cumatrix(&H_device);
  free_device_cumatrix(&N_device);
  free_cumatrix(&A);
  free_cumatrix(&B);
  free_cumatrix(&HOG);
}
*/
