/*
  Select a device according to rank
  author: Yu Xiang
  Date: 05/12/2011
*/

extern "C"
{
#include "select_gpu.h"
}
#include <cuda.h>
#if CUDA_VERSION>=5000
#include <helper_cuda.h>
#define cutilSafeCall checkCudaErrors
#else
#include "cutil_inline.h"
#endif

void select_gpu(int rank)
{
  /* get device count */
  int deviceCount = 0;
  if(cudaGetDeviceCount(&deviceCount) != cudaSuccess)
  {
    printf("cudaGetDeviceCount FAILED CUDA Driver and Runtime version may be mismatched.\n");
    exit(1);
  }
  deviceCount--;
  if(rank == 0)
    printf("%d CUDA enabled devices available.\n", deviceCount);

  // [ORG] changkyu int deviceID = rank % deviceCount + 1;
  int deviceID;
  if( deviceCount > 0 )
  {
  	deviceID = rank % deviceCount + 1;  	
  }
  else
  {
  	deviceID = 0;
  }
  printf("Rank is %d and Device ID is %d.\n", rank, deviceID);
  
  if(cudaSetDevice(deviceID) != cudaSuccess)
  {
    printf("cudaSetDevice FAILED\n");
    exit(1);
  }
  printf("Process %d is running on GPU %d.\n", rank, deviceID);
}
