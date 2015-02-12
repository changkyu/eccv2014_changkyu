/*
  convolution between hog features and hog templates
  author: Yu Xiang
  Date: 04/14/2011
*/

#ifndef CONVOLVE_H
#define CONVOLVE_H

#include "matrix.h"

CUMATRIX fconv(CUMATRIX A, CUMATRIX B);

#endif
