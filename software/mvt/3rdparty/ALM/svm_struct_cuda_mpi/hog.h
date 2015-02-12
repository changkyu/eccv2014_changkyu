/*
  hog feature extraction
  author: Yu Xiang
  Date: 04/10/2011
*/

#ifndef HOG_H
#define HOG_H

#include "matrix.h"

CUMATRIX compute_gradient_image(CUMATRIX A);
CUMATRIX compute_hog_features(CUMATRIX B, int sbin);

#endif
