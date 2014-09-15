#ifndef __MVT_H__
#define __MVT_H__

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <malloc.h>
#include <assert.h>

#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <boost/math/distributions/normal.hpp>

#include <mat.h>

#include <MxArray.hpp>

extern "C"
{
#include "../3rdparty/ALM/svm_struct_cuda_mpi/svm_struct_api.h"
#include "../3rdparty/ALM/svm_struct_cuda_mpi/select_gpu.h"
#include "../3rdparty/ALM/svm_struct_cuda_mpi/rectify.h"
#include "../3rdparty/ALM/svm_struct_cuda_mpi/hog.h"
#include "../3rdparty/ALM/svm_struct_cuda_mpi/convolve.h"
}
#include "../3rdparty/gsoc11_tracking-master/include/object_tracker.h"

#ifndef HOGBINSIZE
#define HOGBINSIZE (6)
#endif

#define LOG(x) f_log << x; std::cout << x;

#include "mvt_types.h"
#include "mvt_param.h"
#include "mvt_2d_object.h"
#include "mvt_3d_object.h"
#include "mvt_model.h"
#include "mvt_sampling.h"
#include "mvt_state.h"

#include "detector_dpm.h"
#include "detector_alm.h"
#include "online_model_mil.h"
#include "motion_pairwise.h"
#include "motion_prior.h"

#include "mvt_tracker.h"
#include "mvt_timer.h"

extern std::ofstream f_result;
extern std::ofstream f_log;
extern bool g_b_initializing;

void compute_integral(const cv::Mat & img, std::vector<cv::Mat_<float> > & ii_imgs);

#endif
