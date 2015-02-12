/*
 * Software License Agreement (BSD License)
 * 
 * Copyright (c)  2012, Wongun Choi
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met: 
 * 
 * 1. Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer. 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution. 
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * 
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies, 
 * either expressed or implied, of the FreeBSD Project.
 */

#ifndef _GLOBAL_H_
#define _GLOBAL_H_

#include <opencv/cv.h>
namespace people {
	// random value generator
	extern cv::RNG		g_rng;

	typedef enum {
		ObjNone = -1,
		ObjPerson,
		ObjCar,
		ObjTypeNum,
	}ObjectType;
	extern ObjectType g_objtype;
#if 0
	typedef enum {
		ObjectStateTypeObjectLoc = 0,
		ObjectStateTypeObjectVel,
		ObjectStateTypeNum,
	}ObjectStateType;
	extern ObjectStateType g_obj_state_type_;

	typedef enum {
		FeatureStateTypeStatic = 0,
		FeatureStateTypeNum,
	}FeatureStateType;
	extern FeatureStateType g_feat_state_type_;

	typedef enum {
		CameraStateTypeSimplified = 0,
		CameraStateTypePinhole,
		CameraStateTypeNum,
	}CameraStateType;
	extern CameraStateType g_cam_state_type_;
#endif

	double gaussian_prob(double x, double m, double std);
	double log_gaussian_prob(double x, double m, double std);
	double log_gaussian_prob(cv::Mat &x, cv::Mat& m, cv::Mat &icov, double det);
};

#endif
