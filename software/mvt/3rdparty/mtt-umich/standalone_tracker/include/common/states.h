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

#ifndef _STATES_H_
#define _STATES_H_

#include <iostream>
#include <iomanip>
#include <common/global.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>

#define NUM_PERSON_SUBTYPE 		1
#define WH_PERSON_RATIO			3
#define MEAN_PERSON_HEIGHT 		1.7
#define STD_PERSON_HEIGHT		0.1
#define NUM_CAR_SUBTYPE		2
#define WH_CAR_RATIO1		0.5
#define WH_CAR_RATIO0		0.75
#define MEAN_CAR_HEIGHT 	1.2
#define STD_CAR_HEIGHT		0.2

namespace people {
	using namespace std;

	class ObjectState;
	typedef boost::shared_ptr<ObjectState> ObjectStatePtr;
	class ObjectState
	{
	public:
		ObjectState():timesec_(0.0),confidence_(0.0),sub_type_(0) {};
		virtual ~ObjectState() {};

		// copy state
		virtual void copyTo(ObjectStatePtr dst) = 0;
		// clone state
		virtual ObjectStatePtr clone() = 0;
		// for debug, show state
		virtual void print() = 0;
		// predict location of the state in timesec
		virtual ObjectStatePtr predict(double timesec) = 0;
		// draw a new sample
		virtual ObjectStatePtr drawSample(double timesec, const std::vector<double> &params) = 0;
		// draw a new sample
		virtual ObjectStatePtr perturbState(const std::vector<double> &params, double c = 1.0) = 0;
		// draw a new sample
		virtual double computeLogPrior(ObjectStatePtr state, double timesec, const std::vector<double> &params) = 0;

		// getter/setter for state elements
		virtual size_t numElement() = 0;
		virtual double getElement(int idx) = 0;
		virtual void setElement(int idx, double val) = 0;
		virtual cv::Mat getMatrix() = 0;
		
		// getter/setter for timestamp/confidence value
		inline double getTS() { return timesec_; }
		inline void setTS(const double &ts) { timesec_ = ts; }
		inline double getConfidence() { return confidence_; }
		inline void setConfidence(const double &confidence) { confidence_ = confidence; }
		// setter/getter object type/subtype
		inline void setObjType(ObjectType type) { obj_type_ = type; }
		inline ObjectType getObjType() { return obj_type_; }
		inline void setSubType(int type) { sub_type_ = type; }
		inline int getSubType() { return sub_type_; }

		inline std::string getStateType() { return state_type_; }
	protected:
		double 	timesec_;
		double 	confidence_;
		// human? car?
		ObjectType	obj_type_;
		// car side? front?
		int			sub_type_;

		std::string state_type_;
	};

	class FeatureState;
	typedef boost::shared_ptr<FeatureState> FeatureStatePtr;
	class FeatureState
	{
	public:
		FeatureState():timesec_(0.0),confidence_(0.0) {};
		virtual ~FeatureState() {};
		// copy state
		virtual void copyTo(FeatureStatePtr dst) = 0;
		// clone state
		virtual FeatureStatePtr clone() = 0;
		// for debug, show state
		virtual void print() = 0;
		// draw a new sample
		virtual FeatureStatePtr drawSample(double timesec, const std::vector<double> &params) = 0;
		// draw a new sample
		virtual FeatureStatePtr perturbState(const std::vector<double> &params, double c = 1.0) = 0;

		virtual double computeLogPrior(FeatureStatePtr state, double timesec, const std::vector<double> &params) = 0;
		// getter/setter for state elements
		virtual size_t numElement() = 0;
		virtual double getElement(int idx) = 0;
		virtual void setElement(int idx, double val) = 0;
		inline virtual cv::Mat getMatrix() = 0;

		// getter/setter for timestamp/confidence value
		inline double getTS() { return timesec_; }
		inline void setTS(const double &ts) { timesec_ = ts; }

		inline double getConfidence() { return confidence_; }
		inline void setConfidence(const double &confidence) { confidence_ = confidence; }

		inline std::string getStateType() { return state_type_; }
	protected:
		double timesec_;
		double confidence_;
		std::string state_type_;
	};

	class CameraState;
	typedef boost::shared_ptr<CameraState> CameraStatePtr;
	class CameraState
	{
	public:
		CameraState():timesec_(0.0) {};
		virtual ~CameraState() {};

		virtual CameraStatePtr clone() = 0;
		virtual CameraStatePtr predict(double timestamp) = 0;
		// draw a new sample
		virtual CameraStatePtr drawSample(double timesec, const std::vector<double> &params) = 0;
		// draw a new sample
		virtual CameraStatePtr perturbState(const std::vector<double> &params, double c = 1.0) = 0;

		virtual double computeLogPrior(CameraStatePtr state, double timesec, const std::vector<double> &params) = 0;

		virtual cv::Rect project(ObjectStatePtr state) = 0;
		virtual ObjectStatePtr iproject(const cv::Rect &rt) = 0;

		virtual cv::Point3f project(FeatureStatePtr state) = 0;
		virtual FeatureStatePtr iproject(const cv::Point2f &pt, const double depth = 0) = 0;

		virtual void print() = 0;

		// getter/setter for state elements
		virtual size_t numElement() = 0;
		virtual double getElement(int idx) = 0;
		virtual void setElement(int idx, double val) = 0;

		inline double getTS() { return timesec_; };
		inline void setTS(const double &ts) { timesec_ = ts; };

		inline std::string getStateType() { return state_type_; }
	protected:
		double timesec_;
		std::string state_type_;
	};
};
#endif // _STATES_H_
