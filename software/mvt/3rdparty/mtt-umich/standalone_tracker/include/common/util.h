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

#ifndef _UTIL_H_
#define _UTIL_H_
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <common/states.h>
#include <common/lines.h>

namespace people {
	void show_image(cv::Mat &im, const std::string &name, int max_height);

	bool in_any_rect(const std::vector<cv::Rect> &rts, const cv::Point2f &pt);
	bool inrect(const cv::Rect &rt, const cv::Point2f &pt);

	float bb_overlap(const cv::Rect& rt1, const cv::Rect& rt2);
	float bb_intunion(const cv::Rect& rt1, const cv::Rect& rt2);
	bool file_exists(const std::string &filename);

	void nms(std::vector<cv::Rect>& bbs, std::vector<double>& confs);

	double state_ground_dist(ObjectStatePtr a, ObjectStatePtr b); // x-z distance
	double state_dist(ObjectStatePtr a, ObjectStatePtr b);
	double state_ground_vel_diff(ObjectStatePtr a, ObjectStatePtr b);
	double feat_state_dist(FeatureStatePtr a, FeatureStatePtr b);

	void getPairIndex(unsigned int min_idx, unsigned int max_idx, unsigned int &pair_index);
	void getPairFromIndex(unsigned int &min_idx, unsigned int &max_idx, unsigned int num_states, unsigned int pair_index);
	double getMinDist2Dets(const std::vector<cv::Rect> &dets, const cv::Rect &rt, const double &sx, const double &sy, const double &sh);
	double getMinDist2Dets(const std::vector<cv::Rect> &dets, int &idx, const cv::Rect &rt, const double &sx, const double &sy, const double &sh);
	double getDist2AllDets(const std::vector<cv::Rect> &dets, const cv::Rect &rt, const double &sx, const double &sy, const double &sh, const double &th);
	double getOverlap2AllDets(const std::vector<cv::Rect> &dets, const cv::Rect &rt, const double &th);
	double soft_max(double x, double scale);

	std::vector<std::string> read_file_list(const std::string &filename);

	void print_rect(cv::Rect &rt);
	void print_line(const longline &line);
	void print_matrix(cv::Mat &mat, bool dblprec = true);

	cv::Scalar get_target_color(int id);
#ifdef MYDEBUG
#define my_assert(x) assert(x)
	void open_dbg_file(const std::string &filename);
	void print_dbg_file(const char *msg);
	void print_dbg_file(const std::string &msg);
	void close_dbg_file();
#else
#define my_assert(x) 
#define open_dbg_file(x) 
#define print_dbg_file(x) 
#define close_dbg_file() 
#endif
};
#endif // _UTIL_H_
