/*
 * mvt_tracker.h
 *
 *  Created on: Jul 10, 2013
 *      Author: changkyu
 */

#ifndef MVT_TRACKER_H_
#define MVT_TRACKER_H_

typedef class MVT_Tracker
{

public:
	MVT_Tracker(MVT_Param& param);
	~MVT_Tracker();

	MVT_State* Initialize(int idx_frame, const MVT_Image mvt_image, const MVT_Param& param);

	MVT_State* Update(int idx_frame, const MVT_Image mvt_image);

	void Train(int idx_frame, MVT_State& state_cur);

private:

	void Initialize_state_cur(const MVT_Param & param);

	void Update_Models(const MVT_Image mvt_image);
	void Update_Models(const MVT_Image mvt_image, MVT_2D_Object* pObject2d);

	MVT_SampleSet* m_pSamples;

	MVT_3D_Object* m_pObject3d;
	unsigned int m_num_of_viewpoint;
	unsigned int m_num_of_parts;
	unsigned int m_num_of_partsNroots;

	MVT_Potential_Model_Root**                     m_pModels_root;        // Likelihood Models using root            : 1D array of pointer
	MVT_Potential_Model_partsNroots***             m_pModels_partsNroots; // Likelihood Models using parts and roots : 2D array of pointer
	MVT_Potential_Model_Root**                     m_pModels_motion;      // Motion Models (Prior + Pairwise)        : 1D array of pointer

	cv::ObjectTrackerParams param_mil;

	bool is_vis;

	MVT_State m_state_cur;
	cv::Mat** m_pImages_rectified_cur;

	cv::Mat*   m_pImage;
	cv::Mat*** m_pImages_rectified; // 2D Array of images [viewpoint] x [partsNroots]

} MVT_Tracker;


#endif /* MVT_TRACKER_H_ */
