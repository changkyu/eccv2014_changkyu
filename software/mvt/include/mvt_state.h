/*
 * mvt_state.h
 *
 *  Created on: Jul 7, 2013
 *      Author: changkyu
 */

#ifndef MVT_STATE_H_
#define MVT_STATE_H_

void MVT_State_Copy(MVT_State* src, MVT_State* dest);
void MVT_State_Print(MVT_State* p_state, bool b_all);

class MVT_SampleSet
{
public:

	MVT_SampleSet(MVT_Param param,
			        MVT_3D_Object* pObject3d,
			        MVT_Potential_Model_Root**         models_root,
			        MVT_Potential_Model_partsNroots*** models_partsNroots,
			        MVT_Potential_Model_Root**         models_motion,
			        cv::Mat*** pImages_rectified);
	~MVT_SampleSet();

	MVT_State* CenterSampling(MVT_State* p_state_cur, cv::Mat* pImage);

	void ViewpointSampling(unsigned int idx_viewpoint, MVT_State* p_state_ref, cv::Mat* pImage);

	MVT_State* ComputePotentials(unsigned int idx_viewpoint, MVT_State* p_state_cur);

	void GetMaxPotential(MVT_State* &p_state, MVT_State* &p_state_online);

	MVT_Potentialmap* potential(unsigned int idx_viewpoint, unsigned idx_part)
	{
		return m_ppPotentialmaps[idx_viewpoint][idx_part];
	}

	MVT_2D_Object* Get2DObject(unsigned int idx_viewpoint){ return m_ppObject2ds[idx_viewpoint]; }
	unsigned int GetNumOfViewpoint(){ return m_num_of_viewpoint; }
	unsigned int GetNumOfSamples(){ return m_num_of_samples; }

	static void Draw(cv::Mat draw, MVT_State* p_state);

private:

	void ComputeOverlap();

	void ComputeLikelihoods_root(cv::Mat *pImage);
	void ComputeLikelihoods_partsNroots(unsigned int idx_viewpoint);

	void NormalizeLikelihoods_root_local();
	void NormalizeLikelihoods_partsNroots_local();

	void NormalizeLikelihoods_Global(MVT_State* states, unsigned int n_states);
	void Normalize_Init();
	void ComputeMotions(MVT_State* samples, unsigned int n_samples , MVT_State* p_state_cur);

	void GetMaxPotential_local(MVT_State* &p_state_max, MVT_State* &p_state_max_with_online);

	MVT_Viewpoint*                                m_pViewpoints;
	MVT_2D_Object**                               m_ppObject2ds;
	cv::Mat**             	                      m_pImages_rectified;
	cv::Mat*                                      m_pImage;

	MVT_Potential_Model_Root**                    m_models_root;
	MVT_Potential_Model_partsNroots***            m_models_partsNroots;
	MVT_Potential_Model_Root**                    m_models_motion;

	MVT_Potentialmap*                             m_pPotentialmap_root;
	MVT_Potentialmap***                           m_ppPotentialmaps;      /* (# of Viewpoint) x (# of PartsNViews) x (x,y) */

	MVT_State*                                    m_pSamples;

	MVT_Sampling* m_pSampling;

	MVT_3D_Object* m_pObject3d;

	unsigned int m_num_of_viewpoint;
	unsigned int m_num_of_center_sample;
	unsigned int m_num_of_samples;

	unsigned int m_num_of_parts;
	unsigned int m_num_of_roots;
	unsigned int m_num_of_partsNroots;

	double m_min_thresh_root[NUM_OF_LIKELIHOOD_TYPE_ROOT];
	double m_min_thresh_partsNroots[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];

	double m_min_root[NUM_OF_LIKELIHOOD_TYPE_ROOT];
	double m_max_root[NUM_OF_LIKELIHOOD_TYPE_ROOT];
	double m_dist_root[NUM_OF_LIKELIHOOD_TYPE_ROOT];
	double m_min_partsNroots[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	double m_min_partsNroots_raw[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	double m_max_partsNroots[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	double m_dist_partsNroots[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	double* m_min_partsNroots_pr[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	double* m_max_partsNroots_pr[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	double* m_dist_partsNroots_pr[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	MVT_State* m_ppSamples_LocalBest;
	MVT_State* m_ppSamples_LocalBest_online;

	unsigned int m_srchwinsz;
};

#endif /* MVT_STATE_H_ */
