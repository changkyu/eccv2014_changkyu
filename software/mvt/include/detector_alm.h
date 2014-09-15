#ifndef __DETECTOR_ALM_H__
#define __DETECTOR_ALM_H__

namespace mvt
{

typedef class DetectorALM : public MVT_Potential_Model_partsNroots
{
public:
	DetectorALM(ENUM_OBJECT_CATEGORY object_category, const char* filepath_alm_model, unsigned int idx_part);
	~DetectorALM();

	bool
	IsInitialize(){return true;}

	virtual void
	SetImage(CUMATRIX cumx_image)
	{
		free_cumatrix(&m_hog);
		m_hog = compute_hog_features(cumx_image, HOGBINSIZE);
	}

	float GetPotential(MVT_State* p_states){return 0;};

	void GetPotentials(std::vector<MVT_State*> &states, unsigned int n_states, float* p_potentials=NULL);

	double
	GetOccludedPotential();

private:

	float compute_similarity(cv::Point &center);

	ENUM_OBJECT_CATEGORY m_object_category;

	std::string           m_name;
	CUMATRIX              m_weight;
	float                 m_weight_occluded;
	unsigned int          m_weight_length;

	CUMATRIX m_hog;
}DetectorALM;

}

#endif
