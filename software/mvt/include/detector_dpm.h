#ifndef __DETECTOR_DPM_H__
#define __DETECTOR_DPM_H__

#include "observation/DetectionReadinNode.h"
using namespace people;

namespace mvt
{

typedef class DetectorDPM : public MVT_Potential_Model_Root
{
public:
	DetectorDPM(MVT_Param &param);
	~DetectorDPM();

	void SetConf(char* filepath_conf);

	void SetImage(cv::Mat *pImage);

	bool IsInitialize(){return true;};

	float GetPotential(MVT_State* p_states);

private:

	boost::math::normal *m_nd;
	double m_pdf_mean;

	int version_conf;
	bool readConf(char* filepath_conf);

	std::vector<DetectionReadinConfidence> m_confidences;

	float m_confidence_outofimage;

	MATFile* m_mat;
	mxArray* m_model;

	double m_x_min;
	double m_x_max;
	double m_y_min;
	double m_y_max;

}DetectorDPM;

}

#endif
