#ifndef __MOTION_PRIOR_H__
#define __MOTION_PRIOR_H__

namespace mvt
{

typedef class Prior : public MVT_Potential_Model_Root
{
public:
	Prior()
	{
		m_state_prev.pObject2d = NULL;
		m_state_prev.centers = NULL;
		m_state_prev.centers_rectified = NULL;
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			m_state_prev.likelihood_partsNroots_pr[m] = NULL;
			m_state_prev.likelihood_partsNroots_pr_global[m] = NULL;
		}

		m_pImages_rectified = NULL;

		m_nd=NULL;
	}
	~Prior()
	{
		if(m_state_prev.pObject2d)         delete m_state_prev.pObject2d;
		if(m_state_prev.centers)           free(m_state_prev.centers);
		if(m_state_prev.centers_rectified) free(m_state_prev.centers_rectified);
		if(m_state_prev.likelihood_partsNroots_pr) free(m_state_prev.likelihood_partsNroots_pr);
		if(m_state_prev.likelihood_partsNroots_pr_global) free(m_state_prev.likelihood_partsNroots_pr_global);
		if(m_nd) delete m_nd;
	}

	void Initialize(MVT_Param param, MVT_3D_Object* pObject3d , bool b_viewpoint, bool b_center );

	bool IsInitialize(){return (m_nd!=NULL);}

	void SetPrevState(MVT_State &state_prev)
	{
		MVT_State_Copy(&state_prev, &m_state_prev);
	}

	void OnOff( bool b_viewpoint, bool b_center )
	{
		m_b_viewpoint = b_viewpoint;
		m_b_center = b_center;
	}


	float GetPotential(MVT_State* p_states);

private:

	MVT_State m_state_prev;
	cv::Mat** m_pImages_rectified;

	boost::math::normal *m_nd;
	double m_pdf_mean;

	double m_std_azimuth;
	double m_std_elevation;
	double m_std_distance;

	bool m_b_viewpoint;
	bool m_b_center;

	unsigned int m_num_of_partsNroots;

} Prior;

}

#endif
