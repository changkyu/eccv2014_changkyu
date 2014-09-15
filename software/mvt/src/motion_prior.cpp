/*
 * mvt_prior.cpp
 *
 *  Created on: Aug 22, 2013
 *      Author: changkyu
 */
#include "mvt.h"

using namespace boost::math;

typedef enum ENUM_IDX_VIEWPOINT
{
	IDX_AZIMUTH=0,
	IDX_ELEVATION,
	IDX_DISTANCE,
	NUM_IDX_VIEWPOINT
} ENUM_IDX_VIEWPOINT;

typedef enum ENUM_IDX_CENTER
{
	IDX_X=0,
	IDX_Y,
	NUM_IDX_CENTER
} ENUM_IDX_CENTER;

namespace mvt
{

void Prior::Initialize(MVT_Param param, MVT_3D_Object* pObject3d, bool b_viewpoint, bool b_center )
{
	m_std_azimuth   = param.std_prior_azimuth;
	m_std_elevation = param.std_prior_elevation;
	m_std_distance  = param.std_prior_distance;

	OnOff(b_viewpoint, b_center);

	m_nd = new boost::math::normal(0.0, 1.0);
	m_pdf_mean = pdf(*m_nd,0.0);

	m_num_of_partsNroots = pObject3d->Num_of_PartsNRoots();
	if(!m_state_prev.centers)           m_state_prev.centers           = (cv::Point2d*)malloc(sizeof(cv::Point2d)*m_num_of_partsNroots);
	if(!m_state_prev.centers_rectified) m_state_prev.centers_rectified = (cv::Point*)  malloc(sizeof(cv::Point)  *m_num_of_partsNroots);
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		if(!m_state_prev.likelihood_partsNroots_pr[m]) m_state_prev.likelihood_partsNroots_pr[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
		if(!m_state_prev.likelihood_partsNroots_pr_global[m]) m_state_prev.likelihood_partsNroots_pr_global[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
	}
	if(!m_pImages_rectified)
	{
		m_pImages_rectified = (cv::Mat**)malloc(sizeof(cv::Mat*)*m_num_of_partsNroots);
		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			m_pImages_rectified[pr] = new cv::Mat;
		}
	}
	if(!m_state_prev.pObject2d)         m_state_prev.pObject2d         = new MVT_2D_Object(pObject3d, m_pImages_rectified);
}

float Prior::GetPotential(MVT_State* p_state)
{
	// Assume that the array of states have a common viewpoint
	double prior_viewpoint = 1;
	if( (m_b_viewpoint) &&
		(m_state_prev.viewpoint.distance==m_state_prev.viewpoint.distance) )
	{
		double diff_azimuth_1 = abs(m_state_prev.viewpoint.azimuth  -p_state->viewpoint.azimuth);
		double diff_azimuth_2 = 360-abs(m_state_prev.viewpoint.azimuth  -p_state->viewpoint.azimuth);
		double diff_azimuth = diff_azimuth_1<diff_azimuth_2?diff_azimuth_1:diff_azimuth_2;

		prior_viewpoint *= pdf(*m_nd, diff_azimuth/m_std_azimuth   ) / m_pdf_mean;
		prior_viewpoint *= pdf(*m_nd, (m_state_prev.viewpoint.elevation-p_state->viewpoint.elevation)/m_std_elevation ) / m_pdf_mean;
		prior_viewpoint *= pdf(*m_nd, (m_state_prev.viewpoint.distance -p_state->viewpoint.distance )/m_std_distance  ) / m_pdf_mean;
	}

	double prior_center = 1;
	if( (m_b_center) &&
		(m_state_prev.viewpoint.distance==m_state_prev.viewpoint.distance) )
	{
#if 0
		prior_center *= pdf(*m_nd, (m_state_prev.center_root.x-p_state->center_root.x)/(m_state_prev.bbox_root.width *4)  ) / m_pdf_mean; // ORG *4 seq*1
		prior_center *= pdf(*m_nd, (m_state_prev.center_root.y-p_state->center_root.y)/(m_state_prev.bbox_root.height*4)  ) / m_pdf_mean; // ORG *4 seq*1
#else
		double unit_width  = m_state_prev.bbox_root.width *1;
		double unit_height = m_state_prev.bbox_root.height*1;

		if( p_state->likelihood_root[MVT_LIKELIHOOD_DPM] > g_param.thresh2_dpm )
		{
			unit_width  = m_state_prev.bbox_root.width *2;
			unit_height = m_state_prev.bbox_root.height*2;
		}

		prior_center *= pdf(*m_nd, (m_state_prev.center_root.x-p_state->center_root.x)/(unit_width)   ) / m_pdf_mean; // ORG *4 seq*1
		prior_center *= pdf(*m_nd, (m_state_prev.center_root.y-p_state->center_root.y)/(unit_height)  ) / m_pdf_mean; // ORG *4 seq*1
#endif
		/*
		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			if( !m_state_prev.pObject2d->IsOccluded(pr) && !p_state->pObject2d->IsOccluded(pr) )
			{
				prior_center *= pdf(*m_nd, (m_state_prev.centers[pr].x-p_state->centers[pr].x)/(m_state_prev.bbox_root.width*4)  ) / m_pdf_mean;
				prior_center *= pdf(*m_nd, (m_state_prev.centers[pr].y-p_state->centers[pr].y)/(m_state_prev.bbox_root.height*4)  ) / m_pdf_mean;
			}
		}
		*/
	}

	p_state->motion_prior = prior_viewpoint * prior_center;
	return p_state->motion_prior;
}


}
