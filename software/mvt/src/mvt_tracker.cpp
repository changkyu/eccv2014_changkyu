/*
 * mvt_tracker.cpp
 *
 *  Created on: Jul 10, 2013
 *      Author: changkyu
 */

#include "mvt.h"

MVT_Tracker::MVT_Tracker(MVT_Param& param)
{
	is_vis = param.is_vis;
	m_pObject3d          = new MVT_3D_Object(param.object_category,param.filepath_3dobject_model.c_str());
	m_num_of_parts       = m_pObject3d->Num_of_Parts();
	m_num_of_partsNroots = m_pObject3d->Num_of_PartsNRoots();
	m_num_of_viewpoint   = param.num_of_viewpoint_sample;

	m_pImage = NULL;
	m_pImages_rectified = (cv::Mat***)malloc(sizeof(cv::Mat**)*m_num_of_viewpoint);
	for( unsigned int v=0; v<m_num_of_viewpoint; v++ )
	{
		m_pImages_rectified[v] = (cv::Mat**)malloc(sizeof(cv::Mat*)*m_num_of_partsNroots);
		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			m_pImages_rectified[v][pr] = new cv::Mat();
		}
	}

	// Likelihood Models using only root part
	m_pModels_root = (MVT_Potential_Model_Root**) malloc(sizeof(MVT_Potential_Model_Root*) * NUM_OF_LIKELIHOOD_TYPE_ROOT);
	if( param.use_dpm == false )
	{
		m_pModels_root[MVT_LIKELIHOOD_DPM] = NULL;
	}
	else
	{
		m_pModels_root[MVT_LIKELIHOOD_DPM] = new mvt::DetectorDPM(param);
	}

	if( param.use_mil_root == false )
	{
		m_pModels_root[MVT_LIKELIHOOD_MIL_ROOT] = NULL;
	}
	else
	{
		m_pModels_root[MVT_LIKELIHOOD_MIL_ROOT] = new mvt::OnlineMILModel(0,0,-1);
		((mvt::OnlineMILModel*)m_pModels_root[MVT_LIKELIHOOD_MIL_ROOT])->SetWeight(param.weight_mil_root);
	}

	// Likelihood Models using parts and roots
	m_pModels_partsNroots = (MVT_Potential_Model_partsNroots***)malloc(sizeof(MVT_Potential_Model_partsNroots**) *m_num_of_partsNroots);
	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		MVT_2D_Part_Front part2d_front = m_pObject3d->GetPartFrontInfo(pr);
		m_pModels_partsNroots[pr] = (MVT_Potential_Model_partsNroots**)malloc( sizeof(MVT_Potential_Model_partsNroots*) * NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS );

		if( (param.use_alm == false) /*|| (param.use_only_root && pr<m_num_of_parts)*/ )
		{
			m_pModels_partsNroots[pr][MVT_LIKELIHOOD_ALM] = NULL;
		}
		else
		{
			m_pModels_partsNroots[pr][MVT_LIKELIHOOD_ALM] = new mvt::DetectorALM(param.object_category,param.filepath_3dobject_model.c_str(),pr);
		}

		if(param.use_mil == false)
		{
			m_pModels_partsNroots[pr][MVT_LIKELIHOOD_MIL] = NULL;
		}
		else
		{
			if( pr<m_num_of_parts )
			{
				m_pModels_partsNroots[pr][MVT_LIKELIHOOD_MIL] = new mvt::OnlineMILModel(part2d_front.width, part2d_front.height,pr);
			}
			else
			{
				m_pModels_partsNroots[pr][MVT_LIKELIHOOD_MIL] = NULL;
			}
		}
	}

	// Motion Models
	m_pModels_motion = (MVT_Potential_Model_Root**) malloc(sizeof(MVT_Potential_Model_Root*) * NUM_OF_MOTION);
	if( param.use_pairwise )
	{
		m_pModels_motion[MVT_MOTION_PAIRWISE] = new mvt::Pairwise();
		((mvt::Pairwise*)m_pModels_motion[MVT_MOTION_PAIRWISE])->Initialize();
	}
	else
	{
		m_pModels_motion[MVT_MOTION_PAIRWISE] = NULL;
	}
	if( param.use_prior )
	{
		m_pModels_motion[MVT_MOTION_PRIOR]    = new mvt::Prior();
		((mvt::Prior*)m_pModels_motion[MVT_MOTION_PRIOR])->Initialize(param, m_pObject3d, param.use_alm || param.use_mil, true);
	}
	else
	{
		m_pModels_motion[MVT_MOTION_PRIOR]    = NULL;
	}

	m_pSamples = new MVT_SampleSet(param, m_pObject3d, m_pModels_root, m_pModels_partsNroots, m_pModels_motion, m_pImages_rectified);

	param_mil = param.param_mil;
}

MVT_Tracker::~MVT_Tracker()
{
	delete m_pSamples;

	for( int m=0; m<(int)NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		delete m_pModels_root[m];
	}
	free( m_pModels_root );

	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		for( int m=0; m<(int)NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			if( m_pModels_partsNroots[pr][m] ) delete m_pModels_partsNroots[pr][m];
		}
		free(m_pModels_partsNroots[pr]);
	}
	free(m_pModels_partsNroots);

	for( int m=0; m<(int)NUM_OF_MOTION; m++ )
	{
		delete m_pModels_motion[m];
	}
	free( m_pModels_motion );

	if( m_state_cur.pObject2d ) delete m_state_cur.pObject2d;
	free(m_state_cur.centers);
	free(m_state_cur.centers_rectified);
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		free(m_state_cur.likelihood_partsNroots_pr[m]);
		free(m_state_cur.likelihood_partsNroots_pr_global[m]);
	}

	delete m_pObject3d;

}

static cv::ObjectTrackerParams param_mil;

void MVT_Tracker::Initialize_state_cur(const MVT_Param & param)
{
	if( param.use_init )
	{
		MVT_Viewpoint viewpoint_init = {param.init_state_a, param.init_state_e, param.init_state_d};

		m_state_cur.idx_viewpoint = -1;
		m_state_cur.viewpoint = viewpoint_init;
		m_state_cur.pObject2d = new MVT_2D_Object(m_pObject3d, m_pImages_rectified_cur);
		m_state_cur.pObject2d->SetViewpoint(viewpoint_init, m_pImage);

		m_state_cur.centers           = (cv::Point2d*)malloc(sizeof(cv::Point2d)*m_num_of_partsNroots);
		m_state_cur.centers_rectified = (cv::Point*  )malloc(sizeof(cv::Point  )*m_num_of_partsNroots);
		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++)
		{
			if( !m_state_cur.pObject2d->IsOccluded(pr) )
			{
				m_state_cur.centers[pr] = cv::Point2d(
				param.init_state_x + m_state_cur.pObject2d->m_2dparts[pr].center.x,
				param.init_state_y + m_state_cur.pObject2d->m_2dparts[pr].center.y   );
				if( !(pr<m_num_of_parts) )
				{
					m_state_cur.center_root = m_state_cur.centers[pr];
				}

				m_state_cur.centers_rectified[pr] = m_state_cur.pObject2d->GetRectifiedPoint(pr,m_state_cur.centers[pr]);
			}
		}
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			m_state_cur.likelihood_partsNroots_pr[m]        = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
			m_state_cur.likelihood_partsNroots_pr_global[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
		}
		m_state_cur.bbox_root        = m_state_cur.pObject2d->GetTargetBoundingBox(m_state_cur);
		m_state_cur.bbox_partsNroots = m_state_cur.bbox_root;
		m_state_cur.potential = -INFINITY;
	}
	else
	{
		MVT_Viewpoint viewpoint_init = {NAN, NAN, NAN};

		m_state_cur.idx_viewpoint = -1;
		m_state_cur.viewpoint = viewpoint_init;
		m_state_cur.pObject2d = new MVT_2D_Object(m_pObject3d, m_pImages_rectified_cur);
		m_state_cur.pObject2d->SetViewpoint(viewpoint_init, m_pImage);

		m_state_cur.centers           = (cv::Point2d*)malloc(sizeof(cv::Point2d)*m_num_of_partsNroots);
		m_state_cur.centers_rectified = (cv::Point*  )malloc(sizeof(cv::Point  )*m_num_of_partsNroots);
		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++)
		{
			m_state_cur.centers[pr] = cv::Point2d(NAN,NAN);
			if( !(pr<m_num_of_parts) )
			{
				m_state_cur.center_root = m_state_cur.centers[pr];
			}

			m_state_cur.centers_rectified[pr] = cv::Point(NAN,NAN);
		}
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			m_state_cur.likelihood_partsNroots_pr[m]        = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
			m_state_cur.likelihood_partsNroots_pr_global[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
		}
		m_state_cur.bbox_root        = cv::Rect(NAN,NAN,NAN,NAN);
		m_state_cur.bbox_partsNroots = m_state_cur.bbox_root;
		m_state_cur.potential = -INFINITY;
	}
}


MVT_State* MVT_Tracker::Initialize(int idx_frame, MVT_Image mvt_image, const MVT_Param& param)
{
	m_pImage = mvt_image.pImage;

	m_pImages_rectified_cur = (cv::Mat**)malloc(sizeof(cv::Mat*)*m_num_of_partsNroots);
	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		m_pImages_rectified_cur[pr] = new cv::Mat();
	}

	Initialize_state_cur(param);

	if( param.use_init )
	{
		Train(-1, m_state_cur);
	}
	else
	{
		Update(idx_frame, mvt_image);
	}

	return &m_state_cur;
}

void MVT_Tracker::Update_Models(const MVT_Image mvt_image)
{
	for( unsigned int m=0; m<(unsigned int)NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		if( m_pModels_root[m]!=NULL )
		{
			switch(m)
			{
				case MVT_LIKELIHOOD_DPM:
				{
					m_pModels_root[m]->SetImage(mvt_image);
				}
				break;

				case MVT_LIKELIHOOD_MIL_ROOT:
				{
					m_pModels_root[m]->SetImage(m_pImage);
				}
				break;
			}
		}
	}
}

void MVT_Tracker::Update_Models(const MVT_Image mvt_image, MVT_2D_Object* pObject2d)
{
	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		bool is_occluded = pObject2d->IsOccluded(pr);

		for( unsigned int m=0; m<(unsigned int)NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			if( m_pModels_partsNroots[pr][m] )
			{
				m_pModels_partsNroots[pr][m]->SetOccluded(is_occluded);
				if( (is_occluded==false) && (m_pModels_partsNroots[pr][m]->IsInitialize()) )
				{
					switch(m)
					{
						case MVT_LIKELIHOOD_ALM:
						{
							CUMATRIX image_gradient_rectified = pObject2d->GetRectifiedGradientImage(pr);
							m_pModels_partsNroots[pr][m]->SetImage(image_gradient_rectified);
						}
						break;

						case MVT_LIKELIHOOD_MIL:
						{
							cv::Mat* pImage_rectified = pObject2d->GetRectifiedImage(pr);
							m_pModels_partsNroots[pr][m]->SetImage(pImage_rectified);
						}
						break;
					}
				}
			}
		}
	}
}

MVT_State* MVT_Tracker::Update(int idx_frame, const MVT_Image mvt_image)
{
	m_pImage = mvt_image.pImage;

	Update_Models(mvt_image);
	MVT_State* p_state = m_pSamples->CenterSampling(&m_state_cur, m_pImage);

	MVT_State* p_state_max = NULL;
	MVT_State* p_state_ref = &m_state_cur;
	p_state_ref->bbox_root     = p_state->bbox_root;
	p_state_ref->center_root.x = p_state->bbox_root.x + p_state->bbox_root.width/2;
	p_state_ref->center_root.y = p_state->bbox_root.y + p_state->bbox_root.height/2;

	if( g_param.use_alm || g_param.use_mil )
	{
LOG( "Start from" );
MVT_State_Print(p_state_ref, false);
LOG( std::endl; );

		MVT_State* p_state_online;

		for( unsigned int v=0; v<m_num_of_viewpoint; v++ )
		{
			// Sampling per a candidate viewpoint
			m_pSamples->ViewpointSampling(v,p_state_ref,m_pImage);

			// Classifier Update
			Update_Models(mvt_image, m_pSamples->Get2DObject(v));

			// Compute Potentials
			p_state = m_pSamples->ComputePotentials(v,&m_state_cur);

			// Prepare for the next sampling

			double prob = v==0?1:(p_state->potential-p_state_ref->potential);

			// TODO exp(-score)
			if( prob > 0/*m_rng.uniform((double)-0.25,(double)0)*/ )
			{
				p_state_ref = p_state;
LOG( std::endl << v << "\t(O)" );
			}
			else
			{
LOG( std::endl << v << "\t(X)" );
			}

MVT_State_Print(p_state, true);
LOG( "max: [" << p_state_ref->idx_viewpoint << ":" << p_state_ref->potential << "] " << std::endl;);
		}

		m_pSamples->GetMaxPotential(p_state_max, p_state_online);
		p_state_max = p_state_online;

		LOG( "MAX=>" );
		MVT_State_Print(p_state_max, true);
		LOG( std::endl );

	}
	else
	{
		p_state_max = p_state;
		MVT_State_Print(p_state_max, true);
	}

	MVT_State_Copy(p_state_max,&m_state_cur);

	Train(idx_frame,m_state_cur);

	return &m_state_cur;
}

void MVT_Tracker::Train(int idx_frame, MVT_State& state_cur)
{
	mvt::OnlineMILModel* pOnlineModel_MIL_root = (mvt::OnlineMILModel*)m_pModels_root[MVT_LIKELIHOOD_MIL_ROOT];
	if( pOnlineModel_MIL_root )
	{
		if( idx_frame == -1 || state_cur.likelihood_root[MVT_LIKELIHOOD_DPM] >= 0 )
		{
			if( !pOnlineModel_MIL_root->IsInitialize() )
			{
				pOnlineModel_MIL_root->initialize(*m_pImage, param_mil, state_cur.bbox_root);
			}
			else
			{
				pOnlineModel_MIL_root->train(idx_frame, m_pImage, state_cur.bbox_root);
			}
		}
		else
		{
LOG("Skip training of MIL root" << std::endl);
		}
	}

	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		if( (idx_frame != -1) &&
			(
				(m_pModels_root[MVT_LIKELIHOOD_DPM] && m_state_cur.likelihood_partsNroots[MVT_LIKELIHOOD_ALM] < -1000) ||
				(m_pModels_partsNroots[pr][MVT_LIKELIHOOD_ALM] && m_state_cur.likelihood_root[MVT_LIKELIHOOD_DPM] < 0 )
			)
		)
		{
			continue;
		}

		if( !state_cur.pObject2d->IsOccluded(pr) )
		{
			mvt::OnlineMILModel* pOnlineModel_MIL = (mvt::OnlineMILModel*)m_pModels_partsNroots[pr][MVT_LIKELIHOOD_MIL];
			if( pOnlineModel_MIL!=NULL )
			{
				cv::Point2d center = state_cur.centers[pr];
				cv::Rect rect = state_cur.pObject2d->GetRectifiedRect(pr,center);
				cv::Mat* pImage_rectified = state_cur.pObject2d->GetRectifiedImage(pr);

				// TODO when the bounding box is placed on the outside of image
				if( rect.x <= 0 ) rect.x = 1;
				if( rect.y <= 0 ) rect.y = 1;
				if( rect.x+rect.width  >= pImage_rectified->cols ) rect.width  = pImage_rectified->cols-rect.x-1;
				if( rect.y+rect.height >= pImage_rectified->rows ) rect.height = pImage_rectified->rows-rect.y-1;

				if( !pOnlineModel_MIL->IsInitialize() )
				{
					pOnlineModel_MIL->initialize(*pImage_rectified, param_mil, rect);
				}
				else
				{
					pOnlineModel_MIL->train(idx_frame, pImage_rectified, rect);
				}
			}
		}
	}

LOG( std::endl );
}

