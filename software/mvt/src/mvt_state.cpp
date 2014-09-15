#include "mvt.h"

void MVT_State_Print(MVT_State* p_state, bool b_all)
{
	LOG( "(" );
	LOG( "a="  << p_state->viewpoint.azimuth   << "," );
	LOG( "e="  << p_state->viewpoint.elevation << "," );
	LOG( "d="  << p_state->viewpoint.distance         );
	LOG( "):[" << p_state->potential           << "] ");
	LOG( "[" << p_state->overlap << "] ");

	if( b_all )
	{
		LOG( "\tDPM[" << p_state->likelihood_root[MVT_LIKELIHOOD_DPM]      << "/" << p_state->likelihood_root_local[MVT_LIKELIHOOD_DPM] << "/" << p_state->likelihood_root_global[MVT_LIKELIHOOD_DPM] << ":" << p_state->overlap << "] ");
		LOG( "ALM[" << p_state->likelihood_partsNroots[MVT_LIKELIHOOD_ALM] << "/" << p_state->likelihood_partsNroots_global[MVT_LIKELIHOOD_ALM]  << "] ");
#if 1
		LOG( "MIL[" << p_state->likelihood_root[MVT_LIKELIHOOD_MIL_ROOT]   << "/" << p_state->likelihood_root_local[MVT_LIKELIHOOD_MIL_ROOT]    );
		LOG( "/" << p_state->likelihood_root_global[MVT_LIKELIHOOD_MIL_ROOT]  << " , ");
		LOG( "" << p_state->likelihood_partsNroots[MVT_LIKELIHOOD_MIL] << "/" << p_state->likelihood_partsNroots_global[MVT_LIKELIHOOD_MIL]  << "] ");
#else
		LOG( "MIL[" << p_state->likelihood_partsNroots[MVT_LIKELIHOOD_MIL] << "/" << p_state->likelihood_partsNroots_global[MVT_LIKELIHOOD_MIL]  << "] ");
#endif
		LOG( "PRW[" << p_state->motion_pairwise  << "] ");
		LOG( "PRR[" << p_state->motion_prior << "] ");
	}
}

void MVT_State_Copy(MVT_State* src, MVT_State* dest)
{
	dest->idx_viewpoint = src->idx_viewpoint;
	dest->viewpoint = src->viewpoint;

	if(g_param.use_alm || g_param.use_mil)
	{
		dest->pObject2d->SetViewpoint(src->viewpoint,src->pObject2d->GetImage());

		if( dest->centers != NULL && dest->centers_rectified != NULL )
		{
			unsigned int n_partsNroots = src->pObject2d->Num_of_PartsNRoots();
			for( unsigned int pr=0; pr<n_partsNroots; pr++ )
			{
				dest->centers[pr]           = src->centers[pr];
				dest->centers_rectified[pr] = src->centers_rectified[pr];
				for( int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
				{
					dest->likelihood_partsNroots_pr[m][pr]        = src->likelihood_partsNroots_pr[m][pr];
					dest->likelihood_partsNroots_pr_global[m][pr] = src->likelihood_partsNroots_pr_global[m][pr];
				}
			}
		}
	}

	dest->bbox_partsNroots = src->bbox_partsNroots;
	dest->bbox_root = src->bbox_root;

	dest->overlap = src->overlap;

	dest->center_root          = src->center_root;

	dest->potential              = src->potential;
	dest->potential_local        = src->potential_local;
	dest->potential_local_online = src->potential_local_online;

	dest->likelihood_global          = src->likelihood_global;
	dest->likelihood_root_all        = src->likelihood_root_all;
	dest->likelihood_partsNroots_all = src->likelihood_partsNroots_all;
	for( int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		dest->likelihood_root[m]        = src->likelihood_root[m];
		dest->likelihood_root_local[m]  = src->likelihood_root_local[m];
		dest->likelihood_root_global[m] = src->likelihood_root_global[m];
	}
	for( int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		dest->likelihood_partsNroots[m]        = src->likelihood_partsNroots[m];
		dest->likelihood_partsNroots_local[m]  = src->likelihood_partsNroots_local[m];
		dest->likelihood_partsNroots_global[m] = src->likelihood_partsNroots_global[m];
	}

	dest->motion_prior    = src->motion_prior;
	dest->motion_pairwise = src->motion_pairwise;
}

MVT_SampleSet::MVT_SampleSet(MVT_Param param,
		                         MVT_3D_Object* pObject3d,
		                         MVT_Potential_Model_Root**         models_root,
		                         MVT_Potential_Model_partsNroots*** models_partsNroots,
		                         MVT_Potential_Model_Root**         models_motion,
		                         cv::Mat*** pImages_rectified)
{
	m_pObject3d          = pObject3d;
	m_num_of_parts       = m_pObject3d->Num_of_Parts();
	m_num_of_partsNroots = m_pObject3d->Num_of_PartsNRoots();
	m_num_of_roots       = m_num_of_partsNroots - m_num_of_parts;

	m_num_of_viewpoint     = param.num_of_viewpoint_sample;
	m_num_of_center_sample = param.num_of_center_sample;
	m_num_of_samples       = param.num_of_partcenter_sample;

	m_srchwinsz = param.srchwinsz;

	// Viewpoint ( azimuth, elevation, distance )
	m_pViewpoints =  (MVT_Viewpoint*)malloc(sizeof(MVT_Viewpoint) *m_num_of_viewpoint);
	m_pImages_rectified = (cv::Mat**)malloc(sizeof(cv::Mat*)*m_num_of_partsNroots);
	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		m_pImages_rectified[pr] = new cv::Mat;
	}

	m_pSamples = (MVT_State*)malloc(sizeof(MVT_State)*m_num_of_center_sample);
	for( unsigned int s=0; s<m_num_of_center_sample; s++ )
	{
		m_pSamples[s].pObject2d         = NULL;
		m_pSamples[s].centers           = (cv::Point2d*)malloc(sizeof(cv::Point2d)*m_num_of_partsNroots);
		m_pSamples[s].centers_rectified = (cv::Point*  )malloc(sizeof(cv::Point  )*m_num_of_partsNroots);
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			m_pSamples[s].likelihood_partsNroots_pr[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
			m_pSamples[s].likelihood_partsNroots_pr_global[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
		}
	}

	m_pPotentialmap_root = new MVT_Potentialmap((MVT_Potential_Model**)models_root,NUM_OF_LIKELIHOOD_TYPE_ROOT, MVT_POTENTIALMAP_ROOT, -1);

	m_ppObject2ds = (MVT_2D_Object**)malloc(sizeof(MVT_2D_Object*)*m_num_of_viewpoint);
	m_ppPotentialmaps = (MVT_Potentialmap***)malloc(sizeof(MVT_Potentialmap**)*m_num_of_viewpoint);

	for( unsigned int v=0 ; v<m_num_of_viewpoint; v++ )
	{
		m_ppObject2ds[v] = new MVT_2D_Object(m_pObject3d, pImages_rectified[v]);

		m_ppPotentialmaps[v] = (MVT_Potentialmap**)malloc(sizeof(MVT_Potentialmap*)*m_num_of_partsNroots);
		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			m_ppPotentialmaps[v][pr] = new MVT_Potentialmap((MVT_Potential_Model**)models_partsNroots[pr],NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS, MVT_POTENTIALMAP_PARTSNROOTS, pr);
		}
	}

	m_ppSamples_LocalBest        = (MVT_State*)malloc(sizeof(MVT_State)*m_num_of_viewpoint);
	m_ppSamples_LocalBest_online = (MVT_State*)malloc(sizeof(MVT_State)*m_num_of_viewpoint);
	for( unsigned int v=0 ; v<m_num_of_viewpoint; v++ )
	{
		m_ppSamples_LocalBest[v].pObject2d                = new MVT_2D_Object(m_pObject3d, pImages_rectified[v]);
		m_ppSamples_LocalBest[v].centers                  = (cv::Point2d*)malloc(sizeof(cv::Point2d)*m_num_of_partsNroots);
		m_ppSamples_LocalBest[v].centers_rectified        = (cv::Point*  )malloc(sizeof(cv::Point  )*m_num_of_partsNroots);
		m_ppSamples_LocalBest_online[v].pObject2d         = new MVT_2D_Object(m_pObject3d, pImages_rectified[v]);
		m_ppSamples_LocalBest_online[v].centers           = (cv::Point2d*)malloc(sizeof(cv::Point2d)*m_num_of_partsNroots);
		m_ppSamples_LocalBest_online[v].centers_rectified = (cv::Point*  )malloc(sizeof(cv::Point  )*m_num_of_partsNroots);
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			m_ppSamples_LocalBest[v].likelihood_partsNroots_pr[m]               = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
			m_ppSamples_LocalBest[v].likelihood_partsNroots_pr_global[m]        = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
			m_ppSamples_LocalBest_online[v].likelihood_partsNroots_pr[m]        = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
			m_ppSamples_LocalBest_online[v].likelihood_partsNroots_pr_global[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
		}
	}

	// Potential Models which does not consider viewpoint
	m_models_root        = models_root;
	m_models_partsNroots = models_partsNroots;
	m_models_motion      = models_motion;

	m_pSampling = new MVT_Sampling(param, m_pObject3d);

	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		m_min_partsNroots_pr[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
		m_max_partsNroots_pr[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
		m_dist_partsNroots_pr[m] = (double*)malloc(sizeof(double)*m_num_of_partsNroots);
	}

	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		m_min_thresh_root[m] = INFINITY;
	}

	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		m_min_thresh_partsNroots[m] = INFINITY;
	}

	if( g_param.use_mil_root )
	{
		m_min_thresh_root[MVT_LIKELIHOOD_DPM]    = param.thresh_dpm;
	}
	else
	{
		m_min_thresh_root[MVT_LIKELIHOOD_DPM]    = 0;
	}
	m_min_thresh_root[MVT_LIKELIHOOD_MIL_ROOT]   = -100000;
	m_min_thresh_partsNroots[MVT_LIKELIHOOD_ALM] = param.thresh_alm;
	m_min_thresh_partsNroots[MVT_LIKELIHOOD_MIL] = param.thresh_mil;
}

MVT_SampleSet::~MVT_SampleSet()
{
	free(m_pViewpoints);

	for( unsigned int v=0 ; v<m_num_of_viewpoint; v++ )
	{
		delete m_ppObject2ds[v];

		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			delete m_ppPotentialmaps[v][pr];
		}
		free(m_ppPotentialmaps[v]);
	}
	free(m_ppPotentialmaps);
	free(m_ppObject2ds);

	delete m_pPotentialmap_root;

	for( unsigned int v=0 ; v<m_num_of_viewpoint; v++ )
	{
		delete m_ppSamples_LocalBest[v].pObject2d;
		free(m_ppSamples_LocalBest[v].centers);
		free(m_ppSamples_LocalBest[v].centers_rectified);
		free(m_ppSamples_LocalBest[v].likelihood_partsNroots_pr);
		free(m_ppSamples_LocalBest[v].likelihood_partsNroots_pr_global);

		delete m_ppSamples_LocalBest_online[v].pObject2d;
		free(m_ppSamples_LocalBest_online[v].centers);
		free(m_ppSamples_LocalBest_online[v].centers_rectified);
		free(m_ppSamples_LocalBest_online[v].likelihood_partsNroots_pr);
		free(m_ppSamples_LocalBest_online[v].likelihood_partsNroots_pr_global);
	}
	free(m_ppSamples_LocalBest);
	free(m_ppSamples_LocalBest_online);

	for( unsigned int s=0; s<m_num_of_samples; s++ )
	{
		delete m_pSamples[s].pObject2d;
		free(m_pSamples[s].centers);
		free(m_pSamples[s].centers_rectified);
		free(m_pSamples[s].likelihood_partsNroots_pr);
		free(m_pSamples[s].likelihood_partsNroots_pr_global);
	}
	free(m_pSamples);

	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		free(m_min_partsNroots_pr[m]);
		free(m_max_partsNroots_pr[m]);
		free(m_dist_partsNroots_pr[m]);
	}
	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		delete m_pImages_rectified[pr];
	}
	free( m_pImages_rectified );

	delete m_pSampling;
}

MVT_State* MVT_SampleSet::CenterSampling(MVT_State* p_state_cur, cv::Mat* pImage)
{
	m_pImage = pImage;

	m_pSampling->SetRefState(p_state_cur);

	// Center Location Sampling
	m_pSampling->Sampling_Centers(m_pSamples, m_num_of_center_sample, p_state_cur->bbox_root);

	ComputeLikelihoods_root(m_pImage);

	((mvt::Prior*)m_models_motion[MVT_MOTION_PRIOR])->OnOff(false, true);
	((mvt::Prior*)m_models_motion[MVT_MOTION_PRIOR])->SetPrevState(*p_state_cur);
	m_models_motion[MVT_MOTION_PRIOR]->GetPotentials(m_pSamples,m_num_of_center_sample);
	((mvt::Prior*)m_models_motion[MVT_MOTION_PRIOR])->OnOff(g_param.use_alm, true);

	NormalizeLikelihoods_root_local();

	double value_max = -INFINITY;
	unsigned int idx_max = 0;
	for( unsigned int s=0; s<m_num_of_center_sample; s++ )
	{
		double potential = m_pSamples[s].likelihood_root_all*m_pSamples[s].motion_prior;
		if( value_max < potential )
		{
			value_max = potential;
			idx_max = s;
		}
	}
	MVT_State* p_state = &(m_pSamples[idx_max]);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for( unsigned int s=0; s<m_num_of_samples; s++ )
	{
		m_pSamples[s].center_root         = p_state->center_root;
		m_pSamples[s].bbox_root           = p_state->bbox_root;
		m_pSamples[s].likelihood_root_all = p_state->likelihood_root_all;
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
		{
			m_pSamples[s].likelihood_root[m]    = p_state->likelihood_root[m];
			m_pSamples[s].likelihood_root_local[m] = p_state->likelihood_root_local[m];
		}
	}

	return p_state;
}


void MVT_SampleSet::ViewpointSampling(unsigned int v /*idx_viewpoint*/, MVT_State* p_state_ref,  cv::Mat *pImage)
{
	MVT_2D_Object* pObject2d = m_ppObject2ds[v];

	m_pSampling->SetRefState(p_state_ref);

	while(true)
	{
		if( (v==0) &&
			(p_state_ref->viewpoint.distance==p_state_ref->viewpoint.distance) )
		{
			m_pViewpoints[v] = p_state_ref->viewpoint;
		}
		else
		{
			m_pViewpoints[v] = m_pSampling->Sampling_Viewpoint();
		}

		pObject2d->SetViewpoint(m_pViewpoints[v], pImage);

		if( m_pSampling->IsValidViewpoint(pObject2d) == false )
		{
			continue;
		}

		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			if( !pObject2d->IsOccluded(pr) )
			{
				cv::Mat* pImage_rectified = pObject2d->GetRectifiedImage(pr);
				m_ppPotentialmaps[v][pr]->resize(pImage_rectified->cols, pImage_rectified->rows);
			}
		}

		break;
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for( unsigned int s=0; s<m_num_of_samples; s++)
	{
		m_pSamples[s].idx_viewpoint = v;
		m_pSamples[s].viewpoint = m_pViewpoints[v];
		m_pSamples[s].pObject2d = pObject2d;
		//m_ppSamples[v][s].b_bbox_uptodate = false;
	}

	m_pSampling->Sampling_PartsCenters(pObject2d, m_pSamples, m_num_of_samples);
}

MVT_State* MVT_SampleSet::ComputePotentials(unsigned int v, MVT_State* p_state_cur)
{
	if( v==0 ) Normalize_Init();

	ComputeOverlap();
	//ComputeLikelihoods_root(m_pImage);
	ComputeLikelihoods_partsNroots(v);

	//NormalizeLikelihoods_root_local();
	NormalizeLikelihoods_partsNroots_local();

	ComputeMotions(m_pSamples,m_num_of_samples,p_state_cur);

	MVT_State* p_state_max = NULL;
	MVT_State* p_state_online_max = NULL;
	GetMaxPotential_local(p_state_max,p_state_online_max);

	MVT_State_Copy( p_state_max,        &(m_ppSamples_LocalBest[v])        );
	MVT_State_Copy( p_state_online_max, &(m_ppSamples_LocalBest_online[v]) );

	NormalizeLikelihoods_Global(m_ppSamples_LocalBest, (v+1));

	for( unsigned int i=0; i<=v; i++ )
	{
		m_ppSamples_LocalBest[i].potential =  m_ppSamples_LocalBest[i].likelihood_global
											 *m_ppSamples_LocalBest[i].motion_pairwise
											 *m_ppSamples_LocalBest[i].motion_prior;
	}

	return &m_ppSamples_LocalBest[v];
}

void MVT_SampleSet::ComputeOverlap()
{
	for( unsigned int s=0; s<m_num_of_center_sample; s++ )
	{
#if 0
		cv::Rect rect = m_pSamples[s].bbox_partsNroots & m_pSamples[s].bbox_root;

		m_pSamples[s].overlap = ((double)(rect.width*rect.height)) / ((double)( (m_pSamples[s].bbox_partsNroots.width*m_pSamples[s].bbox_partsNroots.height)
				                                                               +(m_pSamples[s].bbox_root.width*m_pSamples[s].bbox_root.height)
				                                                               -(rect.width*rect.height) ));
#else
		double width_root         = m_pSamples[s].bbox_root.width;
		double height_root        = m_pSamples[s].bbox_root.height;
		double width_partsNroots  = m_pSamples[s].bbox_partsNroots.width;
		double height_partsNroots = m_pSamples[s].bbox_partsNroots.height;

		double height_max  = height_root>height_partsNroots?height_root:height_partsNroots;
		double dist_height = abs(height_root-height_partsNroots);

		double width_max  = width_root>width_partsNroots?width_root:width_partsNroots;
		double dist_width = abs(width_root-width_partsNroots);

		double aspr1_root = (width_root/height_root);
		double aspr2_root = (height_root/width_root);
		double aspr_root = aspr1_root<aspr2_root?aspr1_root:(1+1/aspr2_root);

		double aspr1_partsNroots = (width_partsNroots/height_partsNroots);
		double aspr2_partsNroots = (height_partsNroots/width_partsNroots);
		double aspr_partsNroots = aspr1_partsNroots<aspr2_partsNroots?aspr1_partsNroots:(1+1/aspr2_partsNroots);

		m_pSamples[s].overlap = ((height_max-dist_height)/height_max) * ((width_max-dist_width)/width_max) * (1-abs(aspr_root-aspr_partsNroots));

#endif

	}
}

void MVT_SampleSet::ComputeLikelihoods_root(cv::Mat *pImage)
{
	// Compute Likelihood for Root part
	m_pPotentialmap_root->resize(pImage->cols, pImage->rows);
	for( unsigned int s=0; s<m_num_of_center_sample; s++ )
	{
		m_pPotentialmap_root->RequestComputePotential(m_pSamples[s]);
	}
	m_pPotentialmap_root->ComputePotentials();

	for( unsigned int s=0; s<m_num_of_center_sample; s++ )
	{
		m_pPotentialmap_root->GetPotential(&m_pSamples[s]);
	}
}

void MVT_SampleSet::ComputeLikelihoods_partsNroots(unsigned int v/*idx_viewpoint*/)
{
	// Preprocessor for Likelihood for partsNroots
	MVT_2D_Object* pObject2d = m_ppObject2ds[v];
	double values_occluded[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS][m_num_of_partsNroots];
	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		if( !pObject2d->IsOccluded(pr) )
		{
			for( unsigned int s=0; s<m_num_of_samples; s++ )
			{
				potential(v,pr)->RequestComputePotential(m_pSamples[s]);
			}
			potential(v,pr)->ComputePotentials();
		}
		else
		{
			for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
			{
				values_occluded[m][pr] = potential(v,pr)->GetPotentialOccluded(m);
			}
		}
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for( unsigned int s=0; s<m_num_of_samples; s++ )
	{
		MVT_State* p_sample = &(m_pSamples[s]);

		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			double value = 0;
			unsigned int num_of_visible = 0;
			for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
			{
				double tmp = 0;
				if( !pObject2d->IsOccluded(pr) )
				{
					tmp = potential(v,pr)->GetPotential(p_sample,m);
					num_of_visible++;
				}
				else
				{
					tmp = values_occluded[m][pr];
				}
				value += tmp;
				p_sample->likelihood_partsNroots_pr[m][pr] = tmp;
			}

			if( m==MVT_LIKELIHOOD_MIL )
			{
				value = value / num_of_visible;
			}

			p_sample->likelihood_partsNroots[m] = value;
		}
	}
}

void MVT_SampleSet::Normalize_Init()
{
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		m_min_root[m] = INFINITY;
		m_max_root[m] = -INFINITY;
	}

	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		m_min_partsNroots[m] = INFINITY;
		m_max_partsNroots[m] = -INFINITY;
	}
}

void MVT_SampleSet::NormalizeLikelihoods_root_local()
{
	// Normalization
	double values_root_min[NUM_OF_LIKELIHOOD_TYPE_ROOT];
	double values_root_max[NUM_OF_LIKELIHOOD_TYPE_ROOT];

	int s=0;
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		values_root_min[m] = m_pSamples[s].likelihood_root[m];
		values_root_max[m] = m_pSamples[s].likelihood_root[m];
	}

	for( unsigned s=1; s<m_num_of_samples; s++ )
	{
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
		{
			if( m_pSamples[s].likelihood_root[m] < m_min_thresh_root[m]  )
			{
				values_root_min[m] = m_min_thresh_root[m];
			}
			else
			{
				values_root_min[m] = values_root_min[m]<m_pSamples[s].likelihood_root[m]?values_root_min[m]:m_pSamples[s].likelihood_root[m];
			}
			values_root_max[m] = values_root_max[m]>m_pSamples[s].likelihood_root[m]?values_root_max[m]:m_pSamples[s].likelihood_root[m];
		}
	}

	float values_root_dist[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		values_root_dist[m] = values_root_max[m] - values_root_min[m];
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for( unsigned int s=0; s<m_num_of_samples; s++ )
	{
		double value = 1;
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
		{
			if( m==MVT_LIKELIHOOD_DPM )
			{
				if( m_pSamples[s].likelihood_root[m] < m_min_thresh_root[m] )
				{
					m_pSamples[s].likelihood_root_local[m] = m_min_thresh_root[m];
				}
				else
				{
					m_pSamples[s].likelihood_root_local[m] = m_pSamples[s].likelihood_root[m]-values_root_min[m] + 1;
				}
			}
			else
			{
				if( values_root_dist[m] > 0 )
				{
					if( m_pSamples[s].likelihood_root[m] < m_min_thresh_root[m] )
					{
						m_pSamples[s].likelihood_root_local[m] = 0.5;
					}
					else
					{
						m_pSamples[s].likelihood_root_local[m]
						 = ((m_pSamples[s].likelihood_root[m]-values_root_min[m]) / values_root_dist[m]) + 1;
					}
				}
				else
				{
					m_pSamples[s].likelihood_root_local[m] = 0.5;
				}
			}
			value *= m_pSamples[s].likelihood_root_local[m];
		}
		m_pSamples[s].likelihood_root_all = value;
	}

	// Store Global min/max value
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		m_min_root[m] = m_min_root[m]<values_root_min[m]?m_min_root[m]:values_root_min[m];
		m_max_root[m] = m_max_root[m]>values_root_max[m]?m_max_root[m]:values_root_max[m];
	}
}

void MVT_SampleSet::NormalizeLikelihoods_partsNroots_local()
{
	// Normalization
	double values_partsNroots_min[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	double values_partsNroots_max[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	double values_partsNroots_pr_min[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS][m_num_of_partsNroots];
	double values_partsNroots_pr_max[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS][m_num_of_partsNroots];

	int s=0;
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		values_partsNroots_min[m] = m_pSamples[s].likelihood_partsNroots[m];
		values_partsNroots_max[m] = m_pSamples[s].likelihood_partsNroots[m];

		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			values_partsNroots_pr_min[m][pr] = m_pSamples[s].likelihood_partsNroots_pr[m][pr];
			values_partsNroots_pr_max[m][pr] = m_pSamples[s].likelihood_partsNroots_pr[m][pr];
		}
	}

	for( unsigned s=1; s<m_num_of_samples; s++ )
	{
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			if( m_pSamples[s].likelihood_partsNroots[m] < m_min_thresh_partsNroots[m]  )
			{
				values_partsNroots_min[m] = m_min_thresh_partsNroots[m];
			}
			else
			{
				values_partsNroots_min[m] = values_partsNroots_min[m]<m_pSamples[s].likelihood_partsNroots[m]?values_partsNroots_min[m]:m_pSamples[s].likelihood_partsNroots[m];
			}
			values_partsNroots_max[m] = values_partsNroots_max[m]>m_pSamples[s].likelihood_partsNroots[m]?values_partsNroots_max[m]:m_pSamples[s].likelihood_partsNroots[m];

			for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
			{
				if( m_pSamples[s].likelihood_partsNroots_pr[m][pr] < m_min_thresh_partsNroots[m] )
				{
					values_partsNroots_pr_min[m][pr] = m_min_thresh_partsNroots[m];
				}
				else
				{
					values_partsNroots_pr_min[m][pr] = values_partsNroots_pr_min[m][pr]<m_pSamples[s].likelihood_partsNroots_pr[m][pr]?values_partsNroots_pr_min[m][pr]:m_pSamples[s].likelihood_partsNroots_pr[m][pr];
				}
				values_partsNroots_pr_max[m][pr] = values_partsNroots_pr_max[m][pr]>m_pSamples[s].likelihood_partsNroots_pr[m][pr]?values_partsNroots_pr_max[m][pr]:m_pSamples[s].likelihood_partsNroots_pr[m][pr];
			}
		}
	}

	float values_partsNroots_dist_raw[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	float values_partsNroots_dist[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS];
	float values_partsNroots_pr_dist[NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS][m_num_of_partsNroots];
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		values_partsNroots_dist[m]     = values_partsNroots_max[m] - values_partsNroots_min[m];
		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			values_partsNroots_pr_dist[m][pr] = values_partsNroots_pr_max[m][pr] - values_partsNroots_pr_min[m][pr];
		}
	}

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for( unsigned int s=0; s<m_num_of_samples; s++ )
	{
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			if( values_partsNroots_dist[m] > 0 )
			{
				if( m_pSamples[s].likelihood_partsNroots[m] < m_min_thresh_partsNroots[m] )
				{
					m_pSamples[s].likelihood_partsNroots_local[m] = 0.5;
				}
				else
				{
					m_pSamples[s].likelihood_partsNroots_local[m]
					 = ((m_pSamples[s].likelihood_partsNroots[m]-values_partsNroots_min[m]) / values_partsNroots_dist[m]) + 1;
				}
			}
			else
			{
				m_pSamples[s].likelihood_partsNroots_local[m] = 0.5;
			}
		}
	}

	// Store Global min/max value
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		m_min_partsNroots[m]     = m_min_partsNroots[m]    <values_partsNroots_min[m]    ?m_min_partsNroots[m]    :values_partsNroots_min[m];
		m_max_partsNroots[m]     = m_max_partsNroots[m]>values_partsNroots_max[m]?m_max_partsNroots[m]:values_partsNroots_max[m];

		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			m_min_partsNroots_pr[m][pr] = m_min_partsNroots_pr[m][pr]<values_partsNroots_pr_min[m][pr]?m_min_partsNroots_pr[m][pr]:values_partsNroots_pr_min[m][pr];
			m_max_partsNroots_pr[m][pr] = m_max_partsNroots_pr[m][pr]>values_partsNroots_pr_max[m][pr]?m_max_partsNroots_pr[m][pr]:values_partsNroots_pr_max[m][pr];
		}
	}
}

void MVT_SampleSet::ComputeMotions(MVT_State* samples, unsigned int n_samples , MVT_State* p_state_cur)
{
	for( unsigned int m=0; m<NUM_OF_MOTION; m++ )
	{
		if( m_models_motion[m] )
		{
			if( m==MVT_MOTION_PRIOR )
			{
				((mvt::Prior*)m_models_motion[MVT_MOTION_PRIOR])->SetPrevState(*p_state_cur);
			}
			m_models_motion[m]->GetPotentials(samples,n_samples);
		}
		else
		{
			for( unsigned int s=0; s<n_samples; s++ )
			{
				if( m==MVT_MOTION_PAIRWISE )
				{
					samples[s].motion_pairwise = 1;
				}
				else if( m==MVT_MOTION_PRIOR )
				{
					samples[s].motion_prior = 1;
				}
				else
				{
					assert(0);
				}
			}
		}
	}
}

void MVT_SampleSet::NormalizeLikelihoods_Global(MVT_State* states, unsigned int n_states)
{
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
	{
		m_dist_root[m] = m_max_root[m] - m_min_root[m];
	}
	for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
	{
		m_dist_partsNroots[m]     = m_max_partsNroots[m] - m_min_partsNroots[m];
		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			m_dist_partsNroots_pr[m][pr] = m_max_partsNroots_pr[m][pr] - m_min_partsNroots_pr[m][pr];
		}
	}

	for( unsigned int s=0; s<n_states; s++ )
	{
		double value = 1;
		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
		{
			if( m_dist_root[m] > 0 )
			{
				if( states[s].likelihood_root[m]<m_min_thresh_root[m] )
				{
					states[s].likelihood_root_global[m] = 0.5;
				}
				else
				{
					states[s].likelihood_root_global[m] = ((states[s].likelihood_root[m]-m_min_root[m]) / m_dist_root[m]) + 1;
				}
			}
			else
			{
				states[s].likelihood_root_global[m] = 0.5;
			}
			value *= states[s].likelihood_root_global[m];
		}

		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			if( m==MVT_LIKELIHOOD_ALM )
			{
				if( m_dist_partsNroots[m] > 0 )
				{
					if( states[s].likelihood_partsNroots[m]<m_min_thresh_partsNroots[m] )
					{
						states[s].likelihood_partsNroots_global[m] = 0.5;
					}
					else
					{
						states[s].likelihood_partsNroots_global[m] = ((states[s].likelihood_partsNroots[m]-m_min_partsNroots[m]) / m_dist_partsNroots[m]) + 1;
					}
				}
				else
				{
					states[s].likelihood_partsNroots_global[m] = 0.5;
				}
				value *= states[s].likelihood_partsNroots_global[m];
			}

			for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
			{
				if( m_dist_partsNroots_pr[m][pr] > 0 )
				{
					if( states[s].likelihood_partsNroots_pr[m][pr] < m_min_thresh_partsNroots[m] )
					{
						states[s].likelihood_partsNroots_pr_global[m][pr] = 0.5;
					}
					else
					{
						states[s].likelihood_partsNroots_pr_global[m][pr]
						 = ((states[s].likelihood_partsNroots_pr[m][pr]-m_min_partsNroots_pr[m][pr]) / m_dist_partsNroots_pr[m][pr]) + 1;
					}
				}
				else
				{
					states[s].likelihood_partsNroots_pr_global[m][pr] = 0.5;
				}
			}
		}
		states[s].likelihood_global = value;
	}
}

void MVT_SampleSet::GetMaxPotential(MVT_State* &p_state_max, MVT_State* &p_state_online)
{
	NormalizeLikelihoods_Global( m_ppSamples_LocalBest, m_num_of_viewpoint );

	p_state_max    = &(m_ppSamples_LocalBest[0]);
	p_state_online = &(m_ppSamples_LocalBest_online[0]);
	for( unsigned int v=1; v<m_num_of_viewpoint; v++ )
	{
		double potential = 1;
		potential *= (m_ppSamples_LocalBest[v].overlap)
					*(m_ppSamples_LocalBest[v].motion_pairwise)
					*(m_ppSamples_LocalBest[v].motion_prior);

		double potential_max = 1;
		potential_max *= p_state_max->overlap
						*p_state_max->motion_pairwise
						*p_state_max->motion_prior;

		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
		{
			potential     *= m_ppSamples_LocalBest[v].likelihood_root_global[m];
			potential_max *= p_state_max->likelihood_root_global[m];
		}

		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			if( m==MVT_LIKELIHOOD_ALM )
			{
				potential     *= m_ppSamples_LocalBest[v].likelihood_partsNroots_global[m];
				potential_max *= p_state_max->likelihood_partsNroots_global[m];
			}
		}

		if( potential_max < potential )
		{
			potential_max = potential;
			p_state_max   = &(m_ppSamples_LocalBest[v]);
			p_state_online= &(m_ppSamples_LocalBest_online[v]);
		}
	}
}

void MVT_SampleSet::GetMaxPotential_local(MVT_State* &p_state_max, MVT_State* &p_state_max_with_online)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
	for( unsigned int s=0; s<m_num_of_samples; s++ )
	{
		double value = m_pSamples[s].overlap
				       *m_pSamples[s].motion_pairwise
				       *m_pSamples[s].motion_prior;
		double value_online = value;

		for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_PARTSNROOTS; m++ )
		{
			if( m==MVT_LIKELIHOOD_ALM )
			{
				value        *= m_pSamples[s].likelihood_partsNroots_local[m];
				value_online *= m_pSamples[s].likelihood_partsNroots_local[m];
			}
			else if( m==MVT_LIKELIHOOD_MIL )
			{
				value_online *= m_pSamples[s].likelihood_partsNroots_local[m];
			}
		}
		m_pSamples[s].potential_local        = value;
		m_pSamples[s].potential_local_online = value_online;
	}

	unsigned int s=0;
	double potential_normalized_max        = m_pSamples[s].potential_local;
	double potential_normalized_online_max = m_pSamples[s].potential_local_online;
	unsigned int potential_normalized_idx=s;
	unsigned int potential_normalized_onilne_idx=s;
	for( s=1; s<m_num_of_samples; s++ )
	{
		if( potential_normalized_max < m_pSamples[s].potential_local )
		{
			potential_normalized_max = m_pSamples[s].potential_local;
			potential_normalized_idx=s;
		}

		if( potential_normalized_online_max < m_pSamples[s].potential_local_online )
		{
			potential_normalized_online_max = m_pSamples[s].potential_local_online;
			potential_normalized_onilne_idx=s;
		}
	}

	p_state_max             = &(m_pSamples[potential_normalized_idx]);
	p_state_max_with_online = &(m_pSamples[potential_normalized_onilne_idx]);
}

void MVT_SampleSet::Draw(cv::Mat draw, MVT_State* p_state)
{
	MVT_2D_Object* pObject2d = p_state->pObject2d;
	unsigned int num_of_partsNroots = pObject2d->Num_of_PartsNRoots();
	unsigned int num_of_parts       = pObject2d->Num_of_Part();

	for( unsigned int pr=0; pr<num_of_partsNroots; pr++ )
	{
		if( !pObject2d->IsOccluded(pr) )
		{
			cv::Scalar color;
			if(pr==0) color = CV_RGB(255,0,0);
			else if(pr==1) color = CV_RGB(0,255,0);
			else if(pr==2) color = CV_RGB(0,0,255);
			else if(pr==3) color = CV_RGB(0,255,255);
			else if(pr==4) color = CV_RGB(255,255,255);
			else if(pr==5) color = CV_RGB(255,0,255);
			else color = CV_RGB(255,255,0);

			pObject2d->Draw(draw, pr, p_state->centers[pr], color );

			if( pr <= num_of_parts )
			{
				cv::line(draw,p_state->centers[pr],pObject2d->m_2dparts[pr].center+p_state->center_root,color);
				cv::circle(draw,p_state->centers[pr],2,cv::Scalar(255,255,0));
			}
		}
	}
}

