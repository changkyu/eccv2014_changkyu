#include "mvt.h"

MVT_Potentialmap::MVT_Potentialmap(MVT_Potential_Model** models, unsigned int n_models, ENUM_POTENTIALMAP_TYPE type, int idx_part)
{
	m_width = 0;
	m_height = 0;

	m_num_of_models = n_models;
	m_models =  (MVT_Potential_Model**) malloc(sizeof(MVT_Potential_Model*) * m_num_of_models);
	m_request_queue =     (std::vector<MVT_State*>*) new std::vector<MVT_State*>[m_num_of_models];
	m_potentialmap  = (std::vector<MVT_Potential>*) new std::vector<MVT_Potential>[m_num_of_models];
	for(unsigned int m=0; m<m_num_of_models; m++)
	{
		m_models[m] = models[m];
		m_request_queue[m].clear();
		m_potentialmap[m].clear();
	}

	m_type = type;
	m_idx_part = idx_part;
}

MVT_Potentialmap::~MVT_Potentialmap()
{
	free(m_models);
	delete[] m_request_queue;
	delete[] m_potentialmap;
}

void MVT_Potentialmap::clear(unsigned int m)
{
	m_request_queue[m].clear();
	int n_potential = m_potentialmap[m].size();
	for( int p=0; p<n_potential; p++ )
	{
		m_potentialmap[m][p].is_requested = false;
		m_potentialmap[m][p].is_updated   = false;
		m_potentialmap[m][p].value        = 0;
	}
}

MVT_Potential* MVT_Potentialmap::potentialmap(MVT_State *p_state, unsigned int m)
{
	int x,y;
	if( m_type==MVT_POTENTIALMAP_ROOT )
	{
		x=p_state->center_root.x;
		y=p_state->center_root.y;
	}
	else
	{
		x=p_state->centers_rectified[m_idx_part].x;
		y=p_state->centers_rectified[m_idx_part].y;
	}
	return &(m_potentialmap[m][ x*m_height + y ]);
}

void MVT_Potentialmap::RequestComputePotential(MVT_State &state, unsigned int m)
{
	MVT_Potential* p_potential = potentialmap(&state,m);
	if( !p_potential->is_updated )
	{
		if( !p_potential->is_requested )
		{
			m_request_queue[m].push_back(&state);
			p_potential->is_requested = true;
		}
	}
}

void MVT_Potentialmap::ComputePotentials(unsigned int m)
{
	if( m_request_queue[m].size() > 0 )
	{
		unsigned int n_requests = m_request_queue[m].size();
		float potentials[n_requests];
		memset(potentials,0,sizeof(float)*n_requests);
		if( (m_models[m]!=NULL) && (m_models[m]->IsInitialize()) )
		{
			m_models[m]->GetPotentials(m_request_queue[m], n_requests, potentials);
		}
		else
		{
			memset(potentials,0,sizeof(float)*n_requests);
		}

#ifdef _OPENMP
#pragma omp parallel for
#endif
		for( unsigned int r=0; r<n_requests; r++ )
		{
			MVT_Potential* p_potential = potentialmap(m_request_queue[m][r],m);
			p_potential->value         = (double)potentials[r];
			p_potential->is_updated    = true;
			p_potential->is_requested  = false;
		}

		m_request_queue[m].clear();
	}
}

