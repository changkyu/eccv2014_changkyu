#include "mvt.h"

namespace mvt
{

DetectorALM::DetectorALM(ENUM_OBJECT_CATEGORY object_category, const char* filepath_alm_model, unsigned int idx_part)
{
	m_object_category = object_category;

	MATFile* mat = matOpen(filepath_alm_model,"r");
	if( mat != NULL )
	{
		mxArray* pmxCad = matGetVariable(mat, "cad");
		pmxCad = mxGetCell(pmxCad,0);

		m_idx_part = idx_part;
		/*
		 * Names of part
		 */
		mxArray* pmxNames    = mxGetField(pmxCad, 0, "pnames");
		mxArray* pmxName = mxGetCell(pmxNames, idx_part);
		m_name = MxArray(pmxName).toString();

		/*
		 * Weight of part
		 */
		mxArray* pmxModel    = matGetVariable(mat, "model");
		mxArray* pmxModelCad = mxGetField(pmxModel, 0, "cad");
		pmxModelCad = mxGetCell(pmxModelCad,0);

		mxArray* pmxWeight = mxGetField(pmxModelCad, 0, m_name.c_str());
		std::vector<double> weight = MxArray(pmxWeight).toVector<double>();
		m_weight_length = weight.size();

		m_weight.dims_num = 1;
		m_weight.dims = (int*)malloc(sizeof(int));
		m_weight.dims[0] = m_weight_length;
		m_weight.length = m_weight_length;
		m_weight.data = (float*)malloc(sizeof(float)*m_weight_length);
		for( unsigned int d=0; d<m_weight_length-1; d++ )
		{
			m_weight.data[d] = weight[d];
		}
		m_weight_occluded = weight[m_weight_length-1];

		/*
		 * Parts2d_Front
		 */
		mxArray* pmxPars2d_front = mxGetField(pmxCad, 0, "parts2d_front");
		m_width    = MxArray(mxGetField(pmxPars2d_front, idx_part, "width"   )).toDouble();
		m_height   = MxArray(mxGetField(pmxPars2d_front, idx_part, "height"  )).toDouble();
	}

	m_hog.data = NULL;
	m_hog.dims = NULL;
}

DetectorALM::~DetectorALM()
{
	free_cumatrix(&m_hog);
	free_cumatrix(&m_weight);
}

float DetectorALM::compute_similarity(cv::Point &center)
{
	// Crop hog
	int b0     = m_height/HOGBINSIZE;
	int b1     = m_width/HOGBINSIZE;
	int b0b1   = b0*b1;

	int w = m_hog.dims[1];
	int h = m_hog.dims[0];
	int wh = w*h;
	int z = m_hog.dims[2];

	int offset_x = - b1/2 + center.x/HOGBINSIZE;
	int offset_y = - b0/2 + center.y/HOGBINSIZE;

	float ret = 0;

	float* data_hog    = m_hog.data;
	float* data_weight = m_weight.data;
	for(int x = 0; x < b1; x++)
	{
		int xx = x + offset_x;
		if( xx >= 0 && xx < w )
		{
			int xb0 = x*b0;
			int xxh = xx*h;
			for(int y = 0; y < b0; y++)
			{
				int yy = y + offset_y;
				if(yy >= 0 && yy < h)
				{
					for(int i = 0; i < z; i++)
					{
						int idx_crop = i*b0b1 + xb0 + y;
						int idx_hog  = i*wh   + xxh + yy;

						// compute dot product
						ret = ret + data_weight[idx_crop] * data_hog[idx_hog];
					}
				}
			}
		}
	}

	return ret;
}

double
DetectorALM::GetOccludedPotential()
{
	return m_weight_occluded;
}

void DetectorALM::GetPotentials(std::vector<MVT_State*> &states, unsigned int n_states, float* p_potentials)
{
	if( m_is_occluded )
	{
		if( p_potentials )
		{
			std::fill(p_potentials, p_potentials + n_states -1, m_weight_occluded);
		}
		for( unsigned int s=0; s<n_states; s++ )
		{
			states[s]->likelihood_partsNroots[MVT_LIKELIHOOD_ALM] = 0;
		}
	}
	else
	{
#if 0 // TODO
		CUMATRIX hog_response;
		hog_response = fconv(m_hog,m_weight);

		free_cumatrix(&hog_response);
#else
#ifdef _OPENMP
#pragma omp parallel for
#endif
		for( unsigned int s=0; s<n_states; s++ )
		{
			states[s]->likelihood_partsNroots[MVT_LIKELIHOOD_ALM] =
					compute_similarity(states[s]->centers_rectified[m_idx_part]);
			if( p_potentials )
			{
				p_potentials[s] = states[s]->likelihood_partsNroots[MVT_LIKELIHOOD_ALM];
			}
		}
#endif
	}
}

}
