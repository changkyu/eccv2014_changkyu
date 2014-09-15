#include "mvt.h"
using namespace boost::math;

namespace mvt
{

double Pairwise::GetPairwisePenalty(MVT_State* p_state)
{
	double value=1;

	MVT_2D_Object* pObject2d = p_state->pObject2d;
	unsigned int num_of_parts = pObject2d->Num_of_Part();
	for( unsigned int p=0; p<num_of_parts; p++ )
	if( !pObject2d->IsOccluded(p) )
	{
		MVT_2D_Part root = pObject2d->GetPartInfo( pObject2d->GetRootIndex() );
		/*
		double dist_x_unit = (root.vertices[2].x - root.vertices[1].x)/32;
		double dist_y_unit = (root.vertices[1].y - root.vertices[0].y)/16;
		 */
		double dist_x_unit = (p_state->bbox_partsNroots.height )/4;
		double dist_y_unit = (p_state->bbox_partsNroots.height )/4;

		double dist_x = ((p_state->centers[p].x - p_state->center_root.x) - (pObject2d->m_2dparts[p].center.x)) ;
		double dist_y = ((p_state->centers[p].y - p_state->center_root.y) - (pObject2d->m_2dparts[p].center.y)) ;

		value *= pdf( *m_nd, dist_x/dist_x_unit ) / m_pdf_mean;
		value *= pdf( *m_nd, dist_y/dist_y_unit ) / m_pdf_mean;
	}

	return value;
}

float Pairwise::GetPotential(MVT_State* p_state)
{
	p_state->motion_pairwise = GetPairwisePenalty(p_state);
	return p_state->motion_pairwise;
}

}
