#ifndef __MOTION_PAIRWISE_H__
#define __MOTION_PAIRWISE_H__

namespace mvt
{

typedef class Pairwise : public MVT_Potential_Model_Root
{
public:
	Pairwise()
	{
		m_nd=NULL;
	}
	~Pairwise()
	{
		if( m_nd ) delete m_nd;
	}

	void Initialize()
	{
		if(!m_nd)
		{
			m_nd = new boost::math::normal(0.0, 1.0);
		}
		m_pdf_mean = boost::math::pdf(*m_nd,0.0);
	}

	bool IsInitialize(){return (m_nd!=NULL);}

	float GetPotential(MVT_State* p_states);

private:

	double GetPairwisePenalty(MVT_State* p_state);

	boost::math::normal* m_nd;
	double m_pdf_mean;

} Pairwise;

}

#endif
