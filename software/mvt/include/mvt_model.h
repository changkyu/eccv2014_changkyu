typedef class MVT_Potential_Model
{
public:
	MVT_Potential_Model(){};

	virtual
	~MVT_Potential_Model(){};

	virtual void
	SetImage(const MVT_Image mvt_image)
	{
		SetImage(mvt_image.pImage);
		SetImage(mvt_image.cumx_image);

		SetConf(mvt_image.filepath_conf);
	}

	virtual void
	SetConf(char* filepath_conf){};

	virtual void
	SetImage(cv::Mat *pImage){};

	virtual void
	SetImage(CUMATRIX image){};

	virtual bool
	IsInitialize()=0;

	virtual float
	GetPotential(MVT_State* p_states)=0;

	virtual void
	GetPotentials(MVT_State* &states, unsigned int n_states, float* p_potentials=NULL)
	{
		for( unsigned int s=0; s<n_states; s++ )
		{
			float value = GetPotential(&(states[s]));
			if( p_potentials )
			{
				p_potentials[s] = value;
			}
		}
	}

	virtual void
	GetPotentials(std::vector<MVT_State*> &states, unsigned int n_states, float* p_potentials=NULL)
	{
		for( unsigned int s=0; s<n_states; s++ )
		{
			float value = GetPotential(states[s]);
			if( p_potentials )
			{
				p_potentials[s] = value;
			}
		}
	}

} MVT_Potential_Model;

typedef class MVT_Potential_Model_Root : public MVT_Potential_Model
{
public:

	MVT_Potential_Model_Root()
	{

	};

	virtual
	~MVT_Potential_Model_Root(){};

private:

} MVT_Potential_Model_Root;

typedef class MVT_Potential_Model_partsNroots : public MVT_Potential_Model
{
public:

	MVT_Potential_Model_partsNroots(){init(0,0,-1);}
	MVT_Potential_Model_partsNroots(unsigned int width, unsigned height, unsigned int idx_part){init(width,height,idx_part);}

	virtual
	~MVT_Potential_Model_partsNroots(){};

	void SetWeight(double weight)
	{
		m_weight = weight;
	}

	virtual double
	GetOccludedPotential()=0;

	void SetOccluded(bool is_occluded)
	{
		m_is_occluded = is_occluded;
	}

	bool IsOccluded()
	{
		return m_is_occluded;
	}

private:
	void init(unsigned int width, unsigned int height, unsigned int idx_part)
	{
		m_idx_part = idx_part;
		m_width  = width;
		m_height = height;
		m_weight = 1;
		m_is_occluded = false;
	}

protected:

	unsigned int m_idx_part;

	unsigned int m_width;
	unsigned int m_height;

	double m_weight;
	bool   m_is_occluded;

} MVT_Potential_Model_partsNroots;

typedef enum ENUM_POTENTIALMAP_TYPE
{
	MVT_POTENTIALMAP_ROOT=0,
	MVT_POTENTIALMAP_PARTSNROOTS
}ENUM_POTENTIALMAP_TYPE;

typedef class MVT_Potentialmap
{

public:

	MVT_Potentialmap(MVT_Potential_Model** models, unsigned int n_models, ENUM_POTENTIALMAP_TYPE type, int idx_part);
	~MVT_Potentialmap();

	void resize(unsigned int width, unsigned int height)
	{
		if( m_width*m_height == width*height )
		{
			clear();
		}
		else
		{
			m_width  = width;
			m_height = height;

			for(unsigned int m=0; m<m_num_of_models; m++)
			{
				m_potentialmap[m].clear();
				m_potentialmap[m].resize(m_width*m_height);
			}
			clear();
		}
	}

	void clear()
	{
		for(unsigned int idx_model=0; idx_model<m_num_of_models; idx_model++)
		{
			clear(idx_model);
		}
	}

	void RequestComputePotential(MVT_State &state)
	{
		for(unsigned int m=0; m<m_num_of_models; m++)
		{
			RequestComputePotential(state,m);
		}
	}

	void GetPotential(MVT_State *p_state)
	{
		if( m_type==MVT_POTENTIALMAP_ROOT )
		{
			for( unsigned int m=0; m<NUM_OF_LIKELIHOOD_TYPE_ROOT; m++ )
			{
				MVT_Potential* p_potential = potentialmap(p_state,m);
				ComputePotentials(m);
				p_state->likelihood_root[m]       = p_potential->value;
			}
		}
		else if( m_type==MVT_POTENTIALMAP_PARTSNROOTS )
		{
			assert(0);
		}
	}

	double GetPotential(MVT_State *p_state, unsigned int idx_model)
	{
		MVT_Potential* p_potential = potentialmap(p_state,idx_model);
		ComputePotentials(idx_model);
		return p_potential->value;
	}

	double GetPotentialOccluded(unsigned int m)
	{
		if( m_type==MVT_POTENTIALMAP_PARTSNROOTS )
		{
			float potential=0;
			if( (m_models[m]!=NULL) && (m_models[m]->IsInitialize()) )
			{
				potential = ((MVT_Potential_Model_partsNroots*)m_models[m])->GetOccludedPotential();
			}
			return (double)potential;
		}
		else
		{
			return 0;
		}
	}

	void ComputePotentials()
	{
		for(unsigned int m=0; m<m_num_of_models; m++)
		{
			ComputePotentials(m);
		}
	}

private:

	void clear(unsigned int idx_model);

	void ComputePotentials(unsigned int idx_model);

	void RequestComputePotential(MVT_State& state, unsigned int idx_model);

	MVT_Potential* potentialmap(MVT_State *p_state, unsigned int m);

	std::vector<MVT_Potential> *m_potentialmap;   // 1D array of vector
	std::vector<MVT_State*>     *m_request_queue;
	unsigned int m_width;
	unsigned int m_height;

	MVT_Potential_Model**       m_models;
	unsigned int               m_num_of_models;

	ENUM_POTENTIALMAP_TYPE m_type;
	int m_idx_part;

} MVT_Potentialmap;

