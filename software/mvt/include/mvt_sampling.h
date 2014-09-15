typedef class MVT_Sampling
{
public:
	MVT_Sampling(MVT_Param param, MVT_3D_Object* pObject3d);
	~MVT_Sampling();

	MVT_Viewpoint Sampling_Viewpoint();

	void Sampling_Centers(MVT_State* states, unsigned int n_states, cv::Rect &bbox);

	void Sampling_PartsCenters(MVT_2D_Object* pObject2d, MVT_State* p_state_ret, unsigned int n_states);

	void SetRefState(MVT_State* p_state_ref);

	bool IsValidViewpoint(MVT_2D_Object* pObject2d);

private:

	// Valid Sampling
	void ValidateAzimuth(MVT_AZIMUTH& azimuth);
	bool IsValidElevation(MVT_ELEVATION elevation){return (m_elevation_min<=elevation /*&& elevation<=m_elevation_max*/);}
	bool IsValidDistance(MVT_DISTANCE distance)   {return ( m_distance_min<=distance  &&  distance<= m_distance_max);}

	void RandomsNormal(unsigned int idx_part, cv::Point2d* means, cv::Point* points, unsigned int n_points);

	MVT_ELEVATION m_elevation_min;
	MVT_ELEVATION m_elevation_max;

	MVT_DISTANCE  m_distance_min;
	MVT_DISTANCE  m_distance_max;

	unsigned int  m_x_min;
	unsigned int  m_x_max;
	unsigned int  m_y_min;
	unsigned int  m_y_max;
	unsigned int* m_x_rectified_min;
	unsigned int* m_x_rectified_max;
	unsigned int* m_y_rectified_min;
	unsigned int* m_y_rectified_max;

	/* [PartsNroots] x [Random number] */
	int*  m_cdf_x_rectified_min;
	int*  m_cdf_x_rectified_max;
	int*  m_cdf_y_rectified_min;
	int*  m_cdf_y_rectified_max;
	int*  m_cdf_inv_x_rectified_min;
	int*  m_cdf_inv_x_rectified_max;
	int*  m_cdf_inv_y_rectified_min;
	int*  m_cdf_inv_y_rectified_max;

	int** m_cdf_mapping_x_rectified;
	int** m_cdf_mapping_y_rectified;
	int** m_cdf_mapping_inv_x_rectified;
	int** m_cdf_mapping_inv_y_rectified;

	// Information for Reference State
	unsigned int m_num_of_partsNroots;
	unsigned int m_num_of_parts;

	double  m_azimuth_ref;
	double  m_elevation_ref;
	double  m_distance_ref;
	double  m_center_root_x_ref;
	double  m_center_root_y_ref;
	double* m_centers_part_x_ref;
	double* m_centers_part_y_ref;
	cv::Rect m_bbox_ref;

	unsigned int m_num_of_samples;

	MVT_STD_Sampling m_std_sampling;

	cv::RNG* m_rng;

}MVT_Sampling;

