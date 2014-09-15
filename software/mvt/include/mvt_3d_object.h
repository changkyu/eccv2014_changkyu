/*
 * mvt_3d_object_model.h
 *
 *  Created on: Jul 7, 2013
 *      Author: changkyu
 */

#ifndef MVT_3D_OBJECT_H_
#define MVT_3D_OBJECT_H_

class MVT_3D_Object
{
public:
	MVT_3D_Object(ENUM_OBJECT_CATEGORY object_category, const char* filepath_3dobject_model);
	~MVT_3D_Object();

	void Project_to_2D(MVT_2D_Object* ret_object2d, MVT_Viewpoint viewpoint);
	unsigned int Num_of_Parts(){ return m_num_of_parts; }
	unsigned int Num_of_PartsNRoots(){ return m_num_of_partsNroots; }

	std::vector<MVT_AZIMUTH>   GetDiscAzimuth()  {return m_disc_azimuth;  }
	std::vector<MVT_ELEVATION> GetDiscElevation(){return m_disc_elevation;}
	std::vector<MVT_DISTANCE>  GetDiscDistance() {return m_disc_distance; }

	std::vector<MVT_2D_Part_Front>* GetPartFrontInfo()
	{
		return &m_2dparts_front;
	}

	MVT_2D_Part_Front GetPartFrontInfo(unsigned int idx_part)
	{
		return m_2dparts_front[idx_part];
	}

	std::string GetName_PartsNViews(int p){ return m_2dparts_front[p].name; }

	void Print();

private:

	MVT_Viewpoint_IDX DiscViewpoint_from_ContViewpoint(MVT_Viewpoint cont_viewpoint);
	cv::Mat Homography(std::vector<cv::Point2d> &src, std::vector<cv::Point2d> &dest);
	cv::Mat Projection( MVT_Viewpoint viewpoint, CvPoint3D64f* p_xyz_camera=NULL );

	ENUM_OBJECT_CATEGORY m_object_category;

	std::vector<MVT_3D_Part>   		 m_3dparts;
	std::vector<MVT_2D_Part_Front>   m_2dparts_front;
	unsigned int m_num_of_parts;
	unsigned int m_num_of_partsNroots;

	std::vector<std::string> m_names_partsNroots;

	unsigned int m_num_of_disc_azimuth;
	unsigned int m_num_of_disc_elevation;
	unsigned int m_num_of_disc_distance;

	std::vector<MVT_AZIMUTH  > m_disc_azimuth;
	std::vector<MVT_ELEVATION> m_disc_elevation;
	std::vector<MVT_DISTANCE > m_disc_distance;

	bool**** m_is_occluded;

};

#endif /* MVT_3D_OBJECT_H_ */
