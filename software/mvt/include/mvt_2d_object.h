/*
 * mvt_2d_object.h
 *
 *  Created on: Jul 7, 2013
 *      Author: changkyu
 */

#ifndef MVT_2D_OBJECT_H_
#define MVT_2D_OBJECT_H_

class MVT_2D_Object
{
public:

	MVT_2D_Object(MVT_3D_Object* pObject3d, cv::Mat** pImages_rectified);
	~MVT_2D_Object();

	void SetViewpoint(MVT_Viewpoint viewpoint, cv::Mat* image);

	unsigned int Num_of_Part(){return m_num_of_parts;}
	unsigned int Num_of_PartsNRoots(){return m_num_of_partsNroots;}

	bool IsOccluded(unsigned int idx_part){ return m_is_occluded[idx_part]; }
	void SetOccluded(unsigned int idx_part, bool is_occluded){ m_is_occluded[idx_part]=is_occluded; }

	bool ValidateRectifiedPoint(unsigned int idx_part, cv::Point &pt);

	cv::Point2d GetRectifiedPoint(unsigned int idx_part, cv::Point2d point)
	{
		return MVT_2D_Object::Rectify( point, m_homography_part2front[idx_part], m_pImage->cols, m_pImage->rows);
	}

	void GetRectifiedPoint(unsigned int idx_part, cv::Point2d* points, unsigned int n_points, cv::Point2d* points_rectified)
	{
		MVT_2D_Object::Rectify( points, n_points, points_rectified, m_homography_part2front[idx_part], m_pImage->cols, m_pImage->rows);
	}

	cv::Point2d GetRestorePoint(unsigned int idx_part, cv::Point2d point_rectified)
	{
		return MVT_2D_Object::Rectify_Inv( point_rectified, m_homography_part2front[idx_part], m_pImage->cols, m_pImage->rows);
	}

	void GetRestorePoints(unsigned int idx_part, cv::Point* points_rectified, unsigned int n_points, cv::Point2d* points)
	{
		MVT_2D_Object::Rectify_Inv( points_rectified, n_points, points, m_homography_part2front[idx_part], m_pImage->cols, m_pImage->rows);
	}

	cv::Rect GetRectifiedRect(unsigned int idx_part, cv::Point2d center)
	{
		cv::Point2d center_rectified = GetRectifiedPoint(idx_part,center);
		double width  = (*m_p2dparts_front)[idx_part].width;
		double height = (*m_p2dparts_front)[idx_part].height;
		return cv::Rect( ((int)(center_rectified.x- width/2)),
	 				      ((int)(center_rectified.y-height/2)),
	 	 			      ((int)width),
					      ((int)height)   				 		);
	}

	cv::Mat* GetImage()
	{
		return m_pImage;
	}

	cv::Mat* GetRectifiedImage(unsigned int idx_part)
	{
		return m_pImages_rectified[idx_part];
	}

	CUMATRIX GetRectifiedGradientImage(unsigned int idx_part)
	{
		return m_cumx_image_gradient_rectified[idx_part];
	}

	MVT_2D_Part GetPartInfo(unsigned int idx_part)
	{
		return m_2dparts[idx_part];
	}

	unsigned int GetRootIndex()
	{
		return m_idx_root;
	}

	cv::Rect GetTargetBoundingBox(MVT_State &p_state);

	cv::Mat Draw(cv::Mat draw, unsigned int idx_part, cv::Point2d center, cv::Scalar color);

	static void         Rectify(CUMATRIX      cumx_image, cv::Mat &homography, CUMATRIX &cumx_rectified);
	static void          Rectify(    cv::Point2d* points, unsigned int n_points, cv::Point2d* ret_points, cv::Mat &homography, unsigned int width_img, unsigned int height_img);
	static void          Rectify_Inv(cv::Point*   points, unsigned int n_points, cv::Point2d* ret_points, cv::Mat &homography, unsigned int width_img, unsigned int height_img);
	static cv::Point2d   Rectify(cv::Point2d       point,  cv::Mat &homography, unsigned int width_img, unsigned int height_img);
	static cv::Point2d   Rectify_Inv(cv::Point     point,  cv::Mat &homography, unsigned int width_img, unsigned int height_img);

	std::vector<MVT_2D_Part>        m_2dparts;
	std::vector<cv::Mat>		    m_homography_part2front;

private:

	unsigned int                 m_idx_root;

	unsigned int                 m_num_of_parts;
	unsigned int                 m_num_of_partsNroots;

	std::vector<bool>			   m_is_occluded;

	cv::Mat* 					   m_pImage;

	cv::Mat**                      m_pImages_rectified;
	std::vector<CUMATRIX>          m_cumx_image_gradient_rectified;

	class MVT_3D_Object*          m_pObject3d;

	std::vector<MVT_2D_Part_Front>* m_p2dparts_front;
};

#endif /* MVT_2D_OBJECT_H_ */
