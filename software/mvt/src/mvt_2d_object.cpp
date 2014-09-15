/*
 * mvt_2d_object.cpp
 *
 *  Created on: Jul 7, 2013
 *      Author: changkyu
 */

#include "mvt.h"

MVT_2D_Object::MVT_2D_Object(MVT_3D_Object* pObject3d, cv::Mat** m_pImage_rectified)
{
	m_pObject3d          = pObject3d;
	m_num_of_parts       = m_pObject3d->Num_of_Parts();
	m_num_of_partsNroots = m_pObject3d->Num_of_PartsNRoots();
	m_p2dparts_front     = pObject3d->GetPartFrontInfo();

	m_2dparts.resize(m_num_of_partsNroots);
	m_is_occluded.resize(m_num_of_partsNroots);
	m_homography_part2front.resize(m_num_of_partsNroots);
	m_pImages_rectified = m_pImage_rectified;

	m_cumx_image_gradient_rectified.resize(m_num_of_partsNroots);
	for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	{
		m_cumx_image_gradient_rectified[pr].dims = NULL;
		m_cumx_image_gradient_rectified[pr].data = NULL;
	}
}

MVT_2D_Object::~MVT_2D_Object()
{

}

void cumatrix_from_cvmat(CUMATRIX &matrix, cv::Mat &mat)
{
	/* initialization */
	matrix.dims_num = 0;
	matrix.dims = NULL;
	matrix.length = 0;
	matrix.data = NULL;

	/* read dimension */
	matrix.dims_num = mat.channels()==1 ? mat.dims: mat.dims+1;

	/* allocate dims */
	matrix.dims = (int*)malloc(sizeof(int)*matrix.dims_num);
	matrix.length = 1;
	for(int i = 0; i < matrix.dims_num; i++)
	{
		matrix.dims[i] = i < 2 ? mat.size[i] : mat.channels();
		matrix.length *= matrix.dims[i];
	}

	/* allocate data */
	matrix.data = (float*)malloc(sizeof(float)*matrix.length);
	int idx=0;
	int n_row = mat.rows;
	int n_col = mat.cols;
	int n_ch  = mat.channels();
	for(int ch=0; ch<n_ch; ch++)
	{
		for(int c=0; c<n_col; c++)
		{
			for(int r=0; r<n_row; r++)
			{
				matrix.data[idx] = (float)mat.data[n_ch*n_col*r + n_ch*c + ch];
				idx++;
			}
		}
	}
}

void cvmat_from_cumatrix(CUMATRIX matrix, cv::Mat* pImage_dest, int type)
{
	int n_row = matrix.dims[0];
	int n_col = matrix.dims[1];
	pImage_dest->release();
	*pImage_dest = cv::Mat(n_row, n_col, type);

	int idx=0;
	int n_ch  = pImage_dest->channels();
	for(int ch=0; ch<n_ch; ch++)
	{
		for(int c=0; c<n_col; c++)
		{
			for(int r=0; r<n_row; r++)
			{
				pImage_dest->data[n_ch*n_col*r + n_ch*c + ch] = (unsigned char)matrix.data[idx];
				idx++;
			}
		}
	}
}

static void Rectify_Min(cv::Mat &homography, unsigned int width, unsigned int height, float H[9], double* pX_min, double* pY_min  )
{
	int idx=0;
	for( int c=0; c<3 ; c++)
	{
		for( int r=0; r<3 ; r++)
		{
			H[idx] = (float)(homography.at<double>(r,c));
			idx++;
		}
	}

	double x_corner[4], y_corner[4]/*, z_corner[4]*/;

	x_corner[0] =                            H[6];
	x_corner[1] =              height*H[3] + H[6];
	x_corner[2] = width*H[0]               + H[6];
	x_corner[3] = width*H[0] + height*H[3] + H[6];

	y_corner[0] =                            H[7];
	y_corner[1] =              height*H[4] + H[7];
	y_corner[2] = width*H[1]               + H[7];
	y_corner[3] = width*H[1] + height*H[4] + H[7];
	/*
	z_corner[0] =                            H[8];
	z_corner[1] =              height*H[5] + H[8];
	z_corner[2] = width*H[2]               + H[8];
	z_corner[3] = width*H[2] + height*H[5] + H[8];

	x_corner[0] = x_corner[0] / z_corner[0];
	x_corner[1] = x_corner[1] / z_corner[1];
	x_corner[2] = x_corner[2] / z_corner[2];
	x_corner[3] = x_corner[3] / z_corner[3];

	y_corner[0] = y_corner[0] / z_corner[0];
	y_corner[1] = y_corner[1] / z_corner[1];
	y_corner[2] = y_corner[2] / z_corner[2];
	y_corner[3] = y_corner[3] / z_corner[3];
	*/

	double x_min, x_max, y_min, y_max;
	x_min = x_corner[0]; x_max=x_corner[0];
	y_min = y_corner[0]; y_max=y_corner[0];
	for( int i=0 ; i<4 ; i++ )
	{
		if( x_corner[i] < x_min ) x_min = x_corner[i];
		if( x_corner[i] > x_max ) x_max = x_corner[i];
		if( y_corner[i] < y_min ) y_min = y_corner[i];
		if( y_corner[i] > y_max ) y_max = y_corner[i];
	}

	*pX_min = x_min;
	*pY_min = y_min;
}

void MVT_2D_Object::Rectify(cv::Point2d* points, unsigned int n_points, cv::Point2d* ret_points, cv::Mat &homography, unsigned int width, unsigned int height)
{
	float H[9];
	double x_min, y_min;
	Rectify_Min( homography, width, height, H, &x_min, &y_min );

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for( unsigned int p=0 ; p<n_points ; p++ )
	{
		double x = points[p].x;
		double y = points[p].y;
		double xH = x*H[0] + y*H[3] + H[6];
		double yH = x*H[1] + y*H[4] + H[7];
		/*
		double zH = x*H[2] + y*H[5] + H[8];
		xH = xH/zH;
		yH = yH/zH;
		*/
		ret_points[p].x = xH-x_min;
		ret_points[p].y = yH-y_min;
	}
}

void MVT_2D_Object::Rectify_Inv(cv::Point* points, unsigned int n_points, cv::Point2d* ret_points, cv::Mat &homography, unsigned int width, unsigned int height)
{
	float H[9];
	double x_min, y_min;
	Rectify_Min( homography, width, height, H, &x_min, &y_min );

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for( unsigned int p=0 ; p<n_points ; p++ )
	{
		double xH    = ((double)points[p].x+x_min);
		double yH    = ((double)points[p].y+y_min);
		double xH_t  = xH - H[6];
		double yH_t  = yH - H[7];
		double x = (xH_t*H[4]-yH_t*H[3]) / (H[0]*H[4]-H[1]*H[3]);
		double y = (xH_t*H[1]-yH_t*H[0]) / (H[3]*H[1]-H[4]*H[0]);

		ret_points[p].x = x;
		ret_points[p].y = y;
	}
}

cv::Point2d MVT_2D_Object::Rectify(cv::Point2d point, cv::Mat &homography, unsigned int width_img, unsigned int height_img)
{
	cv::Point2d ret_point;
	MVT_2D_Object::Rectify(&point, 1, &ret_point, homography, width_img, height_img);
	return ret_point;
}

cv::Point2d MVT_2D_Object::Rectify_Inv(cv::Point point, cv::Mat &homography, unsigned int width_img, unsigned int height_img)
{
	cv::Point2d ret_point;
	MVT_2D_Object::Rectify_Inv(&point, 1, &ret_point, homography, width_img, height_img);
	return ret_point;
}

static void homography_to_H(cv::Mat &homography, float* H)
{
	int idx=0;
	for( int c=0; c<3 ; c++)
	{
		for( int r=0; r<3 ; r++)
		{
			H[idx] = (float)(homography.at<double>(r,c));
			idx++;
		}
	}
}

void MVT_2D_Object::Rectify(CUMATRIX cumx_image, cv::Mat &homography, CUMATRIX &cumx_rectified)
{
	float H[9], T[9];
	homography_to_H(homography,H);
	cumx_rectified = rectify_image(cumx_image, H, T);
}

void MVT_2D_Object::SetViewpoint(MVT_Viewpoint viewpoint, cv::Mat *pImage)
{
	m_pImage = pImage;

	if( viewpoint.distance == viewpoint.distance ) // check NaN
	{
		m_pObject3d->Project_to_2D(this, viewpoint);

		CUMATRIX cumx_image;
		cumatrix_from_cvmat(cumx_image, *m_pImage);

		CUMATRIX cumx_image_gradient = compute_gradient_image(cumx_image);

		for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
		{
			m_pImages_rectified[pr]->release();
			free_cumatrix(&(m_cumx_image_gradient_rectified[pr]));
			m_cumx_image_gradient_rectified[pr].data = NULL;
			m_cumx_image_gradient_rectified[pr].dims = NULL;

			if( !m_is_occluded[pr] )
			{
				CUMATRIX tmp;
				MVT_2D_Object::Rectify(cumx_image,m_homography_part2front[pr], tmp);
				cvmat_from_cumatrix(tmp,m_pImages_rectified[pr],m_pImage->type());
				free_cumatrix(&tmp);

				Rectify(cumx_image_gradient, m_homography_part2front[pr], m_cumx_image_gradient_rectified[pr]);

				if( m_num_of_parts <= pr )
				{
					m_idx_root = pr;
				}
			}
			else
			{

			}
		}

		free_cumatrix(&cumx_image);
		free_cumatrix(&cumx_image_gradient);
	}
}

cv::Rect MVT_2D_Object::GetTargetBoundingBox(MVT_State &p_state)
{
	cv::Point2d pt_lt( INFINITY, INFINITY); // left-top
	cv::Point2d pt_rb(-INFINITY,-INFINITY); // right-bottom

	//for( unsigned int pr=0; pr<m_num_of_partsNroots; pr++ )
	for( unsigned int pr=0; pr<m_num_of_parts; pr++ )
	{
		if( !p_state.pObject2d->IsOccluded(pr) )
		{
			unsigned int n_vertices = m_2dparts[pr].vertices.size();
			for( unsigned int v=0; v<n_vertices; v++)
			{
				cv::Point2d pt(	(int)(p_state.centers[pr].x + m_2dparts[pr].vertices[v].x),
							    (int)(p_state.centers[pr].y + m_2dparts[pr].vertices[v].y) );

				pt_lt.x = pt_lt.x<pt.x?pt_lt.x:pt.x;
				pt_lt.y = pt_lt.y<pt.y?pt_lt.y:pt.y;
				pt_rb.x = pt_rb.x>pt.x?pt_rb.x:pt.x;
				pt_rb.y = pt_rb.y>pt.y?pt_rb.y:pt.y;
			}
		}
	}

	return cv::Rect( (int)(pt_lt.x), (int)(pt_lt.y), (int)(pt_rb.x-pt_lt.x), (int)(pt_rb.y-pt_lt.y) );
}

cv::Mat MVT_2D_Object::Draw(cv::Mat draw, unsigned int idx_part, cv::Point2d center, cv::Scalar color)
{
	unsigned int n_vertices = m_2dparts[idx_part].vertices.size();
	for( unsigned int v=0; v<n_vertices-1; v++)
	{
		cv::Point2i pt_start( (int)(center.x + m_2dparts[idx_part].vertices[v].x),
							  (int)(center.y + m_2dparts[idx_part].vertices[v].y)	);

		cv::Point2i pt_end  ( (int)(center.x + m_2dparts[idx_part].vertices[v+1].x),
							  (int)(center.y + m_2dparts[idx_part].vertices[v+1].y)	);

		cv::line(draw, pt_start, pt_end, color);
	}
	return draw;
}

bool MVT_2D_Object::ValidateRectifiedPoint(unsigned int idx_part, cv::Point &pt)
{
	bool b_updated = false;
	if(!IsOccluded(idx_part))
	{
		 if(pt.x<0)
		 {
			 pt.x=0;
			 b_updated = true;
		 }
		 if(pt.x>=m_pImages_rectified[idx_part]->cols)
		 {
			 pt.x = m_pImages_rectified[idx_part]->cols-1;
			 b_updated = true;
		 }
		 if(pt.y<0)
		 {
			 pt.y=0;
			 b_updated = true;
		 }
		 if(pt.y>=m_pImages_rectified[idx_part]->rows)
		 {
			 pt.y = m_pImages_rectified[idx_part]->rows-1;
			 b_updated = true;
		 }
	}
	return b_updated;
}
