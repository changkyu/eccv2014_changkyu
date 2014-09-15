/*
 * mvt_3d_object_model.cpp
 *
 *  Created on: Jul 7, 2013
 *      Author: changkyu
 */
#include "mvt.h"

MVT_3D_Object::MVT_3D_Object(ENUM_OBJECT_CATEGORY object_category, const char* filepath_3dobject_model)
{
	m_object_category = object_category;

	MATFile* mat = matOpen(filepath_3dobject_model,"r");
	if( mat != NULL )
	{
		mxArray* pmxCad = matGetVariable(mat, "cad");
		pmxCad = mxGetCell(pmxCad,0);

		/*
		 * Names of parts
		 */
		mxArray* pmxNames    = mxGetField(pmxCad, 0, "pnames");
		m_num_of_partsNroots = mxGetNumberOfElements(pmxNames);
		for( unsigned int i=0; i<m_num_of_partsNroots; i++)
		{
			mxArray* pmxName = mxGetCell(pmxNames, i);
			m_names_partsNroots.push_back(MxArray(pmxName).toString());
		}

		/*
		 * Parts2d_Front
		 */
		mxArray* pmxPars2d_front = mxGetField(pmxCad, 0, "parts2d_front");
		for( unsigned int i=0; i<m_num_of_partsNroots; i++)
		{
			MVT_2D_Part_Front front;
			front.width    = MxArray(mxGetField(pmxPars2d_front, i, "width"   )).toDouble();
			front.height   = MxArray(mxGetField(pmxPars2d_front, i, "height"  )).toDouble();
			front.distance = MxArray(mxGetField(pmxPars2d_front, i, "distance")).toDouble();

			cv::Mat mat_tmp = MxArray(mxGetField(pmxPars2d_front, i, "vertices")).toMat();
			unsigned int n_row = mat_tmp.rows;
			for( unsigned int r=0; r<n_row ; r++)
			{
				front.vertices.push_back(cv::Point2d(mat_tmp.at<double>(r,0), mat_tmp.at<double>(r,1)));
			}

			front.center   = MxArray(mxGetField(pmxPars2d_front, i, "center")).toPoint_<double>();
			front.viewport = MxArray(mxGetField(pmxPars2d_front, i, "viewport")).toDouble();
			front.name     = MxArray(mxGetField(pmxPars2d_front, i, "pname")).toString();

			m_2dparts_front.push_back(front);
		}

		/*
		 * Part
		 */
		mxArray* pmxParts = mxGetField(pmxCad, 0, "parts");
		m_num_of_parts = mxGetNumberOfElements(pmxParts);

		for( unsigned int i=0 ; i<m_num_of_parts; i++ )
		{
			MVT_3D_Part part;

			/* Vertices */
			mxArray* mxVertices = mxGetField(pmxParts, i, "vertices");
			cv::Mat matVertices = MxArray(mxVertices).toMat();

			unsigned int n_rows = matVertices.rows;
			for( unsigned int r=0; r<n_rows; r++)
			{
				cv::Mat matTmp = matVertices.row(r);

				part.vertices.push_back
				(
					cv::Point3d( matTmp.at<double>(0),
								 matTmp.at<double>(1),
								 matTmp.at<double>(2)  )
				);
			}

			/* plane */
			MxArray MxPlane(mxGetField(pmxParts, i, "plane"));
			for( int j=0 ; j<4; j++ )
			{
				part.plane[j] = MxPlane.at<double>(j);
			}

			/* center */
			MxArray MxCenter(mxGetField(pmxParts, i, "center"));
			part.center.x = MxCenter.at<double>(0);
			part.center.y = MxCenter.at<double>(1);
			part.center.z = MxCenter.at<double>(2);

			/* xaxis */
			MxArray MxXAxis(mxGetField(pmxParts, i, "xaxis"));
			part.xaxis.x = MxXAxis.at<double>(0);
			part.xaxis.y = MxXAxis.at<double>(1);
			part.xaxis.z = MxXAxis.at<double>(2);

			/* yaxis */
			MxArray MxYAxis(mxGetField(pmxParts, i, "yaxis"));
			part.yaxis.x = MxYAxis.at<double>(0);
			part.yaxis.y = MxYAxis.at<double>(1);
			part.yaxis.z = MxYAxis.at<double>(2);

			m_3dparts.push_back(part);

			/*
			 * Occlusion Information
			 */
			mxArray* pmxAzimuth   = mxGetField(pmxCad, 0, "azimuth" );
			m_disc_azimuth        = MxArray(pmxAzimuth).toVector<MVT_AZIMUTH>();

			mxArray* pmxElevation = mxGetField(pmxCad, 0, "elevation" );
			m_disc_elevation      = MxArray(pmxElevation).toVector<MVT_ELEVATION>();

			mxArray* pmxDistance  = mxGetField(pmxCad, 0, "distance" );
			m_disc_distance       = MxArray(pmxDistance).toVector<MVT_DISTANCE>();

			mxArray* pmxParts2d   = mxGetField(pmxCad, 0, "parts2d");

			m_num_of_disc_azimuth   = m_disc_azimuth.size();
			m_num_of_disc_elevation = m_disc_elevation.size();
			m_num_of_disc_distance  = m_disc_distance.size();

			m_is_occluded = new bool***[m_num_of_disc_azimuth];
			for( unsigned int a=0; a<m_num_of_disc_azimuth; a++ )
			{
				m_is_occluded[a] = new bool**[m_num_of_disc_elevation];
				for( unsigned int e=0; e<m_num_of_disc_elevation; e++ )
				{
					m_is_occluded[a][e] = new bool*[m_num_of_disc_distance];
					for( unsigned int d=0; d<m_num_of_disc_distance; d++ )
					{
						m_is_occluded[a][e][d] = new bool[m_num_of_partsNroots];
						unsigned int idx = m_num_of_disc_distance*m_num_of_disc_elevation*a +
								             m_num_of_disc_distance*e +
								             d;
						for( unsigned int p=0; p<m_num_of_partsNroots; p++)
						{
							mxArray* pmxPart = mxGetField(pmxParts2d, idx, m_names_partsNroots[p].c_str());
							m_is_occluded[a][e][d][p] = mxIsEmpty(pmxPart);
						}
					}
				}
			}
		}

		mxDestroyArray(pmxCad);
		matClose(mat);
	}
}

MVT_3D_Object::~MVT_3D_Object()
{
	for( unsigned int i=0 ; i<m_num_of_parts; i++ )
	{
		m_3dparts[i].vertices.clear();
	}
	m_3dparts.clear();

	for( unsigned int a=0; a<m_num_of_disc_azimuth; a++ )
	{
		for( unsigned int e=0; e<m_num_of_disc_elevation; e++ )
		{
			for( unsigned int d=0; d<m_num_of_disc_distance; d++ )
			{
				delete[] m_is_occluded[a][e][d];
			}
			delete[] m_is_occluded[a][e];
		}
		delete[] m_is_occluded[a];
	}
	delete[] m_is_occluded;
}

MVT_Viewpoint_IDX MVT_3D_Object::DiscViewpoint_from_ContViewpoint(MVT_Viewpoint cont_viewpoint)
{
	MVT_Viewpoint_IDX idx = {0,};

	MVT_AZIMUTH   cont_azimuth   = CHECK_RANGE_DEGREE(cont_viewpoint.azimuth);
	MVT_ELEVATION cont_elevation = CHECK_RANGE_DEGREE(cont_viewpoint.elevation);
	MVT_DISTANCE  cont_distance  = cont_viewpoint.distance;

	MVT_AZIMUTH left_a = m_disc_azimuth[0];
	MVT_AZIMUTH diff_left_a = cont_azimuth-left_a;
	for( unsigned int a=1; a<=m_num_of_disc_azimuth; a++)
	{
		MVT_AZIMUTH right_a = (a==m_num_of_disc_azimuth) ? 360:m_disc_azimuth[a];
		MVT_AZIMUTH diff_right_a = cont_azimuth-right_a;
		if( diff_right_a <= 0 )
		{
			idx.a = diff_left_a < (-diff_right_a) ? a-1 : a;
			if(idx.a==m_num_of_disc_azimuth)idx.a = 0;
			break;
		}
		else
		{
			diff_left_a = diff_right_a;
		}
	}

	idx.e = m_num_of_disc_elevation-1;
	MVT_ELEVATION left_e = m_disc_elevation[0];
	MVT_ELEVATION diff_left_e = cont_elevation-left_e;
	for( unsigned int e=1; e<m_num_of_disc_elevation; e++)
	{
		MVT_ELEVATION right_e = m_disc_elevation[e];
		MVT_ELEVATION diff_right_e = cont_elevation-right_e;
		if( diff_right_e <= 0 )
		{
			idx.e = diff_left_e < (-diff_right_e) ? e-1 : e;
			break;
		}
		else
		{
			diff_left_e = diff_right_e;
		}
	}

	idx.d = m_num_of_disc_distance-1;
	MVT_DISTANCE left_d = m_disc_distance[0];
	MVT_DISTANCE diff_left_d = cont_distance-left_d;
	for( unsigned int d=1; d<m_num_of_disc_distance; d++)
	{
		MVT_DISTANCE right_d = m_disc_distance[d];
		MVT_DISTANCE diff_right_d = cont_distance-right_d;
		if( diff_right_d <= 0 )
		{
			idx.d = diff_left_d < (-diff_right_d) ? d-1 : d;
			break;
		}
		else
		{
			diff_left_d = diff_right_d;
		}
	}

	return idx;
}

cv::Mat MVT_3D_Object::Projection( MVT_Viewpoint viewpoint, CvPoint3D64f* p_xyz_camera/*=NULL*/ )
{
	double a = CONV_RADIUS(viewpoint.azimuth);
	double e = CONV_RADIUS(viewpoint.elevation);
	double d = viewpoint.distance;

	// Camera Center
	CvPoint3D64f xyz_camera;
	xyz_camera.x =  d*cos(e)*sin(a);
	xyz_camera.y = -d*cos(e)*cos(a);
	xyz_camera.z =  d*sin(e);
	double c[3][1] = {{xyz_camera.x},{xyz_camera.y},{xyz_camera.z}};
	cv::Mat C(3,1, CV_64FC1,c);

	if( p_xyz_camera != NULL )
	{
		(*p_xyz_camera) = xyz_camera;
	}

	a   = -a;
	e = -(M_PI/2 - e);

	// Rotation Matrix
	double z[3][3] = {{cos(a), -sin(a), 0},
                       {sin(a),  cos(a), 0},
                       {     0,       0,  1}};

	double x[3][3] = {{1,      0,        0},
                       {0, cos(e), -sin(e)},
                       {0, sin(e),  cos(e)}};

	cv::Mat R = (cv::Mat(3,3,CV_64F,x))*(cv::Mat(3,3,CV_64F,z));

	cv::Mat RC = R * C;

	double p1[4][4] = {{1, 0,  0, 0},
                        {0, 1,  0, 0},
                        {0, 0,  0, 1},
                        {0, 0, -1, 0}};
	double p2[4][4] = {{R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), -RC.at<double>(0,0)},
                        {R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), -RC.at<double>(1,0)},
                        {R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), -RC.at<double>(2,0)},
                        {0        , 0        , 0        ,  1         }};

	cv::Mat P = (cv::Mat(4,4,CV_64F,p1))*(cv::Mat(4,4,CV_64F,p2));

	return P;
}

cv::Mat MVT_3D_Object::Homography(std::vector<cv::Point2d> &src, std::vector<cv::Point2d> &dest)
{
	double m[8][9] =
	{
			{       0,        0, 0, -src[0].x, -src[0].y, -1,  src[0].x*dest[0].y,  src[0].y*dest[0].y,  dest[0].y },
			{src[0].x, src[0].y, 1,         0,         0,  0, -src[0].x*dest[0].x, -src[0].y*dest[0].x, -dest[0].x },
			{       0,        0, 0, -src[1].x, -src[1].y, -1,  src[1].x*dest[1].y,  src[1].y*dest[1].y,  dest[1].y },
			{src[1].x, src[1].y, 1,         0,         0,  0, -src[1].x*dest[1].x, -src[1].y*dest[1].x, -dest[1].x },
			{       0,        0, 0, -src[2].x, -src[2].y, -1,  src[2].x*dest[2].y,  src[2].y*dest[2].y,  dest[2].y },
			{src[2].x, src[2].y, 1,         0,         0,  0, -src[2].x*dest[2].x, -src[2].y*dest[2].x, -dest[2].x },
			{       0,        0, 0, -src[3].x, -src[3].y, -1,  src[3].x*dest[3].y,  src[3].y*dest[3].y,  dest[3].y },
			{src[3].x, src[3].y, 1,         0,         0,  0, -src[3].x*dest[3].x, -src[3].y*dest[3].x, -dest[3].x }
	};
	cv::SVD svd(cv::Mat(8,9,CV_64F, m),cv::SVD::FULL_UV);
	cv::Mat h = svd.vt.row(8);
	cv::Mat H = h.reshape(0,3);

	H = H * (1/H.at<double>(2,2));

	return H;
}

void MVT_3D_Object::Project_to_2D(MVT_2D_Object* ret_object2d, MVT_Viewpoint viewpoint)
{
	MVT_Viewpoint_IDX idx_viewpoint = DiscViewpoint_from_ContViewpoint(viewpoint);

	cv::Mat mat_P = Projection(viewpoint);
	double m_R[3][3] = { {1,  0, 0.5},
						  {0, -1, 0.5},
						  {0,  0,   1}  };

	for(unsigned int pv=0 ; pv<m_num_of_partsNroots ; pv++)
	{
		ret_object2d->SetOccluded(pv,m_is_occluded[idx_viewpoint.a][idx_viewpoint.e][idx_viewpoint.d][pv]);
/*		if( ret_object2d->IsOccluded(pv) )
		{

		}
		else
*/		{
			if(
				( m_object_category==OBJECT_CATEGORY_CHAIR && pv>10 ) ||
				( m_object_category==OBJECT_CATEGORY_TABLE && pv>5 )
			)
			{
				// TODO
				// note render.m
			}
			else
			{
				cv::Mat mat_R = m_2dparts_front[pv].viewport * cv::Mat(3,3,CV_64F, m_R);
				mat_R.at<double>(2,2)=1;

				if( pv < m_num_of_parts )
				{
					unsigned int n_vertices = m_3dparts[pv].vertices.size();
					double m[4][n_vertices];
					for( unsigned int v=0; v<n_vertices; v++ )
					{
						m[0][v] = m_3dparts[pv].vertices[v].x;
						m[1][v] = m_3dparts[pv].vertices[v].y;
						m[2][v] = m_3dparts[pv].vertices[v].z;
						m[3][v] = 1;
					}

					cv::Mat part = mat_P*cv::Mat(4,n_vertices,CV_64F,m);
					for( unsigned int v=0; v<n_vertices; v++ )
					{
						// normalize
						double tmp = part.at<double>(3,v);
						for( unsigned int r=0; r<4; r++ )
						{
							part.at<double>(r,v) = part.at<double>(r,v) / tmp;
						}
					}

					double m_vertices2d[3][n_vertices];
					for( unsigned int v=0; v<n_vertices; v++ )
					{
						m_vertices2d[0][v] = part.at<double>(0,v);
						m_vertices2d[1][v] = part.at<double>(1,v);
						m_vertices2d[2][v] = 1;
					}
					cv::Mat mat_v = mat_R*cv::Mat(3,n_vertices,CV_64F, m_vertices2d);

					double m_center3d[4] = { m_3dparts[pv].center.x,
											  m_3dparts[pv].center.y,
											  m_3dparts[pv].center.z,
											  1 };
					cv::Mat center = mat_P*cv::Mat(4,1,CV_64F, m_center3d);
					center = center * (1/center.at<double>(3));// normalize

					double m_center2d[3] = { center.at<double>(0),
											  center.at<double>(1),
											  1                      };
					cv::Mat mat_c = mat_R*cv::Mat(3,1,CV_64F, m_center2d);

					ret_object2d->m_2dparts[pv].vertices.clear();
					ret_object2d->m_2dparts[pv].vertices.resize(n_vertices);
					for( unsigned int v=0; v<n_vertices; v++ )
					{
						ret_object2d->m_2dparts[pv].vertices[v].x = mat_v.at<double>(0,v) - mat_c.at<double>(0);
						ret_object2d->m_2dparts[pv].vertices[v].y = mat_v.at<double>(1,v) - mat_c.at<double>(1);
					}
					ret_object2d->m_2dparts[pv].center.x
					 = mat_c.at<double>(0) - mat_R.at<double>(0,2);
					ret_object2d->m_2dparts[pv].center.y
					 = mat_c.at<double>(1) - mat_R.at<double>(1,2);
				}
				else
				{
					int x_max=-INFINITY, x_min=INFINITY, y_max=-INFINITY, y_min=INFINITY;
					for( unsigned int p=0; p<m_num_of_parts; p++ )
					{
						unsigned int n_vertices = ret_object2d->m_2dparts[p].vertices.size();
						for( unsigned int v=0; v<n_vertices; v++ )
						{
							int x = ret_object2d->m_2dparts[p].vertices[v].x + ret_object2d->m_2dparts[p].center.x;
							int y = ret_object2d->m_2dparts[p].vertices[v].y + ret_object2d->m_2dparts[p].center.y;

							x_max = (x > x_max) ? x : x_max;
							x_min = (x < x_min) ? x : x_min;
							y_max = (y > y_max) ? y : y_max;
							y_min = (y < y_min) ? y : y_min;
						}
					}

					ret_object2d->m_2dparts[pv].vertices.clear();
					ret_object2d->m_2dparts[pv].vertices.resize(5);
					ret_object2d->m_2dparts[pv].vertices[0] = cv::Point2d(x_min, y_min);
					ret_object2d->m_2dparts[pv].vertices[1] = cv::Point2d(x_min, y_max);
					ret_object2d->m_2dparts[pv].vertices[2] = cv::Point2d(x_max, y_max);
					ret_object2d->m_2dparts[pv].vertices[3] = cv::Point2d(x_max, y_min);
					ret_object2d->m_2dparts[pv].vertices[4] = cv::Point2d(x_min, y_min);

					ret_object2d->m_2dparts[pv].center.x = 0;
					ret_object2d->m_2dparts[pv].center.y = 0;
				}

				ret_object2d->m_homography_part2front[pv].release();
				ret_object2d->m_homography_part2front[pv]
				 = Homography(ret_object2d->m_2dparts[pv].vertices,m_2dparts_front[pv].vertices);
			}
		}
	}
}

void MVT_3D_Object::Print()
{
	std::cout << "# of parts: " << m_num_of_parts << std::endl;
	for( unsigned int p=0; p<m_num_of_parts; p++)
	{
		std::cout << p << "-st part >>" << std::endl;

		int n_vertices = m_3dparts[p].vertices.size();
		for( int j=0; j<n_vertices; j++ )
		{
			std::cout << "vertices["<<j<<"]: " << m_3dparts[p].vertices[j] << std::endl;
		}
		for( int j=0; j<4; j++ )
		{
			std::cout << "plane["<<j<<"]:" << m_3dparts[p].plane[j] << std::endl;
		}
		std::cout << "center: " << m_3dparts[p].center << std::endl;
		std::cout << "xaxis : " << m_3dparts[p].xaxis  << std::endl;
		std::cout << "yaxis : " << m_3dparts[p].yaxis  << std::endl;;
	}
}

