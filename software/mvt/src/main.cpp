#include "mvt.h"

using namespace std;

std::ofstream f_result;
std::ofstream f_log;

bool g_b_initializing;
//void test_dpm(MVT_Image mvt_image);

int main( int argc, const char** argv )
{
cout.precision(5);

	g_param = ParseArguments(argc, argv);
	f_result.open(g_param.filepath_result.c_str());
	f_log.open(g_param.filepath_log.c_str());

	PrintArguments();

	select_gpu(0);

	unsigned int idx_start = g_param.idx_img_start;
	unsigned int idx_end   = g_param.idx_img_end!=0? g_param.idx_img_end:-1;

	MVT_Tracker  tracker(g_param);
	MVT_State*   p_state_cur = NULL;

	for( unsigned int idx_frame=idx_start; idx_frame<=idx_end; idx_frame++ )
	{
		char filepath_image[256] = {0,};
		std::string fmt_image;
		fmt_image.append(g_param.path_imgs).append(g_param.namefmt_imgs);
		sprintf(filepath_image,fmt_image.c_str(),idx_frame);

		char filepath_conf[256] = {0,};
		std::string fmt_dpmconf;
		fmt_dpmconf.append(g_param.path_dpmconfs).append(g_param.namefmt_dpmconfs);
		sprintf(filepath_conf,fmt_dpmconf.c_str(),idx_frame);
LOG( "frame: " << idx_frame << std::endl );

		cv::Mat image = cv::imread(std::string(filepath_image));
		cv::Mat draw;

if( g_param.is_vis )
{
	draw = cv::Mat(image.rows, image.cols, CV_64F);
	image.copyTo(draw);
}
if( image.empty() )
{
	LOG( "Tracking Finished." << std::endl );
	break;
}
		MVT_Image mvt_image;
		mvt_image.filepath_conf  = filepath_conf;
		mvt_image.pImage          = &image;

		if( idx_frame==idx_start )
		{
			g_b_initializing = true;
			p_state_cur = tracker.Initialize((int)idx_frame,mvt_image,g_param);
		}
		else
		{
			g_b_initializing = false;
			p_state_cur = tracker.Update((int)idx_frame,mvt_image);
LOG( "time: " << toc(TIMER_FRAME); );
		}

		cv::Rect rect = p_state_cur->bbox_root;
		f_result << idx_frame << " " << rect.x << " " << rect.y << " " << rect.width << " " << rect.height;
		f_result << " " << p_state_cur->viewpoint.azimuth << " " << p_state_cur->viewpoint.elevation << " " << p_state_cur->viewpoint.distance;
		f_result << " " << p_state_cur->center_root.x << " " << p_state_cur->center_root.y;
		for(unsigned int p=0; p<p_state_cur->pObject2d->Num_of_Part(); p++)
		{
			f_result << " " << (p_state_cur->centers[p].x-p_state_cur->center_root.x) << " " << (p_state_cur->centers[p].y-p_state_cur->center_root.y);
		}
		f_result << std::endl;

if( g_param.is_vis )
{
	if( g_param.use_alm || g_param.use_mil )
	{
		MVT_SampleSet::Draw( draw, p_state_cur );
	}
	cv::rectangle(draw,rect,cv::Scalar(0,0,0),2);
	cv::imshow("Multiview Tracker Display", draw);
	cv::waitKey(1000);
}
		draw.release();
		image.release();
	}

	f_result.close();
	f_log.close();
	return 1;
}
