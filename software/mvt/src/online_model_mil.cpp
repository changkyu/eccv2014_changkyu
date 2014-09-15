/*
 * online_model_mil.cpp
 *
 *  Created on: Jul 12, 2013
 *      Author: changkyu
 */

#include "mvt.h"
#include <time.h>

using namespace cv::mil;
using cv::mil::uint;

namespace cv
{
	namespace mil
	{
		void display(const cv::Mat & img, int fignum, float p);
	}
}

namespace mvt
{
	// OnlineTracker //////////////////////////////////////////////////////////////////////
	void
	OnlineTracker::train(cv::Mat image, std::vector<cv::Mat_<float> > ii_images, const CvRect& rect)
	{
		_curState[0] = (float) rect.x;
		_curState[1] = (float) rect.y;
		_curState[2] = (float) rect.width;
		_curState[3] = (float) rect.height;

		// train location clf (negx are randomly selected from image, posx is just the current tracker location)
		static SampleSet posx, negx;

		if (_trparams._negsamplestrat == 0)
		negx.sampleImage(image, ii_images, _trparams._negnumtrain, (int) _curState[2], (int) _curState[3]);
		else
		negx.sampleImage(image, ii_images, (int) _curState[0], (int) _curState[1], (int) _curState[2], (int) _curState[3],
						 (1.5f * _trparams._srchwinsz), _trparams._posradtrain + 5, _trparams._negnumtrain);

		if (_trparams._posradtrain == 1)
		posx.push_back(image, ii_images, (int) _curState[0], (int) _curState[1], (int) _curState[2], (int) _curState[3]);
		else
		posx.sampleImage(image, ii_images, (int) _curState[0], (int) _curState[1], (int) _curState[2], (int) _curState[3],
						 _trparams._posradtrain, 0, _trparams._posmaxtrain);

		_clf->update(posx, negx);

		// clean up
		posx.clear();
		negx.clear();

		_cnt++;
	}

	void
	OnlineTracker::getScore(cv::Mat image, std::vector<cv::Mat_<float> > ii_images, SampleSet& samples, float *prob)
	{
		_clf->classify(samples, _trparams._useLogR, prob);
	}

	// OnlineMILModel //////////////////////////////////////////////////////////////////////

	OnlineMILModel::OnlineMILModel( unsigned int width, unsigned int height, unsigned int idx_part )
    :
      TrackingAlgorithm(),
      MVT_Potential_Model_partsNroots(width, height, idx_part)
	{
	  cv::mil::RandomGenerator::initialize((int) time(0));
	  clfparams_ = new cv::mil::ClfMilBoostParams();
	  ftrparams_ = &haarparams_;
	  clfparams_->_ftrParams = ftrparams_;
	  m_isInitialized = false;

	  m_resize = 1;
	}

	bool
	OnlineMILModel::initialize(const cv::Mat & image, const ObjectTrackerParams& params,
            const CvRect& init_bounding_box)
	{
		SetResizeFactor(100.0 / (double)init_bounding_box.height);
		SetImage(&image);

		cv::Rect bbox;
		bbox.x      = init_bounding_box.x      * m_resize;
		bbox.y      = init_bounding_box.y      * m_resize;
		bbox.width  = init_bounding_box.width  * m_resize;
		bbox.height = init_bounding_box.height * m_resize;

		if( bbox.x < 0 ) bbox.x = 0;
		if( bbox.y < 0 ) bbox.y = 0;
		if( bbox.x + bbox.width >= image_.cols )
		{
			bbox.width = image_.cols-1 - bbox.x;
		}
		if( bbox.y + bbox.height >= image_.rows )
		{
			bbox.height = image_.rows-1 - bbox.y;
		}

		((cv::mil::ClfMilBoostParams*) clfparams_)->_numSel = params.num_classifiers_;
		((cv::mil::ClfMilBoostParams*) clfparams_)->_numFeat = params.num_features_;
		tracker_params_._posradtrain = params.pos_radius_train_;
		tracker_params_._negnumtrain = params.neg_num_train_;

		// Tracking parameters
		tracker_params_._init_negnumtrain = 65;
		tracker_params_._init_postrainrad = 3.0f;
		tracker_params_._initstate[0] = (float) bbox.x;
		tracker_params_._initstate[1] = (float) bbox.y;
		tracker_params_._initstate[2] = (float) bbox.width;
		tracker_params_._initstate[3] = (float) bbox.height;
		tracker_params_._srchwinsz = g_param.srchwinsz;
		tracker_params_._negsamplestrat = 1;
		tracker_params_._initWithFace = false;
		tracker_params_._debugv = false;
		tracker_params_._disp = false; // set this to true if you want to see video output (though it slows things down)

		clfparams_->_ftrParams->_width = (cv::mil::uint) bbox.width;
		clfparams_->_ftrParams->_height = (cv::mil::uint) bbox.height;

		tracker_.init(image_, tracker_params_, clfparams_);

		// Return success
		m_isInitialized = true;
		return true;
	}

	void
	OnlineMILModel::SetImage(const cv::Mat* pImage)
	{
		ii_images_.clear();

		m_Image_resize.release();
		Size size_small((int)(pImage->cols * m_resize), pImage->rows * m_resize);
		cv::resize(*pImage, m_Image_resize, size_small);

		import_image(m_Image_resize);
		compute_integral(image_,ii_images_);
	}

	double
	OnlineMILModel::GetOccludedPotential()
	{
		return 0;
	}


    void
    OnlineMILModel::GetPotentials(std::vector<MVT_State*> &states, unsigned int n_states, float* p_potentials)
    {
    	if( m_isInitialized==false )
    	{
    		printf("a?");
    	}

    	if( m_idx_part==(unsigned int)-1 )
    	{
    		SampleSet samples;
			for( unsigned int i=0 ; i<n_states ; i++ )
			{
				cv::Rect rect;
				rect.x      = states[i]->bbox_root.x      * m_resize;
				rect.y      = states[i]->bbox_root.y      * m_resize;
				rect.width  = states[i]->bbox_root.width  * m_resize;
				rect.height = states[i]->bbox_root.height * m_resize;

				if( rect.x < 0 ) rect.x = 0;
				if( rect.y < 0 ) rect.y = 0;
				if( rect.x + rect.width >= image_.cols )
				{
					rect.width = image_.cols-1 - rect.x;
				}
				if( rect.y + rect.height >= image_.rows )
				{
					rect.height = image_.rows-1 - rect.y;
				}

				samples.push_back(image_,ii_images_, rect.x,     rect.y,
													 rect.width, rect.height);
			}

			float potentials[n_states];
			for( unsigned int i=0 ; i<n_states ; i++ )
			{
				potentials[i] = 0;
			}
			tracker_.getScore(image_,ii_images_,samples,potentials);

			for( unsigned int i=0 ; i<n_states ; i++ )
			{
				if( potentials[i] != potentials[i] )
				{
					potentials[i] = -INFINITY;
				}
				else
				{
					potentials[i] = potentials[i] * m_weight;
				}
				states[i]->likelihood_root[MVT_LIKELIHOOD_MIL_ROOT] = potentials[i];
				if( p_potentials )
				{
					p_potentials[i] = potentials[i];
				}
			}
    	}
    	else
    	{
			SampleSet samples;
			for( unsigned int i=0 ; i<n_states ; i++ )
			{
				cv::Rect rect;
				rect.x      = states[i]->centers_rectified[m_idx_part].x-(int)(m_width/2);
				rect.y      = states[i]->centers_rectified[m_idx_part].x-(int)(m_height/2);
				rect.width  = m_width;
				rect.height = m_height;

				if( rect.x < 0 ) rect.x = 0;
				if( rect.y < 0 ) rect.y = 0;
				if( rect.x + rect.width >= image_.cols )
				{
					rect.width = image_.cols-1 - rect.x;
				}
				if( rect.y + rect.height >= image_.rows )
				{
					rect.height = image_.rows-1 - rect.y;
				}

				samples.push_back(image_,ii_images_,rect.x, rect.y, rect.width, rect.height);
			}

			float potentials[n_states];
			for( unsigned int i=0 ; i<n_states ; i++ )
			{
				potentials[i] = 0;
			}
			tracker_.getScore(image_,ii_images_,samples,potentials);

			for( unsigned int i=0 ; i<n_states ; i++ )
			{
				if( potentials[i] != potentials[i] )
				{
					potentials[i] = -INFINITY;
				}
				else
				{
					potentials[i] = potentials[i] * m_weight;
				}
				states[i]->likelihood_partsNroots[MVT_LIKELIHOOD_MIL] = potentials[i];
				if( p_potentials )
				{
					p_potentials[i] = potentials[i];
				}
			}
    	}
    }

    void
    OnlineMILModel::train(int idx_frame_trained, cv::Mat* pImage, const CvRect& rc)
	{
    	SetResizeFactor(100.0 / (double)rc.height);

    	if( (idx_frame_trained == -1) ||
    		(idx_frame_trained > (int)(m_idx_frame_trained + 10)) ||
    		(m_idx_part==(unsigned int)-1)
    	)
    	{
LOG("Train MIL ("<< m_idx_part << ")");
    		m_idx_frame_trained = idx_frame_trained;

        	SetImage(pImage);

        	cv::Rect rect = rc;

        	if( rect.x < 0 ) rect.x = 0;
			if( rect.y < 0 ) rect.y = 0;
        	if( rect.x + rect.width >= image_.cols )
			{
				rect.width = image_.cols-1 - rect.x;
			}
			if( rect.y + rect.height >= image_.rows )
			{
				rect.height = image_.rows-1 - rect.y;
			}

    		tracker_.train(image_,ii_images_,rect);
    	}
	}

	void
	OnlineMILModel::import_image(const cv::Mat & image)
	{
		// We want the internal version of the image to be gray-scale, so let's
		// do that here.  We'll handle cases where the input is either RGB, RGBA,
		// or already gray-scale.  I assume it's already 8-bit.  If not then
		// an error is thrown.  I'm not going to deal with converting properly
		// from every data type since that shouldn't be happening.

		// Make sure the input image pointer is valid
		if (image.empty())
		{
			std::cerr << "OnlineBoostingAlgorithm::import_image(...) -- ERROR!  Input image pointer is NULL!\n" << std::endl;
			exit(0); // <--- CV_ERROR?
		}

		// Now copy it in appropriately as a gray-scale, 8-bit image
		if (image.channels() == 4)
		{
			cv::cvtColor(image, image_, CV_RGBA2GRAY);
		}
		else if (image.channels() == 3)
		{
			cv::cvtColor(image, image_, CV_RGB2GRAY);
		}
		else if (image.channels() == 1)
		{
			image.copyTo(image_);
		}
		else
		{
			std::cerr << "OnlineBoostingAlgorithm::import_image(...) -- ERROR!  Invalid number of channels for input image!\n"
					<< std::endl;
			exit(0);
		}
	}

}



