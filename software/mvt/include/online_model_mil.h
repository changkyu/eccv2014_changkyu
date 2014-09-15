#ifndef __ONLINE_MODEL_MIL__
#define __ONLINE_MODEL_MIL__

#ifndef __COMPUTE_INTEGRAL__
#define __COMPUTE_INTEGRAL__
void compute_integral(const cv::Mat & img, std::vector<cv::Mat_<float> > & ii_imgs);
#endif

using namespace cv;
using namespace cv::mil;

namespace mvt
{

#define MVT_WEIGHT_ROOT_MIL (6)

	CV_EXPORTS class OnlineTracker : public SimpleTracker
	{
	public:
		OnlineTracker():SimpleTracker(){}
		~OnlineTracker(){}

		vectorf getCurState()
		{
			return _curState;
		}

		void train(cv::Mat image, std::vector<cv::Mat_<float> > ii_images, const CvRect& rect);

		void getScore(cv::Mat image, std::vector<cv::Mat_<float> > ii_images, SampleSet& p_samples, float* prob);
	};

	CV_EXPORTS class OnlineMILModel : public TrackingAlgorithm, public MVT_Potential_Model_partsNroots, public MVT_Potential_Model_Root
	{
	public:
		OnlineMILModel( unsigned int width, unsigned int height, unsigned int idx_part );
	    ~OnlineMILModel(){}

	    bool
	    initialize(const cv::Mat & image, const ObjectTrackerParams& params, const CvRect& init_bounding_box);

	    bool IsInitialize(){ return m_isInitialized;}

	    void train(int idx_frame, cv::Mat* pImage, const CvRect& rect);

	    void
	    SetImage(const cv::Mat* pImage);

	    void SetResizeFactor(double resize)
	    {
	    	if( resize > 1 )
	    	{
	    		m_resize = 1;
	    	}
	    	else
	    	{
	    		m_resize = resize;
	    	}
	    }

	    double
	    GetOccludedPotential();

	    float
	    GetPotential(MVT_State* p_states){assert(0);};

	    void
	    GetPotentials(std::vector<MVT_State*> &states, unsigned int n_states, float* p_potentials=NULL);

	    // we don't use this
		virtual bool
		update(const cv::Mat & image, const ObjectTrackerParams& params, cv::Rect & track_box){return true;};

	protected:
	    // A method to import an image to the type desired for the current algorithm
	    virtual void
	    import_image(const cv::Mat & image);

	private:

	    cv::Mat m_Image_resize;
	    std::vector<cv::Mat_<float> > ii_images_;

		OnlineTracker tracker_;
		cv::mil::SimpleTrackerParams tracker_params_;
		cv::mil::ClfStrongParams* clfparams_;

		// Feature parameters
		cv::mil::FtrParams* ftrparams_;
		cv::mil::HaarFtrParams haarparams_;

		bool m_isInitialized;

		unsigned int m_idx_frame_trained;

		double m_resize;
	};
}

#endif
