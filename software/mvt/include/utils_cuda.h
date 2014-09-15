namespace cv
{
	namespace mil
	{
		class SampleSet;
		class HaarFtr;
	}
}

//float compute_hog_similarity( CUMATRIX hog1, CUMATRIX hog2 );
CUSampleSet SampleSet_to_CUSamples(cv::mil::SampleSet sampleset);
void Free_CUSampleSet(CUSampleSet css);

CUHaarFtr Alloc_CUHaarFtr(unsigned int n_rects, unsigned int n_weights);
void HaarFtr_to_CUHaarFtr(cv::mil::HaarFtr* haarftr, CUHaarFtr* cuhaarftr);
void Free_CUHaarFtr(CUHaarFtr ftr);
