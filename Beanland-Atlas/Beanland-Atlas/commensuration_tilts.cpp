#include <commensuration_tilts.h>

namespace ba
{
	
	//display_CV(acc);
	//cv::Mat scharr;
	//scharr_amp(acc, scharr);
	//display_CV(scharr);

	/*Amplitude of image's Scharr filtrate
	**Inputs:
	**img: cv::Mat &, Floating point image to get the Scharr filtrate of
	**scharr_amp: cv::Mat &, OpenCV mat to output the amplitude of the Scharr filtrate to
	*/
	void scharr_amp(cv::Mat &img, cv::Mat &scharr_amp)
	{
		//Aquire the image's Sobel filtrate
		cv::Mat gradx, grady;
		cv::Sobel(img, gradx, CV_32FC1, 0, 1, CV_SCHARR);
		cv::Sobel(img, grady, CV_32FC1, 1, 0, CV_SCHARR);

		//Sum the gradients in quadrature
		float *p, *q;
        #pragma omp parallel for
		for (int i = 0; i < img.rows; i++)
		{
			p = gradx.ptr<float>(i);
			q = grady.ptr<float>(i);
			for (int j = 0; j < img.cols; j++)
			{
				p[j] = std::sqrt(p[j]*p[j] + q[j]*q[j]);
			}
		}

		scharr_amp = gradx;
	}
}