#include <commensuration_tilts.h>

namespace ba
{
	/*Get ellipses describing each spot from their Scharr filtrates. Ellipses are checked using heuristic arguments:
	**ellipse shapes vary smoothly with time and ellipse shaps must be compatible with a projection of an array of
	**circles onto a flat detector
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual images to extract spots from
	**spot_pos: std::vector<cv::Point> &, Positions of located spots in aligned diffraction pattern
	**ellipses: std::vector<std::vector<std::vector<cv::Point>>> &, Positions of the minima and maximal extensions of spot ellipses.
	**The ellipses are decribed in terms of 4 points, clockwise from the top left as it makes it easy to use them to perform 
	**homomorphic warps. The nesting is each image, each spot in the order of their positions in the positions vector, set of
	**4 points (1 is extra) desctribing the ellipse, in that order
	**acc: cv::Mat &, Average of the aligned diffraction patterns
	*/
	void get_spot_ellipses(std::vector<cv::Mat> &mats, std::vector<cv::Point> &spot_pos, cv::Mat &acc, 
		std::vector<std::vector<std::vector<cv::Point>>> &ellipses)
	{
		//Use the Scharr filtrate of the aligned diffraction patterns to estimate the size of the ellipses
		std::vector<std::vector<cv::Point>> acc_ellipses(spot_pos.size());
		std::vector<std::vector<float>> thicknesses;
		est_ellipse_sizes(acc, spot_pos, acc_ellipses, thicknesses);

		cv::Mat scharr;
		scharr_amp(acc, scharr);


	}

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

	/*Use the accumulator spots to estimate the size of the ellipses
	**Inputs:
	**acc: cv::Mat &, Average of the aligned diffraction patterns
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**ellipses: std::vector<std::vector<std::vector<cv::Point>>> &, Positions of the minima and maximal extensions of spot ellipses.
	**The ellipses are decribed in terms of 4 points, clockwise from the top left as it makes it easy to use them to perform 
	**homomorphic warps. The nesting is each spot in the order of their positions in the positions vector, set of 4 points
	**(1 is extra) desctribing the ellipse, in that order
	**thicknesses: std::vector<std::vector<float>> &, Thicknesses of the elliptical Scharr filtrates at each of the ellipse points. 
	**These can be used to estimate the range of ellipse parameters contributing to the accumulator ellipses
	*/
	void est_ellipse_sizes(cv::Mat &acc, std::vector<cv::Point> spot_pos, std::vector<std::vector<cv::Point>> &ellipses, 
		std::vector<std::vector<float>> &thicknesses)
	{

	}
}