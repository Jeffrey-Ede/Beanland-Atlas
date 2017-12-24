#pragma once

#include <defines.h>
#include <includes.h>

#include <utility.h>

namespace ba
{
	//Fraction of extracted rectangular region of interest to use when calculating the sum of squared differences to match it against another region
    #define QUANT_SYM_USE_FRAC 0.4

	//Default padding to apply when calculating Pearson's normalised product moment correlation coefficient in the Fourier domain
    #define FOURIER_PEARSON_USE_FRAC 0.5

	/*Sum of squared differences between 2 images. The second images is correlated against the first in the Fourier domain
	**src1: cv::Mat &, One of the images
	**src2: cv::Mat &, The second image
	**frac: float, Proportion of the first image's dimensions to pad it by
	**Returns,
	**cv::Mat, Sum of the squared differences
	*/
	cv::Mat ssd(cv::Mat &src1, cv::Mat &src2, float frac = QUANT_SYM_USE_FRAC);

	/*Pearson product moment correlation coefficient between 2 images. The second images is correlated against the 
	**first in the Fourier domain
	**src1: cv::Mat &, One of the images
	**src2: cv::Mat &, The second image
	**frac: float, Proportion of the first image's dimensions to pad it by
	**Returns,
	**cv::Mat, Pearson normalised product moment correlation coefficients. The result is not normalised
	*/
	cv::Mat fourier_pearson_corr(cv::Mat &src1, cv::Mat &src2, float frac = FOURIER_PEARSON_USE_FRAC);

	/*Create phase correlation specturm
	**src1: cv::Mat &, One of the images
	**src2: cv::Mat &, The second image
	**Returns,
	**cv::Mat, Phase correlation spectrum
	*/
	cv::Mat phase_corr_spectrum(cv::Mat &src1, cv::Mat &src2);
}