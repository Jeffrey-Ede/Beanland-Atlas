#pragma once

#include <includes.h>
#include <defines.h>

namespace ba
{


	/*Amplitude of image's Scharr filtrate
	**Inputs:
	**img: cv::Mat &, Floating point image to get the Scharr filtrate of
	**scharr_amp: cv::Mat &, OpenCV mat to output the amplitude of the Scharr filtrate to
	*/
	void scharr_amp(cv::Mat &img, cv::Mat &scharr_amp);
}