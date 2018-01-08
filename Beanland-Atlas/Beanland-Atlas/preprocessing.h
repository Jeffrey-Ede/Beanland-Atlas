#pragma once

#include <includes.h>

#include <utility.h>

namespace ba
{
	/*Preprocess each of the images by applying a bilateral filter and resizing them
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual images to preprocess
	**med_filt_size: int, Size of median filter
	*/
	void preprocess(std::vector<cv::Mat> &mats, int med_filt_size);
}