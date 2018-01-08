#pragma once

#include <includes.h>

namespace ba
{
	/*Mask noting the positions of the black pixels that are not padding an image
	**img: cv::Mat &, Floating point OpenCV mat to find the black non-bounding pixels of
	**Returns: 
	**cv::Mat, Mask indicating the positions of non-padding black pixels so that they can be infilled
	*/
	cv::Mat infilling_mask(cv::Mat &img);
}