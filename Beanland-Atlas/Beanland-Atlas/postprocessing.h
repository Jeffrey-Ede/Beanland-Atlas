#pragma once

#include <includes.h>

#include <utility.h>

namespace ba
{
	/*Mask noting the positions of the black pixels that are not padding an image
	**img: cv::Mat &, Floating point OpenCV mat to find the black non-bounding pixels of
	**Returns: 
	**cv::Mat, Mask indicating the positions of non-padding black pixels so that they can be infilled
	*/
	cv::Mat infilling_mask(cv::Mat &img);

	/*Navier-Stokes interpolate the value of each pixel in an image from the surrounding pixels using the spatial scales
	**on the image
	**img: cv::Mat &, Image to find the Navier-stokes inflows to the pixels of
	**dst: cv::Mat &, Output Navier-Stokes reconstruction
	*/
	void navier_stokes_reconstruction(cv::Mat &img, cv::Mat &dst);
}