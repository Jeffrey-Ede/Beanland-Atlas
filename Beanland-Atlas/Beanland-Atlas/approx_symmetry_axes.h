#pragma once

#include <includes.h>

namespace ba
{
	/*Downsamples amalgamation of aligned diffraction patterns, then finds approximate axes of symmetry
	**Inputs:
	**amalg: cv::Mat, OpenCV mat containing a diffraction pattern to find the axes of symmetry of
	**origin_x: int, Position of origin accross the OpenCV mat
	**origin_y: int, Position of origin down the OpenCV mat
	**num_angles: int, Number of angles  to look for symmetry at
	**target_size: int, Downsampling factor will be the largest power of 2 that doesn't make the image smaller than this
	**Returns:
	**std::vector<float>, Highest Pearson normalised product moment correlation coefficient for a symmetry lins drawn through 
	**a known axis of symmetry
	*/
	std::vector<float> symmetry_axes(cv::Mat &amalg, int origin_x, int origin_y, size_t num_angles = 120, float target_size = 0);
}