#pragma once

#include <includes.h>

#include <identify_symmetry.h>

namespace ba
{
	/*Get the position of the central spot's symmetry center, whatever it's symmetry may be
	**Inputs:
	**img: cv::Mat &, Rotated central survey so that the symmetry center is easier to extract
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**num_spots: const int, Number of equidistant spots on the aligned diffraction pattern
	**Returns:
	**cv::Point, Position of the symmetry center
	*/
	cv::Point2f center_sym_pos(cv::Mat &img, std::vector<float> &angles, const int num_spots);
}