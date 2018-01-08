#pragma once

#include <includes.h>

#include <utility.h>

namespace ba
{
	/**Refine positions of mirror lines. 
	**Inputs:
	**amalg: cv::Mat, Diffraction pattern to refine symmetry lines on
	**max_pos: std::vector<int>, Array indices corresponding to intensity maxima
	**num_angles: max_pos indices are converted to angles via angle_i = max_pos[i]*PI/num_angles
	**Returns:
	**std::vector<cv::Vec3f>, Refined origin position and angle of each mirror line, in the same order as the input maxima
	*/
	std::vector<cv::Vec3f> refine_mir_pos(cv::Mat amalg, std::vector<int> max_pos, size_t num_angles, int origin_x,
		int origin_y, int range);

	/*Calculate the mean estimate for the centre of symmetry
	**Input:
	**refined_pos: std::vector<cv::Vec3f>, 2 ordinates and the inclination of each line
	**Returns:
	**cv::Vec2f, The average 2 ordinates of the centre of symmetry
	*/
	cv::Vec2f avg_origin(std::vector<cv::Vec3f> lines);

	/*Calculate all unique possible points of intersection, add them up and then divide by their number to get the average
	**Input:
	**refined_pos: std::vector<cv::Vec3f>, 2 ordinates and the inclination of each line
	**Returns:
	**cv::Vec2f, The average 2 ordinates of intersection
	*/
	cv::Vec2f average_intersection(std::vector<cv::Vec3f> lines);
}