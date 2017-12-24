#pragma once

#include <defines.h>
#include <includes.h>

namespace ba
{
	struct align_avg_mats {
		cv::Mat acc;
		cv::Mat num_overlap;
	};
	/*Align the diffraction patterns using their known relative positions and average over the aligned px
	**mats: std::vector<cv::Mat> &, Diffraction patterns to average over the aligned pixels of
	**redined_pos: std::vector<std::vector<int>> &, Relative positions of the images to the first image
	**Return:
	**struct align_avg_mats, The first OpenCV mat is the average of the aligned diffraction patterns, the 2nd is the number of OpenCV mats
	**that contributed to each pixel
	*/
	struct align_avg_mats align_and_avg(std::vector<cv::Mat> &mats, std::vector<std::vector<int>> &refined_pos);

	/*Refine the relative positions of the images using all the known relative positions
	**Inputs:
	**positions: std::vector<cv::Vec2f>, Relative image positions and their weightings
	**Return:
	**std::vector<std::vector<int>>, Relative positions of the images, including the first image, to the first image in the same order as the
	**images in the image stack
	*/
	std::vector<std::vector<int>> refine_rel_pos(std::vector<std::array<float, 5>> &positions);
}