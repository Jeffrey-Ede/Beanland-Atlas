#pragma once

#include <defines.h>
#include <includes.h>

namespace ba
{
	/*Align the diffraction patterns using their known relative positions and average over the aligned px
	**mats: std::vector<cv::Mat> &, Diffraction patterns to average over the aligned pixels of
	**redined_pos: std::vector<std::vector<int>> &, Relative positions of the images
	**acc: cv::Mat &, Average of the aligned diffraction patterns
	**num_overlap: cv::Mat &, Number of images that contributed to each pixel
	*/
	void align_and_avg(std::vector<cv::Mat> &mats, std::vector<std::vector<int>> &refined_pos, cv::Mat &acc, 
		cv::Mat &num_overlap);

	/*Refine the relative positions of the images using all the known relative positions
	**Inputs:
	**positions: std::vector<cv::Vec2f>, Relative image positions and their weightings
	**Return:
	**std::vector<std::vector<int>>, Relative positions of the images, including the first image, to the first image in the same order as the
	**images in the image stack
	*/
	std::vector<std::vector<int>> refine_rel_pos(std::vector<std::array<float, 5>> &positions);
}