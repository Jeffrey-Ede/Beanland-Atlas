#pragma once

#include <includes.h>

#include <commensuration.h>
#include <commensuration_utility.h>
#include <matlab.h> //Matlab-specific includes
#include <utility.hpp>

namespace ba
{
	//Minimum number of overlapping pixels for the affine transform of an overlapping region to be used to estimate the 
	//distortions
    #define MIN_OVERLAP_PX_NUM 10

	/*Use spot overlaps to determine the distortion field
	**Inputs:
	**groups: std::vector<cv::Mat> &, Preprocessed Bragg peaks, ready for dark field decoupled profile extraction
	**group_pos: std::vector<cv::Point> &, Positions of top left corners of circles' bounding squares
	**spot_pos: cv::Point2d, Position of located spot in the aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**grouped_idx: std::vector<std::vector<int>> &, Groups of consecutive image indices where the spots are all in the same position
	**is_in_img: std::vector<bool> &, True when the spot is in the image so that indices can be grouped
	**radius: const int, Radius of the spots
	**col_max: const int, Maximum column difference between spot positions
	**row_max: const int, Maximum row difference between spot positions
	**cols: const int, Number of columns in the image
	**rows: const int, Number of rows in the image
	**Returns:
	**cv::Mat, Condenser lens profile
	*/
	cv::Mat get_distortion_field( std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
		std::vector<std::vector<int>> &rel_pos, std::vector<std::vector<int>> &grouped_idx, std::vector<bool> &is_in_img,
		const int radius, const int col_max, const int row_max, const int cols, const int rows, const int diam );

	/*Create a matrix indicating where a spot overlaps with an affinely transformed spot
	**Inputs:
	**warp_mat: cv::Mat &, Affine warp matrix
	**mask: cv::Mat &, Mask indicating the spot overlap in the absence of affine transformation
	**P1: cv::Point2d, Center of one of the circles
	**r1: const int &, Radius of one of the circles
	**P2: cv::Point2d, Center of the other circle
	**r2: const int &, Radius of the other circle
	**cols: const int, Number of columns in the mask
	**rows: const int, Number of rows in the mask
	**val: const byte, value to set the mask elements. Defaults to 255
	**Returns:
	**cv::Mat, 8 bit image where the overlapping region is marked with ones
	*/
	cv::Mat get_affine_overlap_mask(cv::Mat &warp_mat, cv::Mat &mask, cv::Point2d P1, const int r1, cv::Point2d P2,
		const int r2, const int cols, const int rows, const byte val = 255);
}