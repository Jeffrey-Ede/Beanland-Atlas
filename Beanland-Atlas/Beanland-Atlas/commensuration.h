#pragma once

#include <defines.h>
#include <includes.h>

#include <commensuration_utility.h>

namespace ba
{
	/*Commensurate the individual images so that the Beanland atlas can be constructed
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual floating point images to extract spots from
	**spot_pos: cv::Point2d, Positions of located spot in the aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**col_max: int, Maximum column difference between spot positions
	**row_max: int, Maximum row difference between spot positions
	**radius: const int, Radius about the spot locations to extract pixels from
	**Returns:
	**std::vector<cv::Mat>, Dynamical diffraction effect decoupled Bragg profile
	*/
	std::vector<cv::Mat> condenser_profile(std::vector<cv::Mat> &mats, cv::Point2d &spot_pos, std::vector<std::vector<int>> &rel_pos,
		int col_max, int row_max, int radius);

	/*Identify groups of consecutive spots that all have the same position
	**Input:
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots in the image stack
	**Returns:
	**std::vector<std::vector<int>>, Groups of spots with the same location
	*/
	std::vector<std::vector<int>> consec_same_pos_spots(std::vector<std::vector<int>> &rel_pos);

	/*Extract a Bragg peak from an image stack, averaging the spots that are in the same position in consecutive images
	**Inputs:
	**mats: std::vector<cv::Mat> &, Images to extract the spots from
	**grouped_idx: std::vector<std::vector<int>> &, Groups of consecutive image indices where the spots are all in the same position
	**spot_pos: cv::Point2d &, Position of the spot on the aligned images average px values diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots in the input images to the first image
	**col_max: const int &, Maximum column difference between spot positions
	**row_max: const int &, Maximum row difference between spot positions
	**radius: const int &, Radius of the spot
	**diam: const int &, Diameter of the spot
	**gauss_size: const int &, Size of the Gaussian blurring kernel applied during the last preprocessing step to remove unwanted noise
	**Returns:
	**std::vector<cv::Mat>, Preprocessed Bragg peaks, ready for dark field decoupled profile extraction
	*/
	std::vector<cv::Mat> grouping_preproc(std::vector<cv::Mat> &mats, std::vector<std::vector<int>> &grouped_idx, 
		cv::Point2d &spot_pos, std::vector<std::vector<int>> &rel_pos, const int &col_max, const int &row_max, const int &radius,
		const int &diam, const int &gauss_size);

	/*Extracts a circle of data from an OpenCV mat and accumulates it in another mat. It is assumed that the dimensions specified for
	**the accumulator will allow the full circle-sized extraction to be accumulated
	**Inputs:
	**mat: cv::Mat &, Reference to a floating point OpenCV mat to extract the data from
	**col: const int &, column of circle origin
	**row: const int &, row of circle origin
	**rad: const int &, radius of the circle to extract data from
	**acc: cv::Mat &, Floating point OpenCV mat to accumulate the data in
	**acc_col: const int &, Column of the accumulator mat to position the circle at
	**acc_row: const int &, Row of the accumulator mat to position the circle at
	*/
	void accumulate_circle(cv::Mat &mat, const int &col, const int &row, const int &rad, cv::Mat &acc, const int &acc_col,
		const int &acc_row);

	/*Commensurate the images using homographic perspective warps, homomorphic warps and intensity rescaling
	**Input:
	**commensuration: std::vector<cv::Mat>> &, Preprocessed Bragg peaks. Consecutive Bragg peaks in the same position have been averaged
	**and they have been Gaussian blurred to remove unwanted high frequency noise
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots
	**grouped_idx: std::vector<std::vector<int>> &, Spots where they are grouped if they are consecutively in the same position
	**circ_mask: cv::Mat &, Mask indicating the spot pixels
	**max_dst: const float &, The maximum distance between 2 instances of a spot for the overlap between them to be considered
	*/
	void commensuration_perspective_warp(std::vector<cv::Mat> &commensuration, std::vector<std::vector<int>> &rel_pos, cv::Mat &circ_mask, 
		std::vector<std::vector<int>> &grouped_idx, const float &max_dist);

	/*Calculate an initial estimate for the dark field decoupled Bragg profile using the preprocessed Bragg peaks. This function is redundant.
	**It remains in case I need to generate data from it for my thesis, etc. in the future
	**Input:
	**blur_not_consec: std::vector<cv::Mat>> &, Preprocessed Bragg peaks. Consecutive Bragg peaks in the same position have been averaged
	**and they have been Gaussian blurred to remove unwanted high frequency noise
	**circ_mask: cv::Mat &, Mask indicating the spot pixels
	**Returns:
	**cv::Mat, Dark field decouple Bragg profile of the accumulation
	*/
	cv::Mat get_acc_bragg_profile(std::vector<cv::Mat> &blur_not_consec, cv::Mat &circ_mask);

	//Custom data structure to hold overlapping circle region parameters
	struct circ_overlap_param {
		bool overlap; //Set to true if the circles overlap
		cv::Point2d center; //Center of the overlapping region
		std::vector<cv::Point2d> minima; //Positions of minimal distance from the center on each arc
		std::vector<cv::Point2d> maxima; //Positions of maximal distance from the center on each arc
		cv::Point P1, P2; //Positions of the circle centres
	};
	typedef circ_overlap_param circ_overlap;
}