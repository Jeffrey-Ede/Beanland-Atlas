#pragma once

#include <defines.h>
#include <includes.h>

#include <get_spot_positions.h>

namespace ba
{
	/*Rotate an image in into the image plane
	**Input:
	**img: cv::Mat &, Image to rotate
	**angle_horiz: const float &, Angle to rotate into plane from horizontal axis
	**angle_vert: const float &, Angle to rotate into plane from vertical axis
	**Returns:
	**cv::Mat, Image after rotation into the image plane
	*/
	cv::Mat in_plane_rotate(cv::Mat &img, const float &angle_horiz, const float &angle_vert);

	/*Estimate the radius of curvature of the sample-to-detector sphere
	**Input:
	**spot_pos: std::vector<cv::Point> &, Positions of spots in the aligned images' average px values diffraction pattern
	**xcorr: cv::Mat &, Cross correlation spectrum that's peaks reveal the positions of the spots
	**discard_outer: const int, Spots withing this distance of the boundary are discarded
	**col: const int, column of spot centre
	**row: const int, row of spot centre
	**centroid_size: const int, Size of centroid to take about estimated spot positions to refine them
	**Returns:
	**cv::Vec2f, Initial estimate of the sample-to-detector sphere radius of curvature and average direction, respectively
	*/
	cv::Vec2f get_sample_to_detector_sphere(std::vector<cv::Point> &spot_pos, cv::Mat &xcorr, const int discard_outer, const int col,
		const int row, const int centroid_size = SAMP_TO_DETECT_CENTROID);

	/*Refine the estimated positions of spots to sub-pixel accuracy by calculating centroids around their estimated
	**positions based on an image that weights the likelihood of particular pixels representing spots. It is assumed that the spots
	**are at least have the centroid weighting's width and height away from the weighting spectrum's borders
	**Input:
	**spot_pos: std::vector<cv::Point> &, Positions of spots in the aligned images' average px values diffraction pattern
	**xcorr: cv::Mat &, Cross correlation spectrum that's peaks reveal the positions of the spots
	**col: const int, column of spot centre
	**row: const int, row of spot centre
	**centroid_size: const int, Size of centroid to take about estimated spot positions to refine them
	**Returns:
	**std::vector<cv::Point2f>, Sub-pixely accurate spot positions
	*/
	std::vector<cv::Point2f> refine_spot_pos(std::vector<cv::Point> &spot_pos, cv::Mat &xcorr, const int col, const int row,
		const int centroid_size);

	/*Creat a Beanland atlas survey by using pixels from the nearest spot to each survey position
	**Input:
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the images
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**Returns:
	**cv::Mat, Survey made from the pixels of the nearest spots to each position
	*/
	cv::Mat create_small_spot_survey(std::vector<std::vector<int>> &rel_pos, std::vector<cv::Point> &spot_pos, 
		const int &max_radius);

	/*Commensurate the images using homographic perspective warps, homomorphic warps and intensity rescaling
	**Input:
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of the spots
	**grouped_idx: std::vector<std::vector<int>> &, Spots where they are grouped if they are consecutively in the same position
	**max_dist: const float &, The maximum distance between 2 instances of a spot for the overlap between them to be considered
	**Returns:
	**std::vector<std::vector<std::vector<int>>>, For each group of consecutive spots, for each of the spots it overlaps with, 
	**a vector containing: index 0 - the consecutive group being overlapped, index 1 - the relative column position of the consecutive
	**group that is overlapping relative to the the spot,  index 2 - the relative row position of the consecutive group that is overlapping 
	**relative to the the spot. The nesting is in that order.
	*/
	std::vector<std::vector<std::vector<int>>> get_spot_overlaps(std::vector<std::vector<int>> &rel_pos,
		std::vector<std::vector<int>> &grouped_idx, const float &max_dist);

	/*Find the differences between two overlapping spots where they overlap. Also output a mask indicating which pixels correspond to overlap
	**in case some of the differences are zero
	**Input:
	**img1: cv::Mat &, Image containing a spot
	**img2: cv::Mat &, A second image containing a spot
	**origin1: cv::Point &, Column and row of the spot centre in the first image, respectively
	**origin2: cv::Point &, Column and row of the spot centre in the second image, respectively
	**dx: const int &, Relative column of the second spot to the first
	**dy: const int &, Relative row of the second spot to the first
	**circ_mask: cv::Mat &, Mask indicating the spot pixels
	**diff: cv::Mat &, Difference between the 2 matrices when the first is subtracted from the second
	**mask: cv::Mat &, Mask indicating which pixels are differences of the overlapping regions
	*/
	void get_diff_overlap(cv::Mat &img1, cv::Mat &img2, cv::Point &origin1, cv::Point &origin2, const int &dx, const int &dy,
		cv::Mat &circ_mask, cv::Mat &diff, cv::Mat &mask);
}