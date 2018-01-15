#pragma once

#include <includes.h>

#include <commensuration_utility.h>
#include <distortion_correction.h>
#include <matlab.h> //Matlab-specific includes

namespace ba
{
	//Approximate maximum number of data points to pass to the least squares fitting function
    #define MAX_NLLEASTSQ_DATA 1'000

	//Tolerance to use when least squares fitting a cubic Bezier
    #define LS_TOL 1e-4

	//Maximum number of iterations when least squares fitting a cubic Bezier
    #define LS_MAX_ITER 75

	//Custom data structure to hold overlapping circle region parameters
	struct circ_overlap_param {
		bool overlap; //Set to true if the circles overlap
		cv::Point2d center; //Center of the overlapping region
		std::vector<cv::Point2d> minima; //Positions of minimal distance from the center on each arc
		std::vector<cv::Point2d> maxima; //Positions of maximal distance from the center on each arc
		cv::Point P1, P2; //Positions of the circle centres
		std::vector<cv::Point> bounding_rect; //2 corners of the rectangle fully containing the extrema
	};
	typedef circ_overlap_param circ_overlap;

	/*Calculate the condenser lens profile using the overlapping regions of spots
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual floating point images that have been stereographically corrected to extract spots from
	**spot_pos: cv::Point2d, Position of located spot in the aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images
	**col_max: const int, Maximum column difference between spot positions
	**row_max: const int, Maximum row difference between spot positions
	**radius: const int, Radius about the spot locations to extract pixels from
	**Returns:
	**std::vector<cv::Mat>, Dynamical diffraction effect decoupled Bragg profile
	*/
	std::vector<cv::Mat> bragg_envelope(std::vector<cv::Mat> &mats, cv::Point2d &spot_pos, std::vector<std::vector<int>> &rel_pos,
		const int col_max, const int row_max, const int radius);

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
	**groups: std::vector<cv::Mat> &, Output preprocessed Bragg peaks, ready for dark field decoupled profile extraction
	**group_pos: std::vector<cv::Point> &, Positions of top left corners of circles' bounding squares
	**is_in_img: std::vector<bool> &, Output to mark true when the spot is in the image so that indices can be grouped
	*/
	void grouping_preproc(std::vector<cv::Mat> &mats, std::vector<std::vector<int>> &grouped_idx, cv::Point2d &spot_pos, 
		std::vector<std::vector<int>> &rel_pos, const int &col_max, const int &row_max, const int &radius,
		const int &diam, std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, std::vector<bool> &is_in_img);

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

	/*Calculates the center of and the 2 points closest to and furthest away from the center of the overlapping regions 
	**of 2 spots
	**Inputs:
	**spot_pos: cv::Point2d, Position of located spot in the aligned diffraction pattern
	**rel_pos: std::vector<std::vector<int>> &, Relative positions of images to first image
	**col_max: const int &, Maximum column difference between spot positions
	**row_max: const int &, Maximum row difference between spot positions
	**radius: const int, Radius of the spots
	**m: const int, Index of one of the images to compate
	**n: const int, Index of the other image to compate
	**cols: const int, Number of columns in the image
	**rows: const int, Number of rows in the image
	**Returns:
	**circ_overlap, Structure describing the region where the circles overlap
	*/
	circ_overlap get_overlap(cv::Point2d &spot_pos, std::vector<std::vector<int>> &rel_pos,	const int &col_max,
		const int &row_max, const int radius, const int m, const int n, const int cols, const int rows);

	/*Boolean indicating whether or not a point lies on an image
	**Inputs
	**img: cv::Mat &, Image to check if the point is on
	**point: cv::Point, Point to be checked
	**Returns:
	**bool, If true, the point is on the image
	*/
	bool on_img(cv::Mat &img, cv::Point point);

	/*Boolean indicating whether or not a point lies on an image
	**Inputs
	**cols: const int, Number of columns in the image
	**rows: const int, Number of rows in the image
	**point: cv::Point, Point to be checked
	**Returns:
	**bool, If true, the point is on the image
	*/
	bool on_img(cv::Point point, const int cols, const int rows);

	/*Generate a mask where the overlapping region between 2 circles is marked by ones
	**Inputs:
	**P1: cv::Point2d, Center of one of the circles
	**r1: const int &, Radius of one of the circles
	**P2: cv::Point2d, Center of the other circle
	**r2: const int &, Radius of the other circle
	**cols: const int, Number of columns in the mask
	**rows: const int, Number of rows in the mask
	**val: const byte, value to set the mask elements. Defaults to 1
	**Returns:
	**cv::Mat, 8 bit image where the overlapping region is marked with ones
	*/
	cv::Mat gen_circ_overlap_mask(cv::Point2d P1, const int r1, cv::Point2d P2, const int r2, const int cols,
		const int rows, const byte val = 1);

	/*Use the unique overlaps of each group of spots to determine the condenser lens profile
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
	cv::Mat get_bragg_envelope(std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
		std::vector<std::vector<int>> &rel_pos, std::vector<std::vector<int>> &grouped_idx, std::vector<bool> &is_in_img,
		const int radius, const int col_max, const int row_max, const int cols, const int rows, const int diam);

	/*Overload the << operator to print circ_overlap structures
	**Inputs:
	**os: std::ostream &, Reference to the operating system,
	**co: const circ_overlap &, Overlapping circle region structure
	**Returns:
	**std::ostream &, Reference to the operating system
	*/
	std::ostream & operator<<(std::ostream & os, const circ_overlap & co);

	/*Boolean indicating whether or not a point lies on an image
	**Inputs
	**img: cv::Mat &, Image to check if the point is on
	**col: const int, Column of point to be checked
	**row: const int, Row of point to be checked
	**Returns:
	**bool, If true, the point is on the image
	*/
	bool on_img(cv::Mat &img, const int col, const int row);
}