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
    #define MIN_OVERLAP_PX_NUM 50

	//Maximum expected error of relative positions found by aligning the images in px
	#define DISTORT_MAX_REL_POS_ERR 2.0

	//Number of pixels per keypoint that is looked for when determining the relative positions of overlapping pixels
	#define OVERLAP_REL_POS_PX_PER_KEYPOINT 5
	
	//Scale to look for features on with the ORB detector when 
	#define OVERLAP_REL_POS_SCALE 2 //This can be changed to be dynamically calculated at a later time

	//Minimum number of pixels needed for a Pearson correlation to be valid
    #define MIN_OVERLAP_PX_REG 5

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
	*/
	void get_aberrating_fields( std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
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

	/*Create a rectangle containing the non-zero pixels in a mask that can be used to crop them from it
	**Inputs:
	**mask: cv::Mat &, 8-bit mask image
	**Returns:
	**cv::Rect, Rectangle containing the non-zero pixels in a mask
	*/
	cv::Rect get_non_zero_mask_px_bounds(cv::Mat &mask);

	/*Get the ratio of the intensity profiles where 2 spots overlap
	**Inputs:
	**mask: cv::Mat &, 8-bit mask that's non-zero values indicate where the circles overlap
	**c1: cv::Mat &, One of the circles
	**c2; cv::Mat &, The other circles
	**p1: cv::Point &, Top left point of a square containing the first circle on the detector
	**p2: cv::Point &, Top left point of a square containing the second circle on the detector
	**max_shift: const float, Maximum shift to consider
	**incr_shift: const float, Steps to adjust the shift by
	**ratios: cv::Mat &, Output cropped image containing the overlap region containing the ratios of the intensity profiles.
	**Ratios larger than 1.0 are reciprocated
	**ratios_mask: cv::Mat &, Output mask indicating the ratios pixels containing ratios
	**rel_pos: cv::Point &, The position of the top left corner of ratios and ratios_mask in the c1 mat
	*/
	void get_overlap_ratios(cv::Mat &mask, cv::Mat &c1, cv::Mat &c2, cv::Point &p1, cv::Point &p2, 
		cv::Mat &ratios, cv::Mat &ratios_mask, cv::Point &rel_pos);

	/*Get the relative positions of overlapping regions of spots
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
	*/
	void overlap_rel_pos(std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
		std::vector<std::vector<int>> &rel_pos, std::vector<std::vector<int>> &grouped_idx, std::vector<bool> &is_in_img,
		const int radius, const int col_max, const int row_max, const int cols, const int rows, const int diam);

	/*Relative position of one spot overlapping with another
	**Inputs:
	**mask: cv::Mat &, 8-bit mask that's non-zero values indicate where the circles overlap
	**c1: cv::Mat &, One of the circles
	**c2; cv::Mat &, The other circles
	**p1: cv::Point &, Top left point of a square containing the first circle on the detector
	**p2: cv::Point &, Top left point of a square containing the second circle on the detector
	**max_shift: const float, Maximum shift to consider
	**incr_shift: const float, Steps to adjust the shift by
	**nnz: const int, Number of non-zero pixels in the mask. The number of features looked for will be based on this
	**shift: cv::Vec2f &, Output how much the second image needs to be shifted to align it
	*/
	void get_overlap_rel_pos(cv::Mat &mask, cv::Mat &c1, cv::Mat &c2, cv::Point &p1, cv::Point &p2,
		cv::Ptr<cv::ORB> &orb, const int nnz, cv::Vec2f &shift);

	/*Use Pearson product moment correlation coefficients to determine the relative positions of 2 overlapping regions
	**the idea
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
	*/
	void pearson_overlap_register(std::vector<cv::Mat> &groups, std::vector<cv::Point> &group_pos, cv::Point2d &spot_pos,
		std::vector<std::vector<int>> &rel_pos, std::vector<std::vector<int>> &grouped_idx, std::vector<bool> &is_in_img,
		const int radius, const int col_max, const int row_max, const int cols, const int rows, const int diam);

	/*Relative position of one spot overlapping with another using Pearson product moment correlation to register them
	**Inputs:
	**mask: cv::Mat &, 8-bit mask that's non-zero values indicate where the circles overlap
	**c1: cv::Mat &, One of the circles
	**c2; cv::Mat &, The other circles
	**p1: cv::Point &, Top left point of a square containing the first circle on the detector
	**p2: cv::Point &, Top left point of a square containing the second circle on the detector
	**min_px: const int, Minimum number of pixels in a registration
	**max_shift: cv::Vec2i &, Maximum displacent of the images
	**shift: cv::Vec3f &, Output how much the second image needs to be shifted to align it and the Pearson coefficient
	*/
	void get_pearson_overlap_register(cv::Mat &mask, cv::Mat &c1, cv::Mat &c2, cv::Point &p1, cv::Point &p2, const int min_px,
		cv::Vec2i &max_shift, cv::Vec3f &shift);

	/*Use Pearson product moment correlation to register 2 masked images of the same size
	**Inputs:
	**img1: cv::Mat &, One of the images to register
	**img2: cv::Mat &, Image being registered against the other
	**mask1: cv::Mat &, Indicates which pixels in the first image can be used
	**mask2: cv::Mat &, Indicates which pixels in the second image can be used
	**shift: cv::Vec3f &, Output registration of the second image relative to the first and Pearson coefficient
	**min_px: const int, The minimum number of pixels that the matched region must contain
	**max_shift: cv::Vec2i &, Maximum displacent of the images
	*/
	void masked_pearson_reg(cv::Mat &img1, cv::Mat &img2, cv::Mat &mask1, cv::Mat &mask2, cv::Vec3f &shift,
		const int min_px, cv::Vec2i &max_shift);

	/*Calculate Pearson's product moment correlation coefficent from 2 32-bit images at marked locations
	**Inputs:
	**img1: cv::Mat &, One of the images
	**img2: cv::Mat &, The other image
	**mask: cv::Mat &, 8-bit mask that's non-zero values indicate which pixels to use
	**Returns:
	**double, Pearson product moment correlation coefficient between the images
	*/
	double masked_pearson_corr(cv::Mat &img1, cv::Mat &img2, cv::Mat &mask);

	/*Use the Fisher transform to get a confidence interval for Pearson's coefficient
	**Inputs:
	**rho: const float, Pearson normalised product moment correlation coefficient
	**num: const int, Number of elements in the sample
	**confidence: Confidence to find the interval for e.g. 0.95
	**matlabPtr: std::unique_ptr<matlab::engine::MATLABEngine> &, Optional pointer to a MATLAB engine
	**Returns:
	**Vec2f, Confidence interval
	*/
	cv::Vec2f fisher_pearson_confid(const float rho, const int num, const float confidence,
		std::unique_ptr<matlab::engine::MATLABEngine> &matlabPtr);

	/*Calculate Pearson's product moment correlation coefficent from 2 32-bit images at marked locations and the
	**probability
	**Inputs:
	**img1: cv::Mat, One of the images
	**img2: cv::Mat, The other image
	**mask: cv::Mat, 8-bit mask that's non-zero values indicate which pixels to use
	**matlabPtr: std::unique_ptr<matlab::engine::MATLABEngine> &, Pointer to a MATLAB engine
	**Returns:
	**cv::Vec2f, Pearson product moment correlation coefficient between the images and the confidence
	*/
	cv::Vec2f masked_pearson_corr_with_confid(cv::Mat img1, cv::Mat img2, cv::Mat mask, 
		std::unique_ptr<matlab::engine::MATLABEngine> &matlabPtr);

	/*Take the Kronecker produce of a matrix with a patter matrix
	**A: const cv::Mat &, The matrix
	**B: const cv::Mat &, The pattern
	**K: cv::Mat &, Output Kronecker product
	*/
	void kron( const cv::Mat &A, const cv::Mat &B, cv::Mat &K );

	/*Dilate an image by taking the Gaussian average of the non-zero neigbouring pixels
	**Inputs
	**img: const cv::Mat &, Image to dilate by taking the Gaussian averages of neighbouring pixels
	**mask: const cv::Mat &, Indicates the portion of the image to dilate using the neighbouring mask pixels
	**dst: cv::Mat &, Output dilated image
	*/
	void dilate_avg(const cv::Mat &img, const cv::Mat &mask, cv::Mat &dst);
}