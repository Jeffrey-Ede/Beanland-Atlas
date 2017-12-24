#pragma once

#include <defines.h>
#include <includes.h>

#include <template_matching.h>
#include <utility.h>

namespace ba
{
	/*Get the shift of a second image relative to the first
	**Inputs:
	**img1: cv::Mat &, One of the images
	**img1: cv::Mat &, The second image
	**use_frac: const float, Fraction of extracted rectangular region of interest to use when calculating the sum of squared 
	**differences to match it against another region
	**wisdom: const int, Expected relative position of the symmetry center 
	**grad_sym_use_frac: const float &, Threshold this portion of the gradient based symmetry values to constrain the regions of the
	**sum of squared differences when calculating the relative shift
	**Returns:
	**std::vector<float>, Pearson normalised product moment correlation coefficient and relative row and column shift of the second 
	**image, in that order
	*/
	std::vector<float> quantify_rel_shift(cv::Mat &img1, cv::Mat &img2, const float use_frac = QUANT_SYM_USE_FRAC, 
		const int wisdom = REL_SHIFT_WIS_NONE, const float grad_sym_use_frac = GRAD_SYM_USE);

	/*Get the largest rectangular portion of an image inside a surrounding black background
	**Inputs:
	**img: cv::Mat &, Input floating point image to extract the largest possible non-black region of
	**Returns:
	**cv::Rect, Largest non-black region
	*/
	cv::Rect biggest_not_black(cv::Mat &img);

	/*Decrease the size of the larger rectangular region of interest so that it is the same size as the smaller
	**Inputs:
	**roi1: cv::Rect, One of the regions of interest
	**roi2: cv::Rect, The other region of interest
	**Returns:
	**std::vector<cv::Rect>, Regions of interest that are the same size, in the same order as the input arguments
	*/
	std::vector<cv::Rect> same_size_rois(cv::Rect roi1, cv::Rect roi2);

	/*Rotates an image keeping the image the same size, embedded in a larger black rectangle
	**Inputs:
	**src: cv::Mat &, Image to rotate
	**angle: float, Angle to rotate the image (anticlockwise)
	**Returns:
	**cv::Mat, Rotated image
	*/
	cv::Mat rotate_CV(cv::Mat src, float angle);

	/*Order indices in order of increasing angle from the horizontal
	**Inputs:
	**angles: std::vector<float> &, Angles between the survey spots and a line drawn horizontally through the brightest spot
	**indices_orig: std::vector<int> &, Indices of the surveys to compare to identify the atlas symmetry
	*/
	void order_indices_by_angle(std::vector<float> &angles, std::vector<int> &indices);

	/*Estimate the global symmetry center of an image by finding the pixel that has the closest to zero total Scharr filtrate
	**when it it summed over equidistances in all directions from that point
	**Inputs:
	**img: cv::Mat &, Floating point greyscale OpenCV mat to find the global symmetry center of
	**wisdom: const int, Information about the symmetry that allows a gradients to be summed over a larger area
	**not_calc_val: const float, Value to set elements that do not have at least the minimum area to perform their calculation
	**use_frac: const float, Only global symmetry centers at least this fraction of the image's area will be considered
	**edges will be considered
	**Returns:
	**cv::Mat, Sums of gradients in the largest possible rectangular regions centred on each pixel
	*/
	cv::Mat est_global_sym(cv::Mat &img, const int wisdom = REL_SHIFT_WIS_NONE, const float not_calc_val = SYM_CENTER_NOT_CALC_VAL,
		const float use_frac = SYM_CENTER_USE_FRAC);
}