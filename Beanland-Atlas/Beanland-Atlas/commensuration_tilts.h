#pragma once

#include <includes.h>
#include <defines.h>

namespace ba
{
	/*Get ellipses describing each spot from their Scharr filtrates. Ellipses are checked using heuristic arguments:
	**ellipse shapes vary smoothly with time and ellipse shaps must be compatible with a projection of an array of
	**circles onto a flat detector
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual images to extract spots from
	**spot_pos: std::vector<cv::Point> &, Positions of located spots in aligned diffraction pattern
	**ellipses: std::vector<std::vector<std::vector<cv::Point>>> &, Positions of the minima and maximal extensions of spot ellipses.
	**The ellipses are decribed in terms of 4 points, clockwise from the top left as it makes it easy to use them to perform 
	**homomorphic warps. The nesting is each image, each spot in the order of their positions in the positions vector, set of
	**4 points (1 is extra) desctribing the ellipse, in that order
	**acc: cv::Mat &, Average of the aligned diffraction patterns
	*/
	void get_spot_ellipses(std::vector<cv::Mat> &mats, std::vector<cv::Point> &spot_pos, cv::Mat &acc, 
		std::vector<std::vector<std::vector<cv::Point>>> &ellipses);

	/*Amplitude of image's Scharr filtrate
	**Inputs:
	**img: cv::Mat &, Floating point image to get the Scharr filtrate of
	**scharr_amp: cv::Mat &, OpenCV mat to output the amplitude of the Scharr filtrate to
	*/
	void scharr_amp(cv::Mat &img, cv::Mat &scharr_amp);

	/*Use the accumulator spots to estimate the size of the ellipses
	**Inputs:
	**acc: cv::Mat &, Average of the aligned diffraction patterns
	**spot_pos: std::vector<cv::Point>, Positions of located spots in aligned diffraction pattern
	**ellipses: std::vector<std::vector<std::vector<cv::Point>>> &, Positions of the minima and maximal extensions of spot ellipses.
	**The ellipses are decribed in terms of 4 points, clockwise from the top left as it makes it easy to use them to perform 
	**homomorphic warps. The nesting is each spot in the order of their positions in the positions vector, set of 4 points
	**(1 is extra) desctribing the ellipse, in that order
	**thicknesses: std::vector<std::vector<float>> &, Thicknesses of the elliptical Scharr filtrates at each of the ellipse points. 
	**These can be used to estimate the range of ellipse parameters contributing to the accumulator ellipses
	*/
	void est_ellipse_sizes(cv::Mat &acc, std::vector<cv::Point> spot_pos, std::vector<std::vector<cv::Point>> &ellipses, 
		std::vector<std::vector<float>> &thicknesses);
}