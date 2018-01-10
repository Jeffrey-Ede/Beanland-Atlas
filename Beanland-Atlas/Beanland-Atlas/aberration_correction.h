#pragma once

#include <includes.h>

#include <commensuration_ellipses.h>

namespace ba
{
	//Default stopping criteria when using k-means clustering to identify the high Scharr filtrate
    #define HIGH_SCHARR_MAX_ITER 1000
	#define HIGH_SCHARR_TERM_CRIT_EPS 0.001

	//High Scharr filtrate post k-means clustering erosion and dilation
    #define HIGH_SCHARR_NUM_EROSIONS 1
    #define HIGH_SCHARR_NUM_DILATIONS 2

	/*Calculate the blurriness of an image using the variance of it's Laplacian filtrate. A 3 x 3 
	**{0, -1, 0; -1, 4, -1; 0 -1 0} Laplacian kernel is used. A two pass algorithm is used to avoid catastrophic
	**cancellation
	**Inputs:
	**img: cv::Mat &, 32-bit image to measure the blurriness of
	**mask: cv::Mat &, 8-bit mask where non-zero values indicate values of the Laplacian filtrate to use
	**Returns:
	**float, Variance of the Laplacian filtrate
	*/
	float var_laplacian(cv::Mat &img, cv::Mat &mask);

	/*Use cluster analysis to label the high intensity pixels and produce a mask where their positions are marked. These positions
	**are indicative of edges
	**Inputs:
	**img: cv::Mat &, Image to find the high Scharr filtrate values of
	**dst: cv::Mat &, Output 8-bit image where the high Scharr filtrate values are marked
	**num_erodes: const int, Number of times to erode the mask to remove stray fluctating pixels
	**num_dilates: const int, Number of times to dilate the image after erosion
	**max_iter: const int, Maximum number of iterations to perform during k-means clustering
	**term_crit_eps: const float, Difference between successive k-means clustering iterations for the algorithm to stop
	**num_clusters: const int, Number of clusters to split data into to select the highest of. Defaults to 2.
	**val: const byte, Value to mark the high Scharr filtrate values with. Defaults to 1.
	*/
	void high_Scharr_edges(cv::Mat &img, cv::Mat &dst, const int num_erodes = HIGH_SCHARR_NUM_EROSIONS, 
		const int num_dilates = HIGH_SCHARR_NUM_DILATIONS, const int max_iter = HIGH_SCHARR_MAX_ITER,
		const float term_crit_eps = HIGH_SCHARR_TERM_CRIT_EPS, const int num_clusters = 2, const byte val = 1);
}