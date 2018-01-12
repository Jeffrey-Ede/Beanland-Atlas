#pragma once

#include <includes.h>

#include <commensuration_ellipses.h>
#include <utility.hpp>

namespace ba
{
	//Default stopping criteria when using k-means clustering to identify the high Scharr filtrate
    #define HIGH_SCHARR_MAX_ITER 1000
	#define HIGH_SCHARR_TERM_CRIT_EPS 1e-3

	//High Scharr filtrate post k-means clustering erosion and dilation
    #define HIGH_SCHARR_NUM_EROSIONS 1
    #define HIGH_SCHARR_NUM_DILATIONS 2

	//Default stopping criteria when using k-means clustering to mark specific colours in an image
    #define KMEANS_MASK_MAX_ITER 10'000
    #define KMEANS_MASK_TERM_CRIT_EPS 1e-5

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
	*/
	void high_Scharr_edges(cv::Mat &img, cv::Mat &dst, const int num_erodes = HIGH_SCHARR_NUM_EROSIONS, 
		const int num_dilates = HIGH_SCHARR_NUM_DILATIONS, const int max_iter = HIGH_SCHARR_MAX_ITER,
		const float term_crit_eps = HIGH_SCHARR_TERM_CRIT_EPS, const int num_clusters = 2);

	/*Mark intensity groups in a single channel image using k-means clustering
	**Inputs:
	**img: cv::Mat &, The single-channel 32-bit image to posterise
	**dst: cv::Mat &, 8-bit output image
	**num_clusters: const int, Number of groups to split the intensities into
	**clusters_to_use: std::vector<int>, Colour quantisation levels to use. Clusters can be indicated from low to high using 
	**numbers starting from 0, going up. Clusters can be indicated from high to low using numbers starting from -1, going down
	**mask: cv::Mat &, Optional 8-bit image whose non-zero values are to be k-means clustered. Zero values on the mask will be zeroes
	**in the output image.
	**max_iter: const int, Maximum number of iterations to perform during k-means clustering
	**term_crit_eps: const float, Difference between successive k-means clustering iterations for the algorithm to stop
	**val: const byte, Value to mark values to use on the output image. Defaults to 1
	*/
	void kmeans_mask(cv::Mat &img, cv::Mat &dst, const int num_clusters, std::vector<int> clusters_to_use, cv::Mat &mask = cv::Mat(), 
		const int max_iter = KMEANS_MASK_MAX_ITER, const float term_crit_eps = KMEANS_MASK_TERM_CRIT_EPS, const byte val = 1);

	/*Mark a single intensity group in a single channel image using k-means clustering. This function passes a vector indicating the
	**single intensity group to the variant of the function that accepts multiple intensity groups
	**Inputs:
	**img: cv::Mat &, The single-channel 32-bit image to posterise
	**dst: cv::Mat &, 8-bit output image
	**num_clusters: const int, Number of groups to split the intensities into
	**cluster_to_use: const int, Colour quantisation level to use. Clusters can be indicated from low to high using 
	**numbers starting from 0, going up. Clusters can be indicated from high to low using numbers starting from -1, going down
	**mask: cv::Mat &, Optional 8-bit image whose non-zero values are to be k-means clustered. Zero values on the mask will be zeroes
	**in the output image
	**max_iter: const int, Maximum number of iterations to perform during k-means clustering
	**term_crit_eps: const float, Difference between successive k-means clustering iterations for the algorithm to stop
	**val: const byte, Value to mark values to use on the output image. Defaults to 1
	*/
	void kmeans_mask(cv::Mat &img, cv::Mat &dst, const int num_clusters, const int cluster_to_use, cv::Mat &mask = cv::Mat(), 
		const int max_iter = KMEANS_MASK_MAX_ITER, const float term_crit_eps = KMEANS_MASK_TERM_CRIT_EPS, const byte val = 1);
}