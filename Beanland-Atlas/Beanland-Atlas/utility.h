#pragma once

#include <defines.h>
#include <includes.h>

#include "utility.hpp"

namespace ba
{
	/*Calculate Pearson normalised product moment correlation coefficient between 2 vectors of floats
	**Inputs:
	**vect1: std::vector<float>, One of the datasets to use in the calculation
	**vect2: std::vector<float>, The other dataset to use in the calculation
	**NUM_THREADS: const int, The number of threads to use for OpenMP CPU acceleration
	**Return:
	**float, Pearson normalised product moment correlation coefficient between the 2 datasets
	*/
	float pearson_corr(std::vector<float> vect1, std::vector<float> vect2, const int NUM_THREADS);

	/*Utility function that builds a named kernel from source code. Will print errors if there are problems compiling it
	**Inputs:
	**kernel_sourceFile: const char*, File containing kernel source code
	**kernel_name: const char*, Name of kernel to be built
	**af_context: cl_context, Context to create kernel in
	**af_device_id: cl_device_id, Device to run kernel on
	**Returns:
	**cl_kernel, Build kernel ready for arguments to be passed to it
	*/
	cl_kernel create_kernel(const char* kernel_sourceFile, const char* kernel_name, cl_context af_context, cl_device_id af_device_id);

	/*Calculate weighted 1st order autocorrelation using weighted Pearson normalised product moment correlation coefficient
	**Inputs:
	**data: std::vector<float>, One of the datasets to use in the calculation
	**Errors: std::vector<float>, Errors in dataset elements used in the calculation
	**Return:
	**float, Measure of the autocorrelation. 2-2*<return value> approximates the Durbin-Watson statistic for large datasets
	*/
	float weighted_pearson_autocorr(std::vector<float> data, std::vector<float> err, const int NUM_THREADS);

	/*Calculates the factorial of a small integer
	**Input:
	**n: long unsigned int, Number to find factorial of
	**Return:
	**long unsigned int, Reciprocal of input
	*/
	long unsigned int factorial(long unsigned int n);

	/*Calculates the power of 2 greater than or equal to the supplied number
	**Inputs:
	**n: int, Number to find the first positive power of 2 greater than or equal to
	**ceil: int, This parameter should not be inputted. It is used to recursively find the power of 2 greater than or equal to the supplied number
	**Return:
	**int, Power of 2 greater than or equal to the input
	*/
	int ceil_power_2(int n, int ceil = 1);

	/*Calculate Pearson normalised product moment correlation coefficient between 2 OpenCV mats for some offset between them
	**Beware: function needs debugging
	**Inputs:
	**img1: cv::Mat &, One of the mats
	**img2: cv::Mat &, The other mat
	**j: const int, Offset of the second mat's columns from the first's
	**i: const int, Offset of the second mat's rows from the first's
	**Return:
	**float, Pearson normalised product moment correlation coefficient between the 2 mats
	*/
	float pearson_corr(cv::Mat &img1, cv::Mat &img2, const int j, const int i);

	/*Create a copy of an image where the black pixels are replaced with the mean values of the image
	**img: cv::Mat &, Floating point mat to make a black pixel free copy of
	**Returns:
	**cv::Mat, Copy of input mat where black pixels have been replaced with the mean matrix value
	*/
	cv::Mat black_to_mean(cv::Mat &img);

	/*Gaussian blur an image based on its size
	**img: cv::Mat &, Floating point mam to blur
	**frac: float, Gaussian kernel size as a fraction of the image's smallest dimension's size
	**Returns,
	**cv::Mat, Blurred copy of the input mat
	*/
	cv::Mat blur_by_size(cv::Mat &img, float blur_frac = QUANT_GAUSS_FRAC);

	/*Calculate the autocorrelation of an OpenCV mat
	**img: cv::Mat &, Image to calculate the autocorrelation of
	**Returns,
	**cv::Mat, Autocorrelation of the image
	*/
	cv::Mat autocorrelation(cv::Mat &img);

	/*Calculate Pearson's normalised product moment correlation coefficient between 2 floating point same-size OpenCV mats
	**img1: cv::Mat &, One of the mats
	**img2: cv::Mat &, The other mat
	**Returns,
	**float, Pearson normalised product moment correlation coefficient between the 2 mats
	*/
	float pearson_corr(cv::Mat &img1, cv::Mat &img2);

	/*Calculate the average feature size in an image by summing the components of its 2D Fourier transform in quadrature to produce a 
	**1D frequency spectrum and then finding its weighted centroid
	**Inputs:
	**img: cv::Mat &, Image to get the average feature size in
	**Return:
	**float, Average feature size
	*/
	float get_avg_feature_size(cv::Mat &img);
}