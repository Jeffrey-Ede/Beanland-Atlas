#pragma once

#include <defines.h>
#include <includes.h>

namespace ba
{
	/*Calculate the relative positions between images needed to align them.
	**Inputs:
	**mats: cv::Mat &, Images
	**hann_LUT: cv::Mat &, Precalculated look up table to apply Hann window function with
	**annulus_fft: af::array &, Fourier transform of Gaussian blurred annulus that has been recursively convolved with itself to convolve the gradiated
	**image with
	**circle_fft: af::array &, Fourier transform of Gaussian blurred circle to convolve gradiated image with to remove the annular cross correlation halo
	**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**Return:
	**std::vector<std::array<float, 5>>, Positions of each image relative to the first. The third element of the cv::Vec3f holds the value
	**of the maximum phase correlation between successive images
	*/
	std::vector<std::array<float, 5>> img_rel_pos(std::vector<cv::Mat> &mats, cv::Mat &hann_LUT, af::array &annulus_fft, af::array &circle_fft,
		int mats_rows_af, int mats_cols_af);

	/*Use the convolution theorem to create a filter that performs the recursive convolution of a convolution filter with itself
	**Inputs:
	**filter: af::array &, Filter to recursively convolved with its own convolution
	**n: int, The number of times the filter is recursively convolved with itself
	**Return:
	**af::array, Fourier domain matrix that can be elemtwise with another fft to recursively convolute it with a filter n times
	*/
	af::array recur_conv(af::array &filter, int n);

	/*Finds the position of max phase correlation of 2 images from their Fourier transforms
	**Inputs:
	**fft1: af::array &, One of the 2 Fourier transforms
	**fft2: af::array &, Second of the 2 Fourier transforms
	**img_idx1: int, index of the image used to create the first of the 2 Fourier transforms
	**img_idx2: int, index of the image used to create the second of the 2 Fourier transforms
	**Return:
	**std::array<float, 5>, The 0th and 1st indices are the relative positions of images, the 2nd index is the value of the phase correlation
	**and the 3rd and 4th indices hold the indices of the images being compared in the OpenCV mats container 
	*/
	std::array<float, 5> max_phase_corr(af::array &fft1, af::array &fft2, int img_idx1, int img_idx2);

	/*Primes images for alignment. The primed images are the Gaussian blurred cross correlation of their Hann windowed Sobel filtrate
	**with an annulus after it has been scaled by the cross correlation of the Hann windowed image with a circle
	**img: cv::Mat &, Image to prime for alignment
	**annulus_fft: af::array &, Fourier transform of the convolution of a Gaussian and an annulus
	**circle_fft: af::array &, Fourier transform of the convolution of a Gaussian and a circle
	**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**Return:
	**af::array, Image primed for alignment
	*/
	af::array prime_img(cv::Mat &img, af::array &annulus_fft, af::array &circle_fft, int mats_rows_af, int mats_cols_af);

}