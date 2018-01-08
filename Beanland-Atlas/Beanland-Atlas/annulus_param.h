#pragma once

#include <includes.h>

#include <kernel_launchers.h>

namespace ba
{
	/*Convolve images with annulus with a range of radii and until the autocorrelation of product moment correlation of the
	**spectra decreases when adding additional images' contribution to the spectrum. The annulus radius that produces the best
	**fit is then refined from this spectrum, including the thickness of the annulus
	**Inputs:
	**mats: std::vector<cv::Mat> &, reference to a sequence of input images to use to refine the annulus parameters
	**min_rad: int, minimum radius of radius to create
	**max_rad: int, maximum radius of radius to create
	**init_thickness: int, Thickness to use when getting initial radius of annulus. The radii tested are separated by this value
	**max_contrib: int, Maximum number of images to use. If the autocorrelation of the autocorrelation of the cross correlation
	**still hasn't decreased after this many images, the parameters will be calculated from the images processed so far
	**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**gauss_fft_af: Fourier transform of Gaussian to blur annuluses with
	**af_context: cl_context, ArrayFire context
	**af_device_id: cl_device_id, ArrayFire device
	**af_queue: cl_command_queue, ArrayFire command queue
	**NUM_THREADS: const int, Number of threads supported for OpenMP
	**Returns:
	**std::vector<int>, Refined radius and thickness of annulus, in that order
	*/
	std::vector<int> get_annulus_param(cv::Mat &mat, int min_rad, int max_rad, int init_thickness, int max_contrib,
		int mats_rows_af, int mats_cols_af, af::array &gauss_fft_af, cl_kernel create_annulus_kernel, cl_command_queue af_queue, 
		int NUM_THREADS);

	/*Calculates relative area of annulus to divide cross-correlations by so that they can be compared
	**Inputs:
	**rad: int, Average of annulus inner and outer radii
	**thickness: int, Thickness of annulus. 
	**Returns
	**float, Sum of values of pixels maxing up blurred annulus
	*/
	float sum_annulus_px(int rad, int thickness);

	/*Refines annulus radius and thickness estimate based on a radius estimate and p/m the accuracy that radius is known to
	**Inputs:
	**rad: int, Estimated spot radius
	**range: int, The refined radius estimate will be within this distance from the initial radius estimate.
	**length: size_t, Number of px making up padded annulus
	**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**half_width: int, Half the number of ArrayFire columns
	**half_height: int, Half the number of ArrayFire rows
	**gauss_fft_af: af::array, r2c ArrayFire FFT of Gaussian blurring kernel
	**fft: af::array, r2c ArrayFire FFT of input image
	**create_annulus_kernel: cl_kernel, OpenCL kernel that creates the padded annulus
	**af_queue: cl_command_queue, OpenCL command queue
	**Returns
	**std::vector<int>, Refined radius and thickness of annulus, in that order
	*/
	std::vector<int> refine_annulus_param(int rad, int range, size_t length, int mats_cols_af, int mats_rows_af, int half_width, 
		int half_height, af::array gauss_fft_af, af::array fft, cl_kernel create_annulus_kernel, cl_command_queue af_queue);
}