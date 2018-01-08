#pragma once

#include <includes.h>

#include <kernel_launchers.h>
#include <utility.h>

namespace ba
{
	/*Calculate upper bound for the size of circles in an image. This is done by convolving images with a gaussian low pass filter in the
	**Fourier domain. The amplitudes of freq radii are then caclulated for the processed image. These are rebinned to get a histogram with
	**equally spaced histogram bins. This process is repeated until the improvement in the autocorrelation of the bins no longer significantly
	**increases as additional images are processed. The error-weighted centroid of the 1D spectrum is then used to generate an upper bound for
	**the separation of the circles
	**Inputs:
	**&mats: std::vector<cv::Mat>, Vector of input images
	**mats_rows_af: int, Rows in ArrayFire array containing an input image. ArrayFire arrays are transpositional to OpenCV mats
	**mats_cols_af: int, Rows in ArrayFire array containing an input image. ArrayFire arrays are transpositional to OpenCV mats
	**gauss: af::array, ArrayFire array containing Fourier transform of a gaussian blurring filter to reduce high frequency components of
	**the Fourier transforms of images with
	**min_circ_size: int, Minimum dimeter of circles, in px
	**max_num_imgs: int, If autocorrelations don't converge after this number of images, uses the current total to estimate the size
	**af_context: cl_context, ArrayFire context
	**af_device_id: cl_device_id, ArrayFire device
	**af_queue: cl_command_queue, ArrayFire command queue
	**NUM_THREADS: const int, Number of threads supported for OpenMP
	**Returns:
	**int, Upper bound for circles size
	*/
	int circ_size_ubound(std::vector<cv::Mat> &mats, int mats_rows_af, int mats_cols_af, af::array &gauss, int min_circ_size,
		int max_num_imgs, cl_context af_context, cl_device_id af_device_id, cl_command_queue af_queue, const int NUM_THREADS);
}