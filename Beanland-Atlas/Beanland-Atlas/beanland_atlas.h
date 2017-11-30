#pragma once
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

//Mathematical constant PI
#define PI 3.141592654

//Number of times the perimeter before dividing 2pi by it by when calculating angular sweep resolution 
//to use for mirror line position refinement
#define NUM_PERIM 2

//When navigating Pearson product moment correlation coefficient space, only use 2nd derivative to estimate maximum position
//if it is larger than this small value for a float 
#define MIN_D_2 1e-8

//Maximum number of exploratory moves in Pearson product moment correlation coefficient space as a safeguard against divergence
#define MAX_EXPL 25

//Minimum diameter of circles in pixels. Assume any reasonable data will have spots of at least this size
#define MIN_CIRC_SIZE 5

//Maximum number of images to use in autocorrelation calculation when finding the circle size upper bound from their separation
#define MAX_AUTO_CONTRIB 10

//Maximum number of images to use when performing convolutions to calculate the circle size
#define MAX_SIZE_CONTRIB 10

//Thickness of annulus when searching for circle size
#define INIT_ANNULUS_THICKNESS 2

//Size of Gaussian kernel used for circle size calculation when low pass filtering
#define UBOUND_GAUSS_SIZE 3

//1 divided by the square root of 2. This is the maximum frequency radius of a 2D Fourier transform
#define INV_SQRT_OF_2 0.7071067812

//Square root of 2. It's useful
#define SQRT_OF_2 1.414213562

//Size of Sobel filter kernel
#define SOBEL_SIZE 3

//Gaussian blurring filter size as a fraction of circle radius when blurring the cross correlation of an image's Sobel filtrate with
//an annulus after it has been scaled by the image's cross correlation with a circle
#define IMPULSE_BLUR 0.2

//OpenCL
#include <CL/cl.hpp>

//Input and output
#include <fstream>
#include <iostream>

//Data containers
#include <vector>
#include <array>

#include "opencv2/highgui/highgui.hpp" //Loading images
#include <opencv2/imgproc/imgproc.hpp> //Convert RGB to greyscale
#include "opencv2/core/ocl.hpp"

//ArrayFire GPU accelerated algorithms
#include <arrayfire.h>
#include <af/array.h> //ArrayFire only acts on ArrayFire arrays
#include <opencv2/core/core.hpp>
#include <af/opencl.h> //Use ArrayFire with OpenCL

#include <cstdio>

//Blob detection
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>

//Speed testing
#include <time.h>

//Dynamic, adaptive feature detection
#include <opencv2/features2d/features2d.hpp>

//Number of threads supported
#include <thread>

//CPU Parallelism
#include <omp.h>

//Accelerated CPU-based FFT
#include <complex.h>
#include <fftw3.h>

//Possible mirror symmetries
static const std::array<int, 4> pos_mir_sym = {2, 3, 4, 6};

//Input data location
static const char* inputImagePath = "D:/data/default2.tif";

//Locations of kernels. Note: will update these to use an environmental variable based on where the user installs the software
static const char* annulus_source = "D:/Beanland-Atlas/Beanland-Atlas/Beanland-Atlas/create_annulus.cl"; //Create padded annulus
static const char* gauss_kernel_ext_source = "D:/Beanland-Atlas/Beanland-Atlas/Beanland-Atlas/gauss_kernel_padded.cl"; //Create padded gaussian blurring kernel
static const char* freq_spectrum1D_source = "D:/Beanland-Atlas/Beanland-Atlas/Beanland-Atlas/freq_spectrum1D.cl"; //Convert 2D r2c Fourier spectrum into 1D spetrum
static const char* circle_source = "D:/Beanland-Atlas/Beanland-Atlas/Beanland-Atlas/create_circle.cl"; //Create padded circle

//Names of kernels
static const char* gauss_kernel_ext_kernel = "gauss_kernel_extended"; //Create padded annulus
static const char* freq_spectrum1D_kernel = "freq_spectrum1D"; //Create padded gaussian blurring kernel
static const char* annulus_kernel = "create_annulus"; //Convert 2D r2c Fourier spectrum into 1D spetrum
static const char* circle_kernel = "create_circle"; //Create padded circle

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

/*Create extended Gaussian to blur images with to remove high frequency components. It
**Inputs:
**rows: int, Number of columnss in input matrices. ArrayFire matrices are transposed so this is the number of rows of the ArrayFire
**array
**cols: int, Number of rows in input matrices. ArrayFire matrices are transposed so this is the number of columns of the ArrayFire
**array
**sigma: float, standard deviation of Gaussian used to blur image
**kernel: cl_kernel, OpenCL kernel that creates extended Gaussian
**af_queue: cl_command_queue, ArrayFire command queue
**Returns:
**af::array, ArrayFire array containing extended Guassian distribution
*/
af::array extended_gauss(int cols, int rows, float sigma, cl_kernel kernel, cl_command_queue af_queue);

/*Multiplies 2 C API ArrayFire arrays containing r2c Fourier transforms and returns a histogram containing the frequency spectrum
**of the result. Frequency spectrum bins all have the same width in the frequency domain
**Inputs:
**input_af: af_array, Amplitudes of r2c 2d Fourier transform
**length: size_t, Number of bins for frequency spectrum
**height: int, Height of original image
**width: int, Hidth of original image and of fft
**reduced_height: int, Height of fft, which is half the original hight + 1
**inv_height2: float, 1 divided by the height squared
**inv_width2: float, 1 divided by the width squared
**af_kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
**af_queue: cl_command_queue, ArrayFire command queue
**Returns:
**af::array, ArrayFire array containing the frequency spectrum of the elementwise multiplication of the Fourier transforms
*/
af::array freq_spectrum1D(af::array input_af, size_t length, int height, int width, int reduced_height, float inv_height2, 
	float inv_width2, cl_kernel kernel, cl_command_queue af_queue);

/*Downsamples amalgamation of aligned diffraction patterns, then finds approximate axes of symmetry
**Inputs:
**amalg: cv::Mat, OpenCV mat containing a diffraction pattern to find the axes of symmetry of
**origin_x: int, Position of origin accross the OpenCV mat
**origin_y: int, Position of origin down the OpenCV mat
**num_angles: int, Number of angles  to look for symmetry at
**target_size: int, Downsampling factor will be the largest power of 2 that doesn't make the image smaller than this
**Returns:
**std::vector<float>, highest Pearson normalised product moment correlation coefficient for a symmetry lins drawn through 
**a known axis of symmetry
*/
std::vector<float> symmetry_axes(cv::Mat &amalg, int origin_x, int origin_y, size_t num_angles = 120, float target_size = 0);

/*Calculate Pearson normalised product moment correlation coefficient between 2 vectors of floats
**Inputs:
**vect1: std::vector<float>, One of the datasets to use in the calculation
**vect2: std::vector<float>, The other dataset to use in the calculation
**NUM_THREADS: const int, The number of threads to use for OpenMP CPU acceleration
**Return:
**float, Pearson normalised product moment correlation coefficient between the 2 datasets
*/
float pearson_corr(std::vector<float> vect1, std::vector<float> vect2, const int NUM_THREADS);

///**Calculates the positions of repeating maxima in noisy data. A peak if the data's Fourier power spectrum is used to 
//**find the number of times the pattern repeats, assumung that the data only contains an integer number of repeats.
//**This number of maxima are then searched for.
//**Inputs:
//**corr: std::vector<float>, Noisy data to look for repeatis in
//**num_angles: int, Number of elements in data
//**pos_mir_sym: std::array<int, 4>, Numbers of maxima to look for. Will choose the one with the highest power spectrum value
//**Returns:
//**std::vector<int>, Idices of the maxima in the array. The number of maxima is the size of the array
//*/
//std::vector<int> repeating_max_loc(std::vector<float> corr, int num_angles, std::array<int, 4> pos_mir_sym);

/**Refine positions of mirror lines. 
**Inputs:
**amalg: cv::Mat, Diffraction pattern to refine symmetry lines on
**max_pos: std::vector<int>, Array indices corresponding to intensity maxima
**num_angles: max_pos indices are converted to angles via angle_i = max_pos[i]*PI/num_angles
**Returns:
**std::vector<cv::Vec3f>, Refined origin position and angle of each mirror line, in the same order as the input maxima
*/
std::vector<cv::Vec3f> refine_mir_pos(cv::Mat amalg, std::vector<int> max_pos, size_t num_angles, int origin_x, int origin_y, int range);

/*Calculate the mean estimate for the centre of symmetry
**Input:
**refined_pos: std::vector<cv::Vec3f>, 2 ordinates and the inclination of each line
**Returns:
**cv::Vec2f, The average 2 ordinates of the centre of symmetry
*/
cv::Vec2f avg_origin(std::vector<cv::Vec3f> lines);

/*Calculate all unique possible points of intersection, add them up and then divide by their number to get the average
**Input:
**refined_pos: std::vector<cv::Vec3f>, 2 ordinates and the inclination of each line
**Returns:
**cv::Vec2f, The average 2 ordinates of intersection
*/
cv::Vec2f average_intersection(std::vector<cv::Vec3f> lines);

//Returns factorial of input integer
long unsigned int factorial(long unsigned int n);

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
float wighted_pearson_autocorr(std::vector<float> data, std::vector<float> err, const int NUM_THREADS);

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

/*Create padded unblurred annulus with a given radius and thickness. Its inner radius is radius - thickness/2 and the outer radius
** is radius + thickness/2
**Inputs:
**length: size_t, Number of pixels making up annulus
**width: int, Width of padded annulus
**half_width: int, Half the width of the padded annulus
**height: int, Height of padded annulus
**half_height: int, Half the height of the padded annulus
**radius: int, radius of annulus, approximately halfway between its inner and outer radii
**thickness: int, thickness of annulus. If even, thickness will be rounded up to the next odd integer
**kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
**af_queue: cl_command_queue, ArrayFire command queue
**Returns:
**af::array, ArrayFire array containing unblurred annulus
*/
af::array create_annulus(size_t length, int width, int half_width, int height, int half_height, int radius, int thickness, 
	cl_kernel kernel, cl_command_queue af_queue);

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

/*Calculates values of Hann window function so that they are ready for repeated application
**Inputs:
**mat_rows: int, Number of rows in window
**mat_cols: int, Number of columns in windo
**NUM_THREADS: const int, Number of threads to use for OpenMP CPU acceleration
**Returns:
**cv::Mat, Values of Hann window function at the pixels making up the 
*/
cv::Mat create_hann_window(int mat_rows, int mat_cols, const int NUM_THREADS);

/*Applies Hann window to an image
**Inputs:
**mat: cv::Mat &, Image to apply window to
**win: cv::Mat &, Window to apply
**NUM_THREADS: const int, Number of threads to use for OpenMP CPU acceleration
*/
void apply_win_func(cv::Mat &mat, cv::Mat &win, const int NUM_THREADS);

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
**fft1: af::array &, One of the 2 Fourier transforms
**fft2: af::array &, Second of the 2 Fourier transforms
**img_idx1: int, index of the image used to create the first of the 2 Fourier transforms
**img_idx2: int, index of the image used to create the second of the 2 Fourier transforms
**Return:
**std::array<float, 5>, The 0th and 1st indices are the relative positions of images, the 2nd index is the value of the phase correlation
**and the 3rd and 4th indices hold the indices of the images being compared in the OpenCV mats container 
*/
std::array<float, 5> max_phase_corr(af::array &fft1, af::array &fft2, int img_idx1, int img_idx2);

/*Create padded unblurred circle with a specified radius
**Inputs:
**length: size_t, Number of pixels making up annulus
**width: int, Width of padded annulus
**half_width: int, Half the width of the padded annulus
**height: int, Height of padded annulus
**half_height: int, Half the height of the padded annulus
**radius: int, radius of annulus, approximately halfway between its inner and outer radii
**kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
**af_queue: cl_command_queue, ArrayFire command queue
**Returns:
**af::array, ArrayFire array containing unblurred circle
*/
af::array create_circle(size_t length, int width, int half_width, int height, int half_height, int radius,
	cl_kernel kernel, cl_command_queue af_queue);

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

/*Align the diffraction patterns using their known relative positions and average over the aligned px
**mats: std::vector<cv::Mat> &, Diffraction patterns to average over the aligned pixels of
**positions: std::vector<cv::Vec3f> &, Relative positions of the images
**Return:
**struct align_avg_mats, The first OpenCV mat is the average of the aligned diffraction patterns, the 2nd is the number of OpenCV mats
**that contributed to each pixel
*/
struct align_avg_mats {
	cv::Mat acc;
	cv::Mat num_overlap;
};
struct align_avg_mats align_and_avg(std::vector<cv::Mat> &mats, std::vector<std::array<float, 5>> &positions);

/*Refine the relative positions of the images using all the known relative positions
**positions: std::vector<cv::Vec2f>, Relative image positions and their weightings
**Return:
**std::vector<std::vector<int>>, Relative positions of the images, including the first image, to the first image in the same order as the
**images in the image stack
*/
std::vector<std::vector<int>> refine_rel_pos(std::vector<std::array<float, 5>> &positions);

/*Find the positions of the spots in the aligned image average pattern. Most of these spots are made from the contributions of many images
**so it can be assumed that they are relatively featureless
**align_avg: cv::Mat &, Average values of px in aligned diffraction patterns
**radius: int, Radius of spots
**thickness: int, Thickness of annulus to convolve with spots
**annulus_creator: cl_kernel, OpenCL kernel to create padded unblurred annulus to cross correlate the aligned image average pattern Sobel
**filtrate with
**circle_creator: cl_kernel, OpenCL kernel to create padded unblurred circle to cross correlate he aligned image average pattern with
**gauss_creator: cl_kernel, OpenCL kernel to create padded Gaussian to blur the annulus and circle with
**af_queue: cl_command_queue, ArrayFire command queue
**align_avg_cols: int, Number of columns in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
**align_avg_rows: int, Number of rows in the aligned image average pattern OpenCV mat. ArrayFire arrays are transpositional
**Return:
std::vector<std::array<int, 2>> Positions of spots in the aligned image average pattern
*/
std::vector<std::array<int, 2>> get_spot_pos(cv::Mat &align_avg, int radius, int thickness, cl_kernel annulus_creator, cl_kernel circle_creator, 
	cl_kernel gauss_creator, cl_command_queue af_queue, int align_avg_cols, int align_avg_rows);

/**Calculates the positions of repeating maxima in noisy data. A peak if the data's Fourier power spectrum is used to 
**find the number of times the pattern repeats, assumung that the data only contains an integer number of repeats.
**This number of maxima are then searched for.
**Inputs:
**corr: std::vector<float>, Noisy data to look for repeatis in
**num_angles: int, Number of elements in data
**pos_mir_sym: std::array<int, 4>, Numbers of maxima to look for. Will choose the one with the highest power spectrum value
**Returns:
**std::vector<int>, Idices of the maxima in the array. The number of maxima is the size of the array
*/
std::vector<int> repeating_max_loc(std::vector<float> corr, int num_angles, std::array<int, 4> pos_mir_sym);