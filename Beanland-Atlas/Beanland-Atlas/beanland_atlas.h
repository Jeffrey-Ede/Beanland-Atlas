#pragma once
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

#define TEST 1 //Used to turn testing on and off

//Mathematical constant PI
#define PI 3.141592654

//Number of times the perimeter before dividing 2pi by it by when calculating angular sweep resolution 
//to use for mirror line position refinement
#define NUM_PERIM 2

//When navigating Pearson product moment correlation coefficient space, only use 2nd derivative to estimate maximum position
//if it is larger than this small value for a float 
#define MIN_D_2 1e-8

//Maximum number of exploratory moves in Pearson product moment correlation coefficient space, as a safeguard
#define MAX_EXPL 50

//Minimum diameter of circles in pixels
#define MIN_CIRC_SIZE 5

//Add successive images to upper bound estimation until improvement in autocorrelation is less than this factor
#define AUTOCORR_REQ_CONV 0.1

//Maximum number of images to use in autocorrelation calculation
#define MAX_AUTO_CONTRIB 50

//Size of Gaussian kernel used for circle size calculation when low pass filtering
#define UBOUND_GAUSS_SIZE 3

//1 divided by the square root of 2. This is the maximum frequency radius of a 2D Fourier transform
#define INV_SQRT_OF_2 0.7071067812

//Square root of 2. It's useful
#define SQRT_OF_2 1.414213562

//OpenCL
#include <CL/cl.hpp>

#include <fstream>
#include <iostream>
#include <vector>

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

//Array containers
#include <array>

//Accelerated CPU-based FFT
#include <complex.h>
#include <fftw3.h>

//Possible mirror symmetries
static const std::array<int, 4> pos_mir_sym = {2, 3, 4, 6};

//Input data location
static const char* inputImagePath = "D:/data/default2.tif";

//Locations of kernels
static const char* annulus_kernel = "D:/C_C++_C#/create_annulus.cl";
static const char* circle_kernel = "D:/C_C++_C#/circle_kernel.cl";
static const char* gauss_kernel_ext_source = "D:/C_C++_C#/gauss_kernel_padded.cl";
static const char* freq_spectrum1D_source = "D:/C_C++_C#/freq_spectrum1D.cl";

//Names of kernels
static const char* gauss_kernel_ext_kernel = "gauss_kernel_extended";
static const char* freq_spectrum1D_kernel = "freq_spectrum1D";

/*Calculate upper bound for the size of circles in an image. This is done by convolving images with a gaussian low pass filter in the
**Fourier domain. The amplitudes of freq radii are then caclulated for the processed image. These are rebinned to get a histogram with
**equally spaced histogram bins. This process is repeated until the improvement in the autocorrelation of the bins no longer significantly
**increases as additional images are processed. The maximum amplitude above a minimum threshold is then used to set an upper limit on
**the size of the circles
**Inputs:
**&mats: std::vector<cv::Mat>, Vector of input images
**min_circ_size: int, Minimum dimeter of circles, in px
**autocorr_req_conv: float, Ratio of increased autocorrelation to previous must be at least this
**max_num_imgs: int, If autocorrelations don't converge after this number of images, method has failed
**sigma: float, standard deviation of Gaussian used to blur image
**af_context: cl_context, ArrayFire context
**af_device_id: cl_device_id, ArrayFire device
**af_queue: cl_command_queue, ArrayFire command queue
**NUM_THREADS: const int, Number of threads supported for OpenMP
**Returns:
**int, Upper bound for circles size, 0 if upper bound can't be determined
*/
int circ_size_ubound(std::vector<cv::Mat> &mats, int mats_rows_af, int mats_cols_af, int min_circ_size, float autocorr_req_conv,
	int max_num_imgs, float sigma, cl_context af_context, cl_device_id af_device_id, cl_command_queue af_queue, const int NUM_THREADS);

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
**af_context: cl_context, ArrayFire context
**af_device_id: cl_device_id, ArrayFire device id
**af_queue: cl_command_queue, ArrayFire command queue
**target_size: int, Downsampling factor will be the largest power of 2 that doesn't make the image smaller than this
**num_angles: int, Number of angles  to look for symmetry at
*/
std::vector<float> symmetry_axes(cv::Mat amalg, int origin_x, int origin_y, size_t num_angles = 120, float target_size = 0);

/*Calculate Pearson normalised product moment correlation coefficient between 2 vectors of floats
**Inputs:
**vect1: std::vector<float>, One of the datasets to use in the calculation
**vect2: std::vector<float>, The other dataset to use in the calculation
**Return:
**float, Pearson normalised product moment correlation coefficient between the 2 datasets
*/
float pearson_corr(std::vector<float> vect1, std::vector<float> vect2);

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
float wighted_pearson_autocorr(std::vector<float> data, std::vector<float> err);