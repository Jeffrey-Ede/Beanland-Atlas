#pragma once

#include <defines.h>

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

//Data type size limits
#include <limits>

//Gradient based symmetry detection
//#include <symmetry_func.h> //This algorithm is too slow

//Solve systems of equations
#include <Eigen/Dense>

//Developer utility functions
#include <developer_helper_func.h>

//Possible mirror symmetries
static const std::array<int, 4> pos_mir_sym = {2, 3, 4, 6}; //Add to this - see Richard's paper on CBED atlas symmetry

//Input data location
static const char* inputImagePath = "D:/data/default2.tif";

//Locations of kernels. Note: will update these to use an environmental variable based on where the user installs the software
static const char* annulus_source = "D:/Beanland-Atlas/Beanland-Atlas/Beanland-Atlas/create_annulus.cl"; //Create padded annulus
static const char* gauss_kernel_ext_source = "D:/Beanland-Atlas/Beanland-Atlas/Beanland-Atlas/gauss_kernel_padded.cl"; //Create padded Gaussian blurring kernel
static const char* freq_spectrum1D_source = "D:/Beanland-Atlas/Beanland-Atlas/Beanland-Atlas/freq_spectrum1D.cl"; //Convert 2D r2c Fourier spectrum into 1D spetrum
static const char* circle_source = "D:/Beanland-Atlas/Beanland-Atlas/Beanland-Atlas/create_circle.cl"; //Create padded circle

//Names of kernels
static const char* gauss_kernel_ext_kernel = "gauss_kernel_extended"; //Create padded annulus
static const char* freq_spectrum1D_kernel = "freq_spectrum1D"; //Create padded Gaussian blurring kernel
static const char* annulus_kernel = "create_annulus"; //Convert 2D r2c Fourier spectrum into 1D spetrum
static const char* circle_kernel = "create_circle"; //Create padded circle