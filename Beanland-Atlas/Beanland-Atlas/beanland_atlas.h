#pragma once

//Default parameters
#include <defines.h>

//External libraries
#include <includes.h>

//Developer utility functions
#include <developer_utility.h>

//Atlas construction
#include <atlas_construction.h>

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