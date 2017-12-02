#pragma once

//Enable use of older OpenCL APIs
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

//Minimum anglular separation between lattice vectors in rad
#define LATTICE_VECT_DIR_DIFF 0.52 //About 30 degrees for now

//Proportion of spot radius to search around lattice points for maximum in
#define SCALE_SEARCH_RAD 0.5