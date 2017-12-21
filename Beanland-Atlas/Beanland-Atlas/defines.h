#pragma once

//Enable use of older OpenCL APIs
#define CL_USE_DEPRECATED_OPENCL_2_0_APIS
#define __CL_ENABLE_EXCEPTIONS

namespace ba
{
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

	//Proportion of smallest lattice vector size to search around lattice position for already detected spot
	#define SCALE_SEARCH_RAD 0.8

	//Size of bilateral filter used to preprocess images
	#define PREPROC_MED_FILT_SIZE 3

	//Threshold when determining equidistant spots for atlas symmetry calculation
	#define EQUIDST_THRESH 0.29

	//Factor to convert radians to degrees
	#define RAD_TO_DEG 57.29577951

	//Fraction of rectangular roi size to base blurring Gaussian standard deviation on when calculating the sum of squared differences
	#define QUANT_GAUSS_FRAC 0.03

	//Size of Gaussian blurring filter when preprocessing individual spots to find the Bragg profile
	#define BRAGG_PROF_PREPROC_GAUSS 3

	//Maximum fraction of the pot radius that a spot can lie away from the initially calculated lattice vectors and not be excluded from the lattice
	//vector refinement calculation
	#define SPOT_POS_TOL 0.3

	//Proportion of the larger initially estimated lattice vector size to incrementally iterate over +/- from the initial estimate when refining
	//the lattice vectors
	#define LATT_REF_RANGE 0.02

	//Required accuracy of refined lattice vector
	#define LATT_REF_REQ_ACC 0.1

	//Minimum range of values to interate over when refining the lattice vectors
	#define MIN_LATT_REF_RANGE 1.0f

	//Full width of centroid to use when refining positions to calculate the radius of the Ewald sphere
	#define SAMP_TO_DETECT_CENTROID 5

	//Spots not at least a minimum distance from the boundary of the aligned diffraction patterns are discarded
	#define DISCARD_SPOTS_DEFAULT -1 //-1 indicates that spots within 1 radius of the boundary should be discarded. Otherwised, the value is specified in px

	//Minimum fraction of the maximum mean Pearson normalised product moment correlation coefficient that a mean Pearson coefficient can be for
	//it's symmetry to be registered as present
    #define FRAC_FOR_SYM 0.5

	//Symmetry centers calculated from Scharr filtrates must use at least this fraction of the of the image area
    #define SYM_CENTER_USE_FRAC 0.0f

	//Default value to set symmetry centers when the minimum area is not available to perform their calculation
    #define SYM_CENTER_NOT_CALC_VAL 0.0f

	/* Information about images being compared when calculating their relative shift that can help constrain the shift */

	//Approximate fraction of the gradient based symmetry calculation values to use
    #define GRAD_SYM_USE 0.5

	//Number of bins to use when calculating histogram to determine the which threshold to apply
	#define GRAD_SYM_HIST_SIZE 100 

	typedef enum {
		REL_SHIFT_WIS_NONE, //No symmetry point or it is not pragmatic/possible to information about the symmetry point
		REL_SHIFT_WIS_INTERNAL_MIR0, //Mirror flip is perpendicular to radially outwards direction
		REL_SHIFT_WIS_INTERNAL_MIR1, //Mirror flip is in the radially outwards direction
		REL_SHIFT_WIS_INTERNAL_ROT //Rotation about point in the image
	};

	//Fraction of extracted rectangular region of interest to use when calculating the sum of squared differences to match it against another region
    #define QUANT_SYM_USE_FRAC 0.4

	//Fraction of extracted rectangular region of interest to use when calculating the sum of squared differences to match it against another region
	//for internal rotational symmetry
    #define INTERAL_ROT_SSD_FRAC 1.0

	//Fraction of extracted rectangular region of interest to use when calculating the sum of squared differences to match it against another region
	//for internal mirror symmetry where the flip is perpendicular to radially outwards direction
    #define INTERAL_MIR0_SSD_FRAC 0.7


	//Fraction of extracted rectangular region of interest to use when calculating the sum of squared differences to match it against another region
	//for internal mirror symmetry where the flip is in the radially outwards direction
    #define INTERAL_MIR1_SSD_FRAC 1.0

	//Default padding to apply when calculating Pearson's normalised product moment correlation coefficient in the Fourier domain
    #define FOURIER_PEARSON_USE_FRAC 0.5
}