#ifndef _SYMMETRY_CD_H
#define _SYMMETRY_CD_H

//
// Author:  Christoph Dalitz, Regina Pohle-Froehlich
// Date:    2014/05/26
//

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "symmetry_point.h"

namespace sym
{
	//-----------------------------------------------------------------
	// Convert float image to greyscale image.
	// Arguments:
	//  fimg = float image (input)
	//  gimg = greyscale image (output)
	//  greytrans = method for float to grey transformation (input):
	//                0 = linear transformation over entire range
	//                1 = linear transformation over only positive range
	//                2 = like 1, but with subsequent histogram equalization
	//                3 = equalization over entire float range
	//-----------------------------------------------------------------
	void float2grey(const cv::Mat &fimg, cv::Mat &gimg, int greytrans = 0);

	//-----------------------------------------------------------------
	// Convert float image to BGR image.
	// Arguments:
	//  fimg = float image (input)
	//  cimg = color image (output)
	//  equalize = when set, histogram is equalized (input)
	//-----------------------------------------------------------------
	void float2bgr(const cv::Mat &fimg, cv::Mat &cimg, bool equalize=false);

	//-----------------------------------------------------------------
	// Gradient normalization function
	// Derive your custom normalization routine for special normalizations
	// The call operator has two arguments:
	//  gx = x-component of gradient (input and output)
	//  gy = y-component of gradient (input and output)
	//-----------------------------------------------------------------
	struct GradientNormalization {
	  GradientNormalization(void) {};
	  void operator()(double &gx, double &gy) const;
	};

	//-----------------------------------------------------------------
	// Compute symmetry transform from a greyscale image.
	// To each point, a symmetry score value and a radius is assigned
	// Arguments:
	//  grey     = greyscale image (input)
	//  result   = symmetry scores as CV_64F image (output)
	//  result_r = symmetry radius as CV_64F image (output)
	//  radius   = maximum radius examined (input)
	//  alpha    = exponent for score normalisation with r^alpha (input)
	//  only     = 1 = only dark object, -1 = only light objects, 0 = all objects (input)
	//  gradnorm = optional gradient normalization function (default = NULL)
	//-----------------------------------------------------------------
	void symmetry_transform(const cv::Mat &grey, cv::Mat &result, cv::Mat &result_r, int radius, double alpha=0.5, int only=0, GradientNormalization *gradnorm=NULL);

	//-----------------------------------------------------------------
	// Compute symmetry transform from a greyscale image with rectangular regions.
	// To each point, a symmetry score value and a radius (rx,ry) is assigned
	// Arguments:
	//  grey      = greyscale image (input)
	//  result    = symmetry scores as CV_64F image (output)
	//  result_rx = x-component of symmetry radius as CV_64F image (output)
	//  result_ry = y-component of symmetry radius as CV_64F image (output)
	//  radius    = maximum radius examined (input)
	//  alpha     = exponent for score normalisation with r^alpha (input)
	//  only      = 1 = only dark object, -1 = only light objects, 0 = all objects (input)
	//  gradnorm  = optional gradient normalization function (default = NULL)
	//-----------------------------------------------------------------
	void symmetry_transform_rect(const cv::Mat &grey, cv::Mat &result, cv::Mat &result_rx, cv::Mat &result_ry, int radius, double alpha=0.5, int only=0, GradientNormalization *gradnorm=NULL);

	//-----------------------------------------------------------------
	// Find candidate symmetry centers in symmetry image as local
	// maxima of the symmetry score. Returns the number of points found.
	// Arguments:
	//  symmetry        = symmetry transform image filename (input)
	//  candidatepoints = symmetry candidate points (output)
	//  windowsize      = window size for local maximum search (input)
	//-----------------------------------------------------------------
	int points_local_maxima(const cv::Mat &symmetry, SymmetryPointVector &candidatepoints, int windowsize = 3);


	//-----------------------------------------------------------------
	// Compute the edge dispersion at the given point.
	// Arguments:
	//  gradx    = x-component of gradient image (input)
	//  grady    = y-component of gradient image (input)
	//  point    = the point to be examined (input)
	//  radius   = the radius of the window to be examined (input)
	//  cutoff   = ignore an image border of width cutoff (input)
	//-----------------------------------------------------------------
	double edge_dispersion_cd(const cv::Mat &gradx, const cv::Mat &grady, const cv::Point &point, int radius = 3, int cutoff = 0);

	//-----------------------------------------------------------------
	// Compute the edge directedness as the frequency of the most frequent
	// gradient direction.
	// Arguments:
	//  gradx    = x-component of gradient image (input)
	//  grady    = y-component of gradient image (input)
	//  point    = the point to be examined (input)
	//  radius   = the radius of the window to be examined (input)
	//  nbins    = number of bins in direction histogram (input)
	//  cutoff   = ignore an image border of width cutoff (input)
	//-----------------------------------------------------------------
	double edge_directedness_cd(const cv::Mat &gradx, const cv::Mat &grady, const cv::Point &point, int radius = 3, int nbins = 16, int cutoff = 0);


	//-----------------------------------------------------------------
	// Compute the ratio of the the two eigenvalues of the covariance matrix
	// of the window with the given radius around point.
	// Arguments:
	//  image     = complete image (input)
	//  point     = the point to be examined (input)
	//  radius    = the radius of the window to be examined (input)
	//  cutoff   = ignore an image border of width cutoff (input)
	//-----------------------------------------------------------------
	double coveigenratio_cd(const cv::Mat &image, const cv::Point &point, int radius = 3, int cutoff = 0);

	//-----------------------------------------------------------------
	// Returns the number of antiparallel gradient bins that have a
	// frequency count larger than count_threshold
	// inside the window with the given radius around point.
	// Arguments:
	//  gradx    = x-component of gradient image (input)
	//  grady    = y-component of gradient image (input)
	//  point    = the point to be examined (input)
	//  radius   = the radius of the window to be examined (input)
	//  max_frequency = frequency count of largest bin (output)
	//  weighted = whether edges shall be weighted by gradient strength (input)
	//  count_threshold = threshold when a bin is considered "occupied" (input)
	//  nbins    = number of bins in direction histogram (input)
	//-----------------------------------------------------------------
	double antiparallelity_rpf(const cv::Mat &gradx, const cv::Mat &grady, const cv::Point &point, int radius, double &max_frequency, bool weighted = false, double count_threshold = 0.1, int nbins = 8);

	//-----------------------------------------------------------------
	// Returns the skeleton starting at x until the score value
	// falls below the given percentage of S(x).
	// Arguments:
	//  p          = starting point (input)
	//  S          = symmetry transform (input)
	//  skeleton   = the points of the skeleton (output)
	//  percentage = threshold at which the skeleton is stopped (input)
	//  maxlength  = maximum length of the skeleton (input)
	//-----------------------------------------------------------------
	void get_skeleton(const cv::Point &p, const cv::Mat &S, PointVector* skeleton, double percentage, int maxlength);

	//-----------------------------------------------------------------
	// Returns the ratio skeleton_length/symmetry_region_size
	// Arguments:
	//  p        = starting point (input)
	//  skeleton = the points of the skeleton (input)
	//  rx       = x-component of cricumradius
	//  ry       = y-component of cricumradius
	//-----------------------------------------------------------------
	double skeleton_size(const cv::Point &x, const PointVector &skeleton, double rx, double ry);

	//-----------------------------------------------------------------
	// Returns the sum of scalar products at corresponding points along adjecent
	// diagonals
	// Arguments:
	//  p        = starting point (input)
	//  gradx    = x-component of gradient image (input)
	//  grady    = y-component of gradient image (input)
	//  rx       = x-component of cricumradius
	//  ry       = y-component of cricumradius
	//-----------------------------------------------------------------
	double diagonal_sum(const cv::Point &x, const cv::Mat &gradx, const cv::Mat &grady, double rx, double ry);

	//-----------------------------------------------------------------
	// Returns true, when QDA on all four features finds axial symmetry
	// Input arguments:
	//  edge_dir  = edge directedness
	//  skel_size = skeleton size
	//  anti_par  = antiparallelity
	//  cov_ratio = covariance eigenratio
	//-----------------------------------------------------------------
	bool axial_qda(double edge_dir, double skel_size, double anti_par, double cov_ratio);


	#endif
}