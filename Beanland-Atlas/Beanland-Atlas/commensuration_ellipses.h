#pragma once

#include <includes.h>

#include <aberration_correction.h>
#include <commensuration.h>
#include <ident_sym_utility.h>
#include <matlab.h> //Matlab-specific includes
#include <utility.hpp>

namespace ba
{
	//Number of bins to use when calculating histogram to determine the which threshold to apply to gradient-based symmetry space
    #define THRESH_PROP_HIST_SIZE 100 

	//Hyper-renormalisation ellipse fitting defaults
    #define HYPER_RENORM_DEFAULT_SCALE 1 //Scale of the ellipe. 1 is arbitrary. Choosing a better value will reduce numberical errors
    #define HYPER_RENORM_DEFAULT_THRESH 0.9 //Cosine of angle between eigenvectors divided by ratio of higher sized to smaller for conclusion of iterations
    #define HYPER_RENORM_DEFAULT_ITER 15 //Maximum number of iterations

	//Accuracy to find distances from ellipses as a fraction of the pixel size
    #define DISTS_FROM_EL_ACC 0.1

	//Custom data structure to hold ellipse parameters
	struct ellipse_param {
		bool is_ellipse;
		cv::Point2d center;
		std::vector<cv::Point2d> extrema;
		double a, b, angle;
	};
	typedef ellipse_param ellipse;

	/*Get ellipses describing each spot from their Scharr filtrates. Ellipses are checked using heuristic arguments:
	**ellipse shapes vary smoothly with time and ellipse shaps must be compatible with a projection of an array of
	**circles onto a flat detector
	**Inputs:
	**mats: std::vector<cv::Mat> &, Individual images to extract spots from
	**spot_pos: std::vector<cv::Point> &, Positions of located spots in aligned diffraction pattern
	**ellipses: std::vector<std::vector<std::vector<cv::Point>>> &, Positions of the minima and maximal extensions of spot ellipses.
	**The ellipses are decribed in terms of 3 points, clockwise from the top left as it makes it easy to use them to perform 
	**homomorphic warps. The nesting is each image, each spot in the order of their positions in the positions vector, set of
	**3 points (1 is extra) desctribing the ellipse, in that order
	**acc: cv::Mat &, Average of the aligned diffraction patterns
	*/
	void get_spot_ellipses(std::vector<cv::Mat> &mats, std::vector<cv::Point> &spot_pos, cv::Mat &acc, 
		std::vector<std::vector<std::vector<cv::Point>>> &ellipses);

	/*Amplitude of image's Scharr filtrate
	**Inputs:
	**img: cv::Mat &, Floating point image to get the Scharr filtrate of
	**scharr_amp: cv::Mat &, OpenCV mat to output the amplitude of the Scharr filtrate to
	*/
	void scharr_amp(cv::Mat &img, cv::Mat &scharr_amp);

	/*Use weighted sums of squared differences to calculate the sizes of the ellipses from an image's Scharr filtrate
	**Inputs:
	**img: cv::Mat &, Image to find the size of ellipses at the estimated positions in
	**spot_pos: std::vector<cv::Point>, Positions of spots in the image
	**est_rad: std::vector<cv::Vec2f> &, Two radii to look for the ellipse between
	**ellipses: std::vector<std::vector<std::vector<double>>> &, For each spot that an ellipse can be fitted to, a set of
	**5 parameters describing an ellipse. By index: 0 - x position, 1 - y position, 2 - major axis, 3 - minor axis, 4 - Angle
	**between the major axis and the x axis
	*/
	void get_ellipses(cv::Mat &img, std::vector<cv::Point> spot_pos, std::vector<cv::Vec2f> est_rad,
		std::vector<std::vector<double>> &ellipses);

	/*Create annular mask
	**Inputs:
	**size: const int, Size of the mask. This should be an odd integer
	**inner_rad: const int, Inner radius of the annulus
	**outer_rad: const int, Outer radius of the annulus
	**val, const byte, Value to set the elements withing the annulus. Defaults to 1
	*/
	void create_annular_mask(cv::Mat annulus, const int size, const int inner_rad, const int outer_rad, const byte val = 1);

	/*Extracts values at non-zero masked elements in an image, constraining the boundaries of a mask so that only maked 
	**points that lie in the image are extracted. It is assumed that at least some of the mask is on the image
	**Inputs:
	**img: cv::Mat &, Image to apply the mask to
	**dst: cv::Mat &, Extracted pixels of image that mask has been applied to
	**mask: cv::Mat &, Mask being applied to the image
	**top_left: cv::Point2i &, Indices of the top left corner of the mask on the image
	*/
	void get_mask_values(cv::Mat &img, cv::Mat &dst, cv::Mat &mask, cv::Point2i &top_left);

	/*Threshold a proportion of values using an image histogram
	**Inputs:
	**img: cv::Mat &, Image of floats to threshold
	**thresh: cv::Mat &, Output binary thresholded image
	**thresh_frac: const float, Proportion of the image to threshold
	**thresh_mode: const int, Type of binarialisation. Defaults to cv::THRESH_BINARY, which marks the highest 
	**proportion
	**hist_bins: const int, Number of bins in histogram used to determine threshold
	**non-zero: const bool, If true, only use non-zero values when deciding the threshold. Defaults to false.
	**Returns:
	**unsigned int, Number of points in the thresholded image
	*/
	unsigned int threshold_proportion(cv::Mat &img, cv::Mat &thresh, const float thresh_frac, const int thresh_mode = cv::THRESH_BINARY,
		const int hist_bins = THRESH_PROP_HIST_SIZE, const bool non_zero = false);

	/*Weight the fit an ellipse to a noisy set of data using hyper-renormalisation. The function generates the coefficients
	**A0 to A5 when the data is fit to the equation A0*x*x + 2*A1*x*y + A2*y*y + 2*f0*(A3*x + A4*y) + f0*f0*A5 = 0, in
	**that order
	**Inputs:
	**mask: cv::Mat &, 8-bit mask. Data points to be used are non-zeros
	**weights: cv::Mat &, 32-bit mask. Weights of the individual data points
	**f0: const double, Approximate size of the ellipse. This is arbitrary, but choosing a value close
	**to the correct size reduces numerical errors
	**thresh: const double, Iterations will be concluded if the cosine of the angle between successive eigenvectors divided
	**by the amplitude ratio (larger divided by the smaller) is smaller than the threshold
	**max_iter: const int, The maximum number of iterations to perform. If this limit is reached, the last iteration's conic
	**coefficients will be returned
	**Returns:
	**std::vector<double>, Coefficients of the conic equation
	*/
	std::vector<double> hyper_renorm_ellipse(cv::Mat &mask, cv::Mat weights, const double f0 = HYPER_RENORM_DEFAULT_SCALE,
		const double thresh = HYPER_RENORM_DEFAULT_THRESH, const int max_iter = HYPER_RENORM_DEFAULT_ITER);

	/*Calculate the center and 4 extremal points of an ellipse (at maximum and minimum distances from the center) from
	**the coefficients of the conic equation A*x*x + B*x*y + Cy*y + D*x + E*y + F = 0
	**Input:
	**conic: std::vector<double> &, Coefficients of the conic equation
	**Returns:
	**Ellipse, Points describing the ellipse. If the conic does not describe an ellipse, the ellipse is
	**returned empty
	*/
	ellipse ellipse_points_from_conic(std::vector<double> &conic);

	/*Rotate a point anticlockwise
	**Inputs:
	**point: cv::Point &, Point to rotate
	**angle: const double, Angle to rotate the point anticlockwise
	**Returns:
	**cv::Point2d, Rotated point
	*/
	cv::Point2d rotate_point2D(cv::Point2d &point, const double angle);

	/*Exploit the inverse square law to find the sign of the inciding angle from an image's bacgkround
	**Inputs:
	**img: cv::Mat, Diffraction pattern to find which side the electron beam is inciding from
	**img_spot_pos: std::vector<cv::Point2d> &, Approximate positions of the spots on the image
	**fear: const float, Only use background at least this distance from the spots
	**dir: cv::Vec2d &, Vector indicating direction of maximum elongation due to the incidence angularity
	**Returns:
	**double, +/- 1.0: +1 means that elongation is in the same direction as decreasing intensity
	*/
	double inv_sqr_inciding_sign(cv::Mat img, std::vector<ellipse> &ellipses, const float fear, cv::Vec2d &dir);

	/*Perform weighted k-means clustering using a MATLAB script
	**Inputs:
	**data: std::vector<std::vector<double>> &, Data set to apply weighted k-means clustering to. The data set for each variable
	**should be the same size. The inner vector is the values for a particular varaible
	**weights: std::vector<double> &, Weights to apply when k-means clustering
	**k: const int, Number of clusters
	**centers: std::vector<std::vector<double>> &, Centroid locations
	**labels: std::vector<int> &, Cluster each data point is in
	*/
	void weighted_kmeans(std::vector<std::vector<double>> &data, std::vector<double> &weights, const int k, 
		std::vector<std::vector<double>> &centers, std::vector<int> &labels);

	/*Get distances of points from an ellipse moving across columns in each row in that order
	**Inputs:
	**mask: cv::Mat &, 8-bit mask whose non-zero values indicate the positions of points
	**img: cv::Mat &, Image to record the values of at the positions marked on the mask
	**param: std::vector<double> &, Parameters describing an ellipse. By index: 0 - x position, 1 - y position, 2 - major
	**axis, 3 - minor axis, 4 - Angle between the major axis and the x axis
	**dists: std::vector<double> &, Output distances from the ellipse,
	**accuracy: const double, Accuracy to find distances from ellipses to
	*/
	void dists_from_ellipse(cv::Mat &mask, cv::Mat &img, std::vector<double> &param, std::vector<double> &dists,
		const double accuracy = DISTS_FROM_EL_ACC);
}