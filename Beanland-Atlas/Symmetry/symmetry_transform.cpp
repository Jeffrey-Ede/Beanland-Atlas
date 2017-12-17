//
// Author:  Christoph Dalitz, Regina Pohle-Froehlich
// Version: 2.0a from 2014/04/16
//

#include <make_defs.h>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "symmetry_point.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <limits>
#include <vector>
#include <algorithm>

namespace sym
{
	//-----------------------------------------------------------------
	// sign of floating point number
	//-----------------------------------------------------------------
	inline int theta(double x) { if (x>0) return 1; else return 0;}

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
	void float2grey(const cv::Mat &fimg, cv::Mat &gimg, int greytrans/*=0*/) {
	  double minval,maxval;
	  cv::minMaxLoc(fimg, &minval, &maxval);
	  if (greytrans == 0) {
		cv::convertScaleAbs(fimg, gimg, 255.0/(maxval-minval), -minval*255.0/(maxval-minval));
		return;
	  }
	  else if (greytrans == 1 || greytrans == 2) {
		cv::Mat tmp(fimg);
		for (int y=0; y<fimg.rows; y++) {
		  for (int x=0; x<fimg.cols; x++) {
			if (fimg.at<double>(y,x) < 0) tmp.at<double>(y,x) = 0;
		  }
		}
		cv::convertScaleAbs(tmp, gimg, 255.0/maxval, 0);
		if (greytrans == 2) cv::equalizeHist(gimg, gimg);
		return;
	  }

	  // approximately equalize float image density
	  size_t n = 8192;
	  double N = (double)fimg.rows*fimg.cols;
	  std::vector<double> hist(n, 0.0);
	  std::vector<double> cdf(n);
	  double range = maxval-minval;

	  // compute histogram and cdf
	  for (int y=0; y<fimg.rows; y++) {
		for (int x=0; x<fimg.cols; x++) {
		  if (fimg.at<double>(y,x) == maxval) {
			hist[n-1] += 1;
		  } else {
			hist[(int)n*(fimg.at<double>(y,x)-minval)/range] += 1;
		  }
		}
	  }
	  cdf[0] = hist[0]/N;
	  for (size_t i=1; i<cdf.size(); i++) {
		cdf[i] = cdf[i-1] + hist[i]/N;
	  }

	  // compute equalized float image and convert it linearly to greyscale
	  cv::Mat eimg(fimg.rows, fimg.cols, CV_64F);
	  for (int y=0; y<fimg.rows; y++) {
		for (int x=0; x<fimg.cols; x++) {
		  if (fimg.at<double>(y,x) == maxval) {
			eimg.at<double>(y,x) = 1;
		  } else {
			eimg.at<double>(y,x) = cdf[(int)n*(fimg.at<double>(y,x)-minval)/range];
		  }
		}
	  }
	  cv::convertScaleAbs(eimg, gimg, 255.0, 0.0);
	}

	// variant that only rescales the positive values
	// (currently not used)
	void float2grey_onlypos(const cv::Mat &fimg, cv::Mat &gimg, bool equalize/*=false*/) {
	  double minval,maxval;
	  cv::minMaxLoc(fimg, &minval, &maxval);
	  if (!equalize) {
		cv::convertScaleAbs(fimg, gimg, 255.0/(maxval-minval), -minval*255.0/(maxval-minval));
		return;
	  }

	  // approximately equalize float image density
	  size_t n = 8192;
	  double N = 0;
	  std::vector<double> hist(n, 0.0);
	  std::vector<double> cdf(n);

	  // compute histogram and cdf
	  for (int y=0; y<fimg.rows; y++) {
		for (int x=0; x<fimg.cols; x++) {
		  if (fimg.at<double>(y,x) == maxval) {
			hist[n-1] += 1;
			N += 1;
		  } else {
			if (fimg.at<double>(y,x) > 0) {
			  hist[(int)n*fimg.at<double>(y,x)/maxval] += 1;
			  N += 1;
			}
		  }
		}
	  }
	  cdf[0] = hist[0]/N;
	  for (size_t i=1; i<cdf.size(); i++) {
		cdf[i] = cdf[i-1] + hist[i]/N;
	  }

	  // compute equalized float image and convert it linearly to greyscale
	  cv::Mat eimg(fimg.rows, fimg.cols, CV_64F);
	  for (int y=0; y<fimg.rows; y++) {
		for (int x=0; x<fimg.cols; x++) {
		  if (fimg.at<double>(y,x) == maxval) {
			eimg.at<double>(y,x) = 1;
		  }
		  else if (fimg.at<double>(y,x) <= 0) {
			eimg.at<double>(y,x) = 0;
		  }
		  else {
			eimg.at<double>(y,x) = cdf[(int)n*fimg.at<double>(y,x)/maxval];
		  }
		}
	  }
	  cv::convertScaleAbs(eimg, gimg, 255.0, 0.0);
	}

	//-----------------------------------------------------------------
	// Convert float image to BGR image.
	// Arguments:
	//  fimg = float image (input)
	//  cimg = color image (output)
	//  equalize = when set, histogram is equalized (input)
	//-----------------------------------------------------------------
	void float2bgr(const cv::Mat &fimg, cv::Mat &cimg, bool equalize/*=false*/) {
	  cv::Mat gimg;
	  float2grey(fimg, gimg, equalize);
	  cv::cvtColor(gimg, cimg, CV_GRAY2BGR);
	}

	//-----------------------------------------------------------------
	// Gradient normalization function
	// Derive your custom normalization routine for special normalizations
	// The call operator has two arguments:
	//  gx = x-component of gradient (input and output)
	//  gy = y-component of gradient (input and output)
	//-----------------------------------------------------------------
	struct GradientNormalization {
	  //GradientNormalization(void) {}
	  void operator()(double &gx, double &gy) const {
		// logarithmic normalization
		double norm = sqrt(gx*gx + gy*gy);
		if (norm > 0.1) {
		  double lognorm = log(1+norm);
		  gx = gx*lognorm/norm;
		  gy = gy*lognorm/norm;
		}
	  }
	};

	//-----------------------------------------------------------------
	// Compute symmetry score S and radius R at point (x,y)
	// Arguments:
	//  grad(x|y) = gradient image (input)
	//  R         = best radius (output)
	//  S         = score at best radius (output)
	//  radius    = maximum radius examined (input)
	//  norm_at_r = precalculated normalisation factors r^alpha (input)
	//  x,y       = point position (input)
	//  only      = 1 = only dark object, -1 = only light objects, 0 = all objects (input)
	//-----------------------------------------------------------------
	void s_and_r_at_point(const cv::Mat &gradx, const cv::Mat &grady, int* R, double* S, int radius, const std::vector<double> &norm_at_r, int x, int y, int only=0)
	{
	  int dx,dy,r,max_radius;
	  double symmetry, scalarprod;
	  std::vector<double> weight_at_r(radius+1);
	  //weight_at_r[0] = 0.0; // not needed
  
	  // adjust radius so that it does not extend beyond the image border
	  max_radius = radius;
	  if (y < max_radius)
		max_radius = y;
	  if (y >= gradx.rows - max_radius)
		max_radius = gradx.rows - y - 1;
	  if (x < max_radius)
		max_radius = x;
	  if (x >= gradx.cols - max_radius)
		max_radius = gradx.cols - x - 1;

	  // compute weight in window with radius max_radius
	  if (max_radius < 1) {
		*R = 0;
		*S = 0.0;
	  } else {
		symmetry = 0.0;
		if (only == 0) {
		  for (r=1; r<=max_radius; r++) {
			dy = r;
			for (dx=-r; dx<=r; dx++) {
			  scalarprod = gradx.at<double>(y+dy,x+dx)*gradx.at<double>(y-dy,x-dx)
				+ grady.at<double>(y+dy,x+dx)*grady.at<double>(y-dy,x-dx);
			  symmetry += -scalarprod;
			}
			dx = r;
			for (dy=-r+1; dy<r; dy++) {
			  scalarprod = gradx.at<double>(y+dy,x+dx)*gradx.at<double>(y-dy,x-dx)
				+ grady.at<double>(y+dy,x+dx)*grady.at<double>(y-dy,x-dx);
			  symmetry += -scalarprod;
			}
			weight_at_r[r] = symmetry/norm_at_r[r];
		  }
		}
		else { // only != 0
		  for (r=1; r<=max_radius; r++) {
			dy = r;
			for (dx=-r; dx<=r; dx++) {
			  scalarprod = gradx.at<double>(y+dy,x+dx)*gradx.at<double>(y-dy,x-dx)
				+ grady.at<double>(y+dy,x+dx)*grady.at<double>(y-dy,x-dx);
			  symmetry += - scalarprod * 
				theta(only*(dx*gradx.at<double>(y+dy,x+dx) + dy*grady.at<double>(y+dy,x+dx)));
			}
			dx = r;
			for (dy=-r+1; dy<r; dy++) {
			  scalarprod = gradx.at<double>(y+dy,x+dx)*gradx.at<double>(y-dy,x-dx)
				+ grady.at<double>(y+dy,x+dx)*grady.at<double>(y-dy,x-dx);
			  symmetry += - scalarprod * 
				theta(only*(dx*gradx.at<double>(y+dy,x+dx) + dy*grady.at<double>(y+dy,x+dx)));
			}
			weight_at_r[r] = symmetry/norm_at_r[r];
		  }
		}

		// find radius of maximum symmetry
		*R = 1;
		*S = weight_at_r[1];
		for (r=2; r<=max_radius; r++) {
		  if (weight_at_r[r] > *S) {
			*R = r;
			*S = weight_at_r[r];
		  }
		}
	  }
	}


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
	void symmetry_transform(const cv::Mat &grey, cv::Mat &result, cv::Mat &result_r, int radius, double alpha/*=0.5*/, int only/*=0*/, GradientNormalization *gradnorm/*=NULL*/)
	{
	  cv::Mat gradx, grady;
	  double gx,gy;
	  int x,y,r;

	  cv::Sobel(grey, gradx, CV_64F, 1, 0);
	  cv::Sobel(grey, grady, CV_64F, 0, 1);
	  result.create(grey.rows, grey.cols, CV_64F);
	  result_r.create(grey.rows, grey.cols, CV_64F);

	  // optionally transform gradient strength
	  if (gradnorm) {
		for (y=0; y<grey.rows; y++) {
		  for (x=0; x<grey.cols; x++) {
			if (gradnorm) {
			  gx = gradx.at<double>(y,x);
			  gy = grady.at<double>(y,x);
			  (*gradnorm)(gx,gy);
			  gradx.at<double>(y,x) = gx;
			  grady.at<double>(y,x) = gy;
			}
		  }
		}   
	  }

	  // compute symmetry weight
	  std::vector<double> norm_at_r(radius+1);
	  norm_at_r[0] = 1.0; // not needed
	  for (r=1; r<=radius; r++) norm_at_r[r] = pow(r,alpha);

	#if (NUM_THREADS > 1)
	  #pragma omp parallel num_threads(NUM_THREADS) shared(result, result_r)
	  #pragma omp for schedule(dynamic)
	#endif
	  for (int y=0; y<grey.rows; y++) {
		for (int x=0; x<grey.cols; x++) {
		  int R;
		  double S;
		  s_and_r_at_point(gradx, grady, &R, &S, radius, norm_at_r, x, y, only);
		  result.at<double>(y,x) = S;
		  result_r.at<double>(y,x) = (double)R;
		}
	  }

	  return;
	}

	//-----------------------------------------------------------------
	// Compute symmetry score S and radius R at point (x,y) with rectangular regions.
	// Arguments:
	//  grad(x|y) = gradient image (input)
	//  Rx        = x-component of best radius (output)
	//  Ry        = y-component of best radius (output)
	//  S         = score at best radius (output)
	//  radius    = maximum radius examined (input)
	//  norm_at_r = precalculated normalisation factors r^alpha (input)
	//  x,y       = point position (input)
	//  only      = 1 = only dark object, -1 = only light objects, 0 = all objects (input)
	//-----------------------------------------------------------------
	void s_and_r_at_point_rect(const cv::Mat &gradx, const cv::Mat &grady, int* Rx, int* Ry, double* S, int radius, const std::vector<double> &norm_at_r, int x, int y, int only=0)
	{
	  int rx,ry,max_radius;
	  double symmetry;
  
	  // adjust radius so that it does not extend beyond the image border
	  max_radius = radius;
	  if (y < max_radius)
		max_radius = y;
	  if (y >= gradx.rows - max_radius)
		max_radius = gradx.rows - y - 1;
	  if (x < max_radius)
		max_radius = x;
	  if (x >= gradx.cols - max_radius)
		max_radius = gradx.cols - x - 1;
	  cv::Mat weight_at_r(max_radius+1, max_radius+1, CV_64F);
	  //  weight_at_r.create(max_radius+1, max_radius+1, CV_64F);

	  // compute weight in window with radius max_radius
	  if (max_radius < 1) {
		*Rx = 0;
		*Ry = 0;
		*S = 0.0;
	  } else {
		if (only == 0) {
		  // center row with ry=1
		  ry = 1;
		  symmetry = - gradx.at<double>(y+ry,x)*gradx.at<double>(y-ry,x)
			- grady.at<double>(y+ry,x)*grady.at<double>(y-ry,x);
		  for (rx=1; rx<=max_radius; rx++) {
			symmetry += - gradx.at<double>(y+ry,x+rx)*gradx.at<double>(y-ry,x-rx)
			  - grady.at<double>(y+ry,x+rx)*grady.at<double>(y-ry,x-rx)
			  - gradx.at<double>(y,x+rx)*gradx.at<double>(y,x-rx)
			  - grady.at<double>(y,x+rx)*grady.at<double>(y,x-rx)
			  - gradx.at<double>(y-ry,x+rx)*gradx.at<double>(y+ry,x-rx)
			  - grady.at<double>(y-ry,x+rx)*grady.at<double>(y+ry,x-rx);
			weight_at_r.at<double>(ry,rx) = symmetry;
		  }
		  // use recursion formula for rest
		  for (ry=2; ry<=max_radius; ry++) {
			rx = 1;
			symmetry = weight_at_r.at<double>(ry-1,rx);
			symmetry += - gradx.at<double>(y+ry,x+rx)*gradx.at<double>(y-ry,x-rx)
			  - grady.at<double>(y+ry,x+rx)*grady.at<double>(y-ry,x-rx)
			  - gradx.at<double>(y+ry,x)*gradx.at<double>(y-ry,x)
			  - grady.at<double>(y+ry,x)*grady.at<double>(y-ry,x)
			  - gradx.at<double>(y+ry,x-rx)*gradx.at<double>(y-ry,x+rx)
			  - grady.at<double>(y+ry,x-rx)*grady.at<double>(y-ry,x+rx);
			weight_at_r.at<double>(ry,rx) = symmetry;
			for (rx=2; rx<=max_radius; rx++) {
			  symmetry += weight_at_r.at<double>(ry-1,rx) - weight_at_r.at<double>(ry-1,rx-1)
				- gradx.at<double>(y+ry,x+rx)*gradx.at<double>(y-ry,x-rx)
				- grady.at<double>(y+ry,x+rx)*grady.at<double>(y-ry,x-rx)
				- gradx.at<double>(y+ry,x-rx)*gradx.at<double>(y-ry,x+rx)
				- grady.at<double>(y+ry,x-rx)*grady.at<double>(y-ry,x+rx);
			  weight_at_r.at<double>(ry,rx) = symmetry;
			}
		  }
		}
		else { // only != 0
		  // center row with ry=1
		  ry = 1;
		  symmetry = - theta(only*grady.at<double>(y+ry,x)) *
			( gradx.at<double>(y+ry,x)*gradx.at<double>(y-ry,x)
			+ grady.at<double>(y+ry,x)*grady.at<double>(y-ry,x) );
		  for (rx=1; rx<=max_radius; rx++) {
			symmetry += - theta(only*(rx*gradx.at<double>(y+ry,x+rx) + ry*grady.at<double>(y+ry,x+rx))) *
			  ( gradx.at<double>(y+ry,x+rx)*gradx.at<double>(y-ry,x-rx)
				+ grady.at<double>(y+ry,x+rx)*grady.at<double>(y-ry,x-rx) )
			  - theta(only*rx*gradx.at<double>(y,x+rx)) *
			  ( gradx.at<double>(y,x+rx)*gradx.at<double>(y,x-rx)
				+ grady.at<double>(y,x+rx)*grady.at<double>(y,x-rx) )
			  - theta(only*(rx*gradx.at<double>(y-ry,x+rx) - ry*grady.at<double>(y-ry,x+rx))) *
			  ( gradx.at<double>(y-ry,x+rx)*gradx.at<double>(y+ry,x-rx)
				+ grady.at<double>(y-ry,x+rx)*grady.at<double>(y+ry,x-rx) );
			weight_at_r.at<double>(ry,rx) = symmetry;
		  }
		  // use recursion formula for rest
		  for (ry=2; ry<=max_radius; ry++) {
			rx = 1;
			symmetry = weight_at_r.at<double>(ry-1,rx);
			symmetry += - theta(only*(rx*gradx.at<double>(y+ry,x+rx) + ry*grady.at<double>(y+ry,x+rx))) *
			  ( gradx.at<double>(y+ry,x+rx)*gradx.at<double>(y-ry,x-rx)
				+ grady.at<double>(y+ry,x+rx)*grady.at<double>(y-ry,x-rx) )
			  - theta(only*ry*grady.at<double>(y+ry,x+rx)) *
			  ( gradx.at<double>(y+ry,x)*gradx.at<double>(y-ry,x)
				+ grady.at<double>(y+ry,x)*grady.at<double>(y-ry,x) )
			  - theta(only*(-rx*gradx.at<double>(y+ry,x+rx) + ry*grady.at<double>(y+ry,x+rx))) *
			  ( gradx.at<double>(y+ry,x-rx)*gradx.at<double>(y-ry,x+rx)
				+ grady.at<double>(y+ry,x-rx)*grady.at<double>(y-ry,x+rx) );
			weight_at_r.at<double>(ry,rx) = symmetry;
			for (rx=2; rx<=max_radius; rx++) {
			  symmetry += weight_at_r.at<double>(ry-1,rx) - weight_at_r.at<double>(ry-1,rx-1)
				- theta(only*(rx*gradx.at<double>(y+ry,x+rx) + ry*grady.at<double>(y+ry,x+rx))) *
				( gradx.at<double>(y+ry,x+rx)*gradx.at<double>(y-ry,x-rx)
				  + grady.at<double>(y+ry,x+rx)*grady.at<double>(y-ry,x-rx) )
				- theta(only*(-rx*gradx.at<double>(y+ry,x+rx) + ry*grady.at<double>(y+ry,x+rx))) *
				( gradx.at<double>(y+ry,x-rx)*gradx.at<double>(y-ry,x+rx)
				  + grady.at<double>(y+ry,x-rx)*grady.at<double>(y-ry,x+rx) );
			  weight_at_r.at<double>(ry,rx) = symmetry;
			}
		  }
		}
    
		/*>>>>>>>>> too slow => do it inline at later point
		// normalize weights
		for (ry=1; ry<=max_radius; ry++) {
		  for (rx=1; rx<=max_radius; rx++) {
			weight_at_r.at<double>(ry,rx) /= norm_at_r[rx+ry];
		  }
		}
		<<<<<<<<<<< */

		// find radius of maximum symmetry
		*Rx = 1;
		*Ry = 1;
		*S = weight_at_r.at<double>(1,1)/norm_at_r[2];
		for (ry=1; ry<=max_radius; ry++) {
		  for (rx=1; rx<=max_radius; rx++) {
			if (weight_at_r.at<double>(ry,rx)/norm_at_r[rx+ry] > *S) {
			  *Rx = rx;
			  *Ry = ry;
			  *S = weight_at_r.at<double>(ry,rx)/norm_at_r[rx+ry];
			}
		  }
		}
	  }
	}

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
	void symmetry_transform_rect(const cv::Mat &grey, cv::Mat &result, cv::Mat &result_rx, cv::Mat &result_ry, int radius, double alpha/*=0.5*/, int only/*=0*/, GradientNormalization *gradnorm/*=NULL*/)
	{
	  cv::Mat gradx, grady;
	  double gx,gy;
	  int x,y,r;

	  cv::Sobel(grey, gradx, CV_64F, 1, 0);
	  cv::Sobel(grey, grady, CV_64F, 0, 1);
	  result.create(grey.rows, grey.cols, CV_64F);
	  result_rx.create(grey.rows, grey.cols, CV_64F);
	  result_ry.create(grey.rows, grey.cols, CV_64F);

	  // optionally transform gradient strength
	  if (gradnorm) {
		for (y=0; y<grey.rows; y++) {
		  for (x=0; x<grey.cols; x++) {
			if (gradnorm) {
			  gx = gradx.at<double>(y,x);
			  gy = grady.at<double>(y,x);
			  (*gradnorm)(gx,gy);
			  gradx.at<double>(y,x) = gx;
			  grady.at<double>(y,x) = gy;
			}
		  }
		}   
	  }

	  // compute symmetry weight
	  std::vector<double> norm_at_r(2*radius+1);
	  norm_at_r[0] = 1.0; // not needed
	  for (r=1; r<=2*radius; r++) norm_at_r[r] = pow(r,alpha);

	#if (NUM_THREADS > 1)
	  #pragma omp parallel num_threads(NUM_THREADS) shared(result, result_rx, result_ry)
	  #pragma omp for schedule(dynamic)
	#endif
	  for (int y=0; y<grey.rows; y++) {
		for (int x=0; x<grey.cols; x++) {
		  int Rx, Ry;
		  double S;
		  s_and_r_at_point_rect(gradx, grady, &Rx, &Ry, &S, radius, norm_at_r, x, y, only);
		  result.at<double>(y,x) = S;
		  result_rx.at<double>(y,x) = (double)Rx;
		  result_ry.at<double>(y,x) = (double)Ry;
		}
	  }

	  return;
	}

	//-----------------------------------------------------------------
	// Find candidate symmetry centers in symmetry image as local
	// maxima of the symmetry score. Returns the number of points found.
	// Arguments:
	//  symmetry        = symmetry transform image filename (input)
	//  candidatepoints = symmetry candidate points (output)
	//  windowsize      = window size for local maximum search (input)
	//-----------------------------------------------------------------
	int points_local_maxima(const cv::Mat &symmetry, SymmetryPointVector &candidatepoints, int windowsize /*=3*/)
	{
		if (windowsize%2 == 0)
		{
			fprintf(stderr, "odd number for windowsize parameter expected\n");
			return -1;
		}
		candidatepoints.clear();
		double minv, maxv;
		cv::minMaxLoc(symmetry, &minv, &maxv);
		int s = windowsize/2;
		int maxindex=-1;
		int maxi=s*windowsize+s;
		for (int y=s; y<=symmetry.rows-s-1; y++)
			for (int x=s; x<= symmetry.cols-s-1; x++)
			{
				double maxivalue=minv;			
				for (int yy=0; yy<=2*s; yy++)
					for (int xx=0; xx<=2*s; xx++)
					{
						double d = symmetry.at<double>(y+yy-s,x+xx-s);
						if (d > maxivalue)
						{
							maxindex=yy*windowsize+xx;
							maxivalue=d;
						}
					}
				if ((maxindex == maxi) && (maxivalue > 0.0))
				{
					candidatepoints.push_back(SymmetryPoint(cv::Point(x,y),symmetry.at<double>(y,x)));
				}
			}

		return candidatepoints.size();
	}


	//-----------------------------------------------------------------
	// Compute the edge dispersion at the given point.
	// Arguments:
	//  gradx    = x-component of gradient image (input)
	//  grady    = y-component of gradient image (input)
	//  point    = the point to be examined (input)
	//  radius   = the radius of the window to be examined (input)
	//  cutoff   = ignore an image border of width cutoff (input)
	//-----------------------------------------------------------------
	double edge_dispersion_cd(const cv::Mat &gradx, const cv::Mat &grady, const cv::Point &point, int radius /* = 3*/, int cutoff /* = 0*/)
	{
	  //  when G_i are the gradient vectors inside the window,
	  //  then we compute
	  //
	  //    1 - R / (n max{ ||G_i|| })  with  R = || \sum_i G_i ||
	  //
	  //  The problem is that we do not know how to mirror the vectors
	  //  G_i (if we do not do this, the dispersion is always maximal).
	  //  We therefore first mirror negative y-values, and then negative
	  //  x-values, and eventaully choose the smaller dispersion of both.

	  int x,y,left,right,top,bot,n;
	  double G2, maxG2; // square of absolute value
	  double gx, gy;    // gradient values
	  double dispersion1, dispersion2;
	  double dispersion1_x = 0.0;
	  double dispersion1_y = 0.0;
	  double dispersion2_x = 0.0;
	  double dispersion2_y = 0.0;

	  left  = std::max(cutoff, point.x - radius);
	  right = std::min(gradx.cols - cutoff - 1, point.x + radius);
	  top   = std::max(cutoff, point.y - radius);
	  bot   = std::min(gradx.rows - cutoff - 1, point.y + radius);

	  maxG2 = 0.0; n=0;
	  for (y = top; y <= bot; y++) {
		for (x = left; x <= right; x++) {
		  n++;
		  gx = gradx.at<double>(y,x); gy = grady.at<double>(y,x);
		  G2 = gx*gx + gy*gy;
		  if (G2 > maxG2)
			maxG2 = G2;
		  if (gx < 0) {
			dispersion1_x -= gx; dispersion1_y -= gy;
		  } else {
			dispersion1_x += gx; dispersion1_y += gy;        
		  }
		  if (gy < 0) {
			dispersion2_x -= gx; dispersion2_y -= gy;
		  } else {
			dispersion2_x += gx; dispersion2_y += gy;        
		  }
		}
	  }

	  // apply formula for dispersion
	  if (maxG2 > 0.0) {
		dispersion1 = 1 - sqrt((dispersion1_x*dispersion1_x + dispersion1_y*dispersion1_y) / maxG2)/n;
		dispersion2 = 1 - sqrt((dispersion2_x*dispersion2_x + dispersion2_y*dispersion2_y) / maxG2)/n;
	  } else {
		return 0.0;
	  }

	  // choose the smaller dispersion
	  return std::min(dispersion1, dispersion2);
	}


	//-----------------------------------------------------------------
	// Compute the edge directedness as the frequency of the most frequent
	// gradient direction.
	// Arguments:
	//  gradx    = x-component of gradient image (input)
	//  grady    = y-component of gradient image (input)
	//  point    = the point to be examined (input)
	//  radius   = the radius of the window to be examined (input)
	//  cutoff   = ignore an image border of width cutoff (input)
	//-----------------------------------------------------------------
	double edge_directedness_cd(const cv::Mat &gradx, const cv::Mat &grady, const cv::Point &point, int radius /*= 3*/, int nbins /*= 16*/, int cutoff /*= 0*/)
	{
	  int x,y,left,right,top,bot,i;
	  double G, sumG;  // absolute values of gradient
	  double gx, gy;   // gradient values
	  double angle, maxhist, pi_n;
	  double* hist = new double[nbins];
  
	  pi_n = M_PI / nbins;
	  left  = std::max(cutoff, point.x - radius);
	  right = std::min(gradx.cols - cutoff - 1, point.x + radius);
	  top   = std::max(cutoff, point.y - radius);
	  bot   = std::min(gradx.rows - cutoff - 1, point.y + radius);

	  sumG = 0.0;
	  for (i=0; i<nbins; i++) hist[i] = 0.0;
	  for (y = top; y <= bot; y++) {
		for (x = left; x <= right; x++) {
		  gx = gradx.at<double>(y,x); gy = grady.at<double>(y,x);
		  G = sqrt(gx*gx + gy*gy);
		  sumG += G;
		  angle = atan2(gy,gx);
		  // shift angle bins by pi/n so that zero etc. lie in middle of bin
		  if (angle < -pi_n) angle += 2*M_PI;
		  i = (int)((angle+pi_n)*nbins/(2*M_PI));
		  hist[i] += G;
		  //hist[(i+4)%8] += G;
		}
	  }
	  maxhist = 0.0;
	  for (i=0; i<nbins; i++) 
		if (hist[i] > maxhist) maxhist = hist[i];

	  delete[] hist;

	  if (sumG == 0.0)
		return 0.0;
	  else
		return (maxhist/sumG);
	  //return (maxhist/(2*sumG));
	}

	//-----------------------------------------------------------------
	// Hu moments of region around point.
	// Arguments:
	//  image     = complete image (input)
	//  point     = the point to be examined (input)
	//  radius    = the radius of the window to be examined (input)
	//  humoments = the computed moments (output, must be preallocated of size 8)
	//-----------------------------------------------------------------
	void humoments_cd(double* humoments, const cv::Mat &image, const cv::Point &point, int radius /*= 3*/, int cutoff /*= 0*/)
	{
	  int left,right,top,bot;
	  cv::Mat roi, roigrey;
	  cv::Moments m;

	  left  = std::max(cutoff, point.x - radius);
	  right = std::min(image.cols - cutoff - 1, point.x + radius);
	  top   = std::max(cutoff, point.y - radius);
	  bot   = std::min(image.rows - cutoff - 1, point.y + radius);

	  roi = image(cv::Rect(left,top,right-left+1,bot-top+1));
	  float2grey(roi, roigrey, false); // normalize values in region
	  m = cv::moments(roigrey);
	  cv::HuMoments(m, humoments);
	}

	//-----------------------------------------------------------------
	// Bamieh moment invariants (BMI) of region around point.
	// Arguments:
	//  image     = complete image (input)
	//  point     = the point to be examined (input)
	//  radius    = the radius of the window to be examined (input)
	//  bmi       = the computed moments (output, must be preallocated of size 2)
	//-----------------------------------------------------------------
	void bmimoments_cd(double* bmi, const cv::Mat &image, const cv::Point &point, int radius /*= 3*/, int cutoff /*= 0*/)
	{
	  int left,right,top,bot;
	  cv::Mat roi, roigrey;
	  cv::Moments m;

	  left  = std::max(cutoff, point.x - radius);
	  right = std::min(image.cols - cutoff - 1, point.x + radius);
	  top   = std::max(cutoff, point.y - radius);
	  bot   = std::min(image.rows - cutoff - 1, point.y + radius);

	  roi = image(cv::Rect(left,top,right-left+1,bot-top+1));
	  float2grey(roi, roigrey, false); // normalize values in region
	  m = cv::moments(roigrey);
	  bmi[0] = m.nu02*m.nu20 - m.nu11*m.nu11;
	  bmi[1] = (m.nu03*m.nu30 - m.nu21*m.nu12)*(m.nu03*m.nu30 - m.nu21*m.nu12)
		- 4*(m.nu03*m.nu12 - m.nu21*m.nu21)*(m.nu30*m.nu21 - m.nu12*m.nu12);
	}

	//-----------------------------------------------------------------
	// Compute the ratio of the the two eigenvalues of teh covariance matrix
	// of the window with the given radius around point.
	// Arguments:
	//  image     = complete image (input)
	//  point     = the point to be examined (input)
	//  radius    = the radius of the window to be examined (input)
	//  cutoff   = ignore an image border of width cutoff (input)
	//-----------------------------------------------------------------
	double coveigenratio_cd(const cv::Mat &image, const cv::Point &point, int radius /*= 3*/, int cutoff /*= 0*/)
	{
	  int x,y,left,right,top,bot,i,j;
	  cv::Mat cov(cv::Size(2,2), CV_64F);
	  cv::Mat eigenvalues;
	  double sumf;  // normalization factor
	  double dy,dx;
	  double value = 0.0;

	  left  = std::max(cutoff, point.x - radius);
	  right = std::min(image.cols - cutoff - 1, point.x + radius);
	  top   = std::max(cutoff, point.y - radius);
	  bot   = std::min(image.rows - cutoff - 1, point.y + radius);

	  // compute covariance matrix
	  for (i=0; i<2; i++)
		for (j=0; j<2; j++)
		  cov.at<double>(i,j) = 0.0;
	  sumf = 0.0;
	  for (y = top; y <= bot; y++) {
		for (x = left; x <= right; x++) {
		  if (image.type() == CV_64F) {
			value = image.at<double>(y,x);
		  }
		  else if (image.type() == CV_32F) {
			value = (double)image.at<float>(y,x);
		  }
		  else if (image.type() == CV_8U) {
			value = (double)image.at<uchar>(y,x);
		  }
		  else if (image.type() == CV_16U) {
			value = (double)image.at<ushort>(y,x);
		  }
		  sumf += value;
		  dx = x - point.x; dy = y - point.y;
		  cov.at<double>(0,0) += value*dx*dx;
		  cov.at<double>(0,1) += value*dx*dy;
		  cov.at<double>(1,0) += value*dy*dx;
		  cov.at<double>(1,1) += value*dy*dy;
		}
	  }
	  for (i=0; i<2; i++)
		for (j=0; j<2; j++)
		  cov.at<double>(i,j) /= sumf;

	  // determine eigenvalue ratio
	  cv::eigen(cov, eigenvalues);

	  //cv::Size s = eigenvalues.size();
	  //printf("Size of eigenvalues = %ix%i\n", s.width, s.height);
	  //printf("Eigenvalues = (%f,%f)", eigenvalues.at<double>(0,0), eigenvalues.at<double>(1,0));

	  if (eigenvalues.at<double>(0,0) == 0.0 && eigenvalues.at<double>(1,0) == 0.0)
		return 0.0;
	  if (fabs(eigenvalues.at<double>(0,0)) > fabs(eigenvalues.at<double>(1,0)))
		return (fabs(eigenvalues.at<double>(1,0)) / fabs(eigenvalues.at<double>(0,0)));
	  else
		return (fabs(eigenvalues.at<double>(0,0)) / fabs(eigenvalues.at<double>(1,0)));
	}


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
	double antiparallelity_rpf(const cv::Mat &gradx, const cv::Mat &grady, const cv::Point &point, int radius, double &max_frequency, bool weighted /*= false*/, double count_threshold /*= 0.1*/, int nbins /*= 8*/)
	{
	  double scalarprod, result;
	  int x,y,dx,dy,r,i,count;
	  double v1,v2;  // norms of the opposite gradients
	  double angle;  // angle between gradients
	  double sumG;   // normalization factor (sum of all gradient norms)
	  // threshold when angle between two vectors can be considered antiparallel
	  double parallelity_threshold = -0.975;
	  // pi divided by number of bins (shift for histogram bins)
	  double pi_n = M_PI / nbins;

	  // compute antiparallelity
	  std::vector<double> bin(nbins);
	  for (i=0; i<nbins; i++) bin[i] = 0.0;
	  sumG = 0.0;
	  x = point.x;
	  y = point.y;
	  for (r=1; r<=radius; r++) {
		dy = r;
		for (dx=-r; dx<=r; dx++) {
		  scalarprod = gradx.at<double>(y+dy,x+dx)*gradx.at<double>(y-dy,x-dx)
			+ grady.at<double>(y+dy,x+dx)*grady.at<double>(y-dy,x-dx);
		  if (scalarprod < 0.0) { // only when in opposite directions
			v1 = sqrt(gradx.at<double>(y+dy,x+dx)*gradx.at<double>(y+dy,x+dx) +
					  grady.at<double>(y+dy,x+dx)*grady.at<double>(y+dy,x+dx));
			v2 = sqrt(gradx.at<double>(y-dy,x-dx)*gradx.at<double>(y-dy,x-dx) +
					  grady.at<double>(y-dy,x-dx)*grady.at<double>(y-dy,x-dx));
			if (scalarprod/(v1*v2) < parallelity_threshold) {
			  angle = atan2(grady.at<double>(y+dy,x+dx),gradx.at<double>(y+dy,x+dx));
			  // shift angle bins by pi/n so that zero etc. lie in middle of bin
			  if (angle < -pi_n) angle += 2*M_PI;
			  i = (int)((angle+pi_n)*nbins/(2*M_PI));
			  if (weighted) {
				bin[i] += v1;
				bin[(i+nbins/2)%nbins] += v2;
				sumG += v1+v2;
			  } else {
				bin[i]++;
				bin[(i+nbins/2)%nbins]++;
				sumG += 2.0;
			  }
			}
		  }
		}
		dx = r;
		for (dy=-r+1; dy<r; dy++) {
		  scalarprod = gradx.at<double>(y+dy,x+dx)*gradx.at<double>(y-dy,x-dx)
			+ grady.at<double>(y+dy,x+dx)*grady.at<double>(y-dy,x-dx);
		  if (scalarprod < 0.0) { // only when in opposite directions
			v1 = sqrt(gradx.at<double>(y+dy,x+dx)*gradx.at<double>(y+dy,x+dx) +
					  grady.at<double>(y+dy,x+dx)*grady.at<double>(y+dy,x+dx));
			v2 = sqrt(gradx.at<double>(y-dy,x-dx)*gradx.at<double>(y-dy,x-dx) +
					  grady.at<double>(y-dy,x-dx)*grady.at<double>(y-dy,x-dx));
			if (scalarprod/(v1*v2) < parallelity_threshold) {
			  angle = atan2(grady.at<double>(y+dy,x+dx),gradx.at<double>(y+dy,x+dx));
			  // shift angle bins by pi/n so that zero etc. lie in middle of bin
			  if (angle < -pi_n) angle += 2*M_PI;
			  i = (int)((angle+pi_n)*nbins/(2*M_PI));
			  if (weighted) {
				bin[i] += v1;
				bin[(i+nbins/2)%nbins] += v2;
				sumG += v1+v2;
			  } else {
				bin[i]++;
				bin[(i+nbins/2)%nbins]++;
				sumG += 2.0;
			  }
			}
		  }
		}
	  }

	  for (i=0; i<nbins; i++) bin[i] = bin[i]/sumG;
	  count = 0;
	  for (i=0; i<nbins; i++)
		if (bin[i] > count_threshold) count++;

	  max_frequency = *std::max_element(bin.begin(), bin.end());
	  result = count/((double)nbins);
	  return result;
	}

	//-----------------------------------------------------------------
	// Returns the skeleton starting at x until the score value
	// falls below the given percentage of S(x).
	// Arguments:
	//  x          = starting point (input)
	//  S          = symmetry transform (input)
	//  skeleton   = the points of the skeleton (output)
	//  percentage = threshold at which the skeleton is stopped (input)
	//  maxlength  = maximum length of the skeleton (input)
	//-----------------------------------------------------------------
	void get_skeleton(const cv::Point &p, const cv::Mat &S, PointVector* skeleton, double percentage, int maxlength)
	{
	  int r,i,nn;
	  double maxval;
	  cv::Point maxpoint, startpoint;
	  PointVector neighbors;
	  int maxx = S.cols - 1;
	  int maxy = S.rows - 1;
	  double threshold = percentage*S.at<double>(p);
	  skeleton->clear();
	  skeleton->push_back(p);

	  // find startpoint in first direction
	  nn = get_neighbors(p, &neighbors, maxx, maxy);
	  if (!nn) return;
	  maxpoint = neighbors[0];
	  maxval = S.at<double>(maxpoint);
	  for (i=1; i<nn; i++) {
		if (S.at<double>(neighbors[i]) > maxval) {
		  maxpoint = neighbors[i]; maxval = S.at<double>(maxpoint);
		}
	  }
	  startpoint = maxpoint; // remember for opposite direction later
	  if (maxval > threshold)
		skeleton->push_back(maxpoint);

	  // follow skeleton in first direction
	  for (r=2; (r<maxlength) && (maxval>threshold); r++) {
		nn = get_neighbors(maxpoint, &neighbors, maxx, maxy);
		if (!nn) break;
		//maxval = std::numeric_limits<double>::lowest();
		maxval = -1000;
		for (i=0; i<nn; i++) {
		  if ((abs(neighbors[i].x-p.x)==r || abs(neighbors[i].y-p.y)==r)
			  && (S.at<double>(neighbors[i]) > maxval)) {
			maxpoint = neighbors[i]; maxval = S.at<double>(maxpoint);
		  }
		}
		if (maxval > threshold) {
		  skeleton->push_back(maxpoint);
		}
	  }

	  // look for startpoint near mirrored position
	  maxpoint = cv::Point(2*p.x-startpoint.x, 2*p.y-startpoint.y);
	  maxval = S.at<double>(maxpoint);
	  nn = get_neighbors(maxpoint, &neighbors, maxx, maxy);
	  if (!nn) return;
	  for (i=0; i<nn; i++) {
		if ((abs(neighbors[i].x-p.x)==1 || abs(neighbors[i].y-p.y)==1)
			&& (S.at<double>(neighbors[i]) > maxval)) {
		  maxpoint = neighbors[i]; maxval = S.at<double>(maxpoint);
		}
	  }
	  if (maxval > threshold)
		skeleton->push_back(maxpoint);

	  // follow skeleton in opposite direction
	  for (r=2; (r<maxlength) && (maxval>threshold); r++) {
		nn = get_neighbors(maxpoint, &neighbors, maxx, maxy);
		if (!nn) break;
		//maxval = std::numeric_limits<double>::lowest();
		maxval = -1000;
		for (i=0; i<nn; i++) {
		  if ((abs(neighbors[i].x-p.x)==r || abs(neighbors[i].y-p.y)==r)
			  && (S.at<double>(neighbors[i]) > maxval)) {
			maxpoint = neighbors[i]; maxval = S.at<double>(maxpoint);
		  }
		}
		if (maxval > threshold) {
		  skeleton->push_back(maxpoint);
		}
	  }

	  return;
	}


	//-----------------------------------------------------------------
	// Returns the ratio skeleton_length/symmetry_region_size
	// Arguments:
	//  p        = starting point (input)
	//  skeleton = the points of the skeleton (input)
	//  rx       = x-component of cricumradius
	//  ry       = y-component of cricumradius
	//-----------------------------------------------------------------
	double skeleton_size(const cv::Point &x, const PointVector &skeleton, double rx, double ry)
	{
	  double result = sqrt(rx*rx+ry*ry);
	  result = double(skeleton.size()) / result;
	  if (result > 1.0)
		return 1.0;
	  else
		return result;
	}

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
	double diagonal_sum(const cv::Point &p, const cv::Mat &gradx, const cv::Mat &grady, double rx, double ry)
	{
	  double sum = 0.0;
	  double abs1, abs2, abs3, abs4, gx1, gy1, gx2, gy2, gx3, gy3, gx4, gy4;
	  for (int dx=1; dx <= rx; dx++) {
		for (int dy=1; dy <= ry; dy++) {
		  gx1 = gradx.at<double>(p.y+dy, p.x+dx);
		  gy1 = grady.at<double>(p.y+dy, p.x+dx);
		  abs1 = sqrt(gx1*gx1 + gy1*gy1) + 0.000001;
		  gx2 = gradx.at<double>(p.y+dy, p.x-dx);
		  gy2 = grady.at<double>(p.y+dy, p.x-dx);
		  abs2 = sqrt(gx2*gx2 + gy2*gy2) + 0.000001;
		  gx3 = gradx.at<double>(p.y-dy, p.x-dx);
		  gy3 = grady.at<double>(p.y-dy, p.x-dx);
		  abs3 = sqrt(gx3*gx3 + gy3*gy3) + 0.000001;
		  gx4 = gradx.at<double>(p.y-dy, p.x+dx);
		  gy4 = grady.at<double>(p.y-dy, p.x+dx);
		  abs4 = sqrt(gx4*gx4 + gy4*gy4) + 0.000001;
		  sum += (gx1*gx2+gy1*gy2)/(abs1*abs2) + (gx2*gx3+gy2*gy3)/(abs2*abs3) +
			(gx3*gx4+gy3*gy4)/(abs3*abs4) + (gx4*gx1+gy4*gy1)/(abs4*abs1);
		}
	  }
	  return fabs(sum/(4*(rx+ry)));
	}


	//-----------------------------------------------------------------
	// Returns true, when QDA on all four features finds axial symmetry
	// Input arguments:
	//  edge_dir  = edge directedness
	//  skel_size = skeleton size
	//  anti_par  = antiparallelity
	//  cov_ratio = covariance eigenratio
	//-----------------------------------------------------------------
	bool axial_qda(double edge_dir, double skel_size, double anti_par, double cov_ratio)
	{
	  // values measured from test data set
	  const double ma[4] = {0.4334325, 0.4974716, 0.4417170, 0.4352612};
	  const double mr[4] = {0.18586523, 0.06537124, 0.23017472, 0.71357609};
	  const double Da = -15.03338;
	  const double Dr = -21.49041;
	  const double Sa[4][4] = {{-9.392123, 3.145148, 7.8727492, 0.8677436},
							   {0.000000, -2.436357, 0.3521709, 0.1780096},
							   {0.000000,  0.0000, -18.5275062, 3.1907446},
							   {0.000000,  0.000000, 0.0000000, 4.3364655}};
	  const double Sr[4][4] = {{14.31157, -6.191231, 15.545277, -6.7814399},
							   {0.00000,  20.318334,  0.375387, -0.9648399},
							   {0.00000,   0.00000, -20.595436, -3.8729805},
							   {0.00000,   0.00000,   0.000000, -7.7488568}};
	  double ga, gr, tmp;
	  double x[4];
	  size_t j,k;
	  // discriminant function for axial
	  x[0] = edge_dir  - ma[0];
	  x[1] = skel_size - ma[1];
	  x[2] = anti_par  - ma[2];
	  x[3] = cov_ratio - ma[3];
	  ga = -Da;
	  for (j=0; j<4; j++) {
		tmp = 0.0;
		for (k=0; k<4; k++) {
		  tmp += x[k]*Sa[k][j];
		}
		ga = ga - tmp*tmp;
	  }
	  // discriminant function for rotational
	  x[0] = edge_dir  - mr[0];
	  x[1] = skel_size - mr[1];
	  x[2] = anti_par  - mr[2];
	  x[3] = cov_ratio - mr[3];
	  gr = -Dr;
	  for (j=0; j<4; j++) {
		tmp = 0.0;
		for (k=0; k<4; k++) {
		  tmp += x[k]*Sr[k][j];
		}
		gr = gr - tmp*tmp;
	  }
	  // decision
	  //printf("%5.4f, %5.4f, %5.3f, %5.4f => ga=%5.4f  gr=%5.4f\n", edge_dir, skel_size, anti_par, cov_ratio, ga, gr);
	  return (ga > gr);
	}
}