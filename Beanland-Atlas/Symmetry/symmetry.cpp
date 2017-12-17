//
// Author:  Christoph Dalitz; Modified by Jeffrey M. Ede
// Version: 2.0a from 2014/05/23; Modified 17/12/2017
//
// Computes the symmetry transformation after
// VISAPP 2013 paper by Dalitz, Pohle-Froehlich, and Bolten
// and return the maximum rotational symmetry point
//

#include <make_defs.h>
#include <stdio.h>

#ifdef _WIN32 //Windows
#include <io.h>
#else //Linux
#include <unistd.h>
#endif

#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "symmetry_transform.h"
#include <array>

#define TRACEPREFIX "trace-"

namespace sym
{
	/*
	**Finds center of symmetry of an image using gradient-based methods. More information can be found in Christoph Dalitz's
	**papers
	**Required input parameters:-
	**img: cv::Mat, OpenCV matrix to find the center of symmetry of
	**opt_r: int, The maximum radius of the symmetry that will be searched for
	**imgmask: cv::Mat, OpenCV mask matrix in which black indicates background to be ignored.
	**Beware that the mask is enlarged by the radius => use mask images only in combination with small r!
	**Optional input parameters:-
	**opt_alpha: double, Exponent for normalisation of symmetry score with r^alpha [0.5]
	**opt_rect: bool, if false do not try rectangular regions; only squares (rx=ry)
	**opt_axial: int, criterion for sorting out axial symmetries
	**    0: edge_dir >= 0.27 (VISAPP'13 paper)
	**    1: skel_size > 0.46 or edge_dir >= 0.27
	**    2: edge_dir >= 0.285 or anti_par > 0.287 (decision tree)
	**    3: QDA on all four features (VISAPP'15 paper, default)
	**opt_only: int, Specify whether to only look for symmetries in dark (or light) objects
	**    -1: light
	**     0: Look everywhere
	**     1: dark
	**opt_outfile: char*, File to output input image with the output rectangle drawn on it to show the symmetry
	**opt_log: bool, If true, transform gradient strength as log(1+|G|)
	**opt_trace: bool, If true, write trace of interim images
	**opt_trace_greytrans: int, same as -trace, but also specifies the transformation from float to grey:"
	**    0 = linear transformation over entire range
	**    1 = linear transformation over only positive range
	**    2 = like 1, but with subsequent histogram equalization
	**    3 = equalization over entire float range
	**Returns: 
	**std::array<int, 4>, Array describing the center of symmetry {maxpoint x, maxpoint y, radius x, radius y}
	**Note:
	**This functionality was originally written by Christoph Dalitz and his paper(s) should be cited when using this 
	*/
	std::array<int, 4> symmetry(cv::Mat img, int opt_r, cv::Mat imgmask, double opt_alpha, bool opt_rect, int opt_axial,
		int opt_only, char* opt_outfile, bool opt_log, bool opt_trace, int opt_trace_greytrans)
	{
	  cv::Mat grey, bgr, imgmask, greymask, result, result_rx, result_ry, gradx, grady;
	  double edge_dir, anti_par, skel_size, cov_ratio;
	  int rx,ry,i;
	  cv::Point maxpoint(0,0);

	  //Check that radius is greater than zero
	  if (opt_r <= 0) {
		  fprintf(stderr, "Radius must be greater than zero");
		  return {-1, 1, 0, 0};
	  }

	  //Process img and call symmetry transform
	  if (!img.data) {
		fprintf(stderr, "Image is empty\n");
		return {-1, 1, 0, 0};;
	  }
	  cv::cvtColor(img,grey,CV_RGB2GRAY);
	  //cv::GaussianBlur(grey,grey,cv::Size(3,3),1);
	  if (opt_log) {
		GradientNormalization gradnorm;
		if (opt_rect) {
		  symmetry_transform_rect(grey, result, result_rx, result_ry, opt_r, opt_alpha, opt_only, &gradnorm);
		} else {
		  symmetry_transform(grey, result, result_rx, opt_r, opt_alpha, opt_only, &gradnorm);
		}
	  } else {
		if (opt_rect) {
		  symmetry_transform_rect(grey, result, result_rx, result_ry, opt_r, opt_alpha, opt_only);
		} else {
		  symmetry_transform(grey, result, result_rx, opt_r, opt_alpha, opt_only);
		}
	  }

	  // reset all points that touch mask image to zero
	  if (imgmask.data) {
		if (imgmask.rows != result.rows || imgmask.cols != result.cols) {
		  fprintf(stderr, "Mask image must have same size as input image\n");
		  return {-1, 1, 0, 0};;
		}
		cv::cvtColor(imgmask,greymask,CV_RGB2GRAY);
		int rx,ry;
		for (int y=0; y<greymask.rows; y++) {
		  for (int x=0; x<greymask.cols; x++) {
			if (greymask.at<uchar>(y,x) == 0) {
			  result.at<double>(y,x) = 0.0;
			}
			else {
			  //ry = rx = result_rx.at<double>(y,x);
			  //if (opt_rect) ry = result_ry.at<double>(y,x);
			  ry = rx = opt_r;
			  for (int yy=std::max(0,y-ry); yy<std::min(result.rows,y+ry+1); yy++) {
				for (int xx=std::max(0,x-rx); xx<std::min(result.cols,x+rx+1); xx++) {
				  if (greymask.at<uchar>(yy,xx) == 0) {
					result.at<double>(y,x) = 0.0;
					goto endofloop;
				  }
				}
			  }
		  endofloop: {};
			}
		  }
		}
	  }

	  if (opt_trace) {
		float2grey(result, grey, opt_trace_greytrans);
		cv::imwrite(TRACEPREFIX "symmetry.png", grey);
	  }

	  // find all local maxima and sort them by symmetry score
	  // BEWARE: sorting is in ascending order!
	  SymmetryPointVector sp;
	  if (0 >= points_local_maxima(result, sp)) {
		fputs("No symmetry points found\n", stderr);
		return {-1, -2, 0, 0};;
	  }
	  std::sort(sp.begin(), sp.end());

	  // find first candidate point that matches rotational criterion
	  cv::Sobel(result, gradx, CV_64F, 1, 0);
	  cv::Sobel(result, grady, CV_64F, 0, 1);
	  if (opt_trace) {
		float2bgr(result, bgr, opt_trace_greytrans);
	  }
	  for (i=(int)sp.size()-1; i>=0; i--) {
		bool isaxial = false;
		ry = rx = result_rx.at<double>(sp[i].point);
		if (opt_rect) ry = result_ry.at<double>(sp[i].point);

		// classifiction axial versus rotational
		if (opt_axial == 0) {
		  edge_dir = edge_directedness_cd(gradx, grady, sp[i].point, 3, 16);
		  isaxial = (edge_dir >= 0.27);
		}
		else if (opt_axial == 1) {
		  edge_dir = edge_directedness_cd(gradx, grady, sp[i].point, 3, 16);
		  PointVector skeleton;
		  get_skeleton(sp[i].point, result, &skeleton, 0.5, opt_r);
		  skel_size = skeleton_size(sp[i].point, skeleton, rx, ry);
		  isaxial = ((skel_size > 0.46) || (edge_dir >= 0.27));
		}
		else if (opt_axial == 2) {
		  edge_dir = edge_directedness_cd(gradx, grady, sp[i].point, 3, 16);
		  antiparallelity_rpf(gradx, grady, sp[i].point, std::min(rx,ry), anti_par);
		  isaxial = ((anti_par > 0.287) || (edge_dir >= 0.285));
		}
		else {
		  edge_dir = edge_directedness_cd(gradx, grady, sp[i].point, 3, 16);
		  PointVector skeleton;
		  get_skeleton(sp[i].point, result, &skeleton, 0.5, opt_r);
		  skel_size = skeleton_size(sp[i].point, skeleton, rx, ry);
		  antiparallelity_rpf(gradx, grady, sp[i].point, std::min(rx,ry), anti_par);
		  cov_ratio = coveigenratio_cd(result, sp[i].point, 3);
		  isaxial = axial_qda(edge_dir, skel_size, anti_par, cov_ratio);
		}

		if (!isaxial) {
		  maxpoint = sp[i].point;
		  if (opt_trace) {
			bgr.at<cv::Vec3b>(sp[i].point) = cv::Vec3b(0,255,0);
			ry = rx = (int)result_rx.at<double>(sp[i].point);
			if (opt_rect) {
			  ry = (int)result_ry.at<double>(sp[i].point);
			}
			cv::rectangle(bgr,cv::Point(sp[i].point.x-rx,sp[i].point.y-ry),cv::Point(sp[i].point.x+rx,sp[i].point.y+rx),cv::Scalar(0,255,0));
			//cv::rectangle(bgr,cv::Point(sp[i].point.x-rx-1,sp[i].point.y-ry-1),cv::Point(sp[i].point.x+rx+1,sp[i].point.y+rx+1),cv::Scalar(0,255,0));
		  }
		  break;
		} else {
		  if (opt_trace)
			bgr.at<cv::Vec3b>(sp[i].point) = cv::Vec3b(0,0,255);
			ry = rx = (int)result_rx.at<double>(sp[i].point);
			if (opt_rect) {
			  ry = (int)result_ry.at<double>(sp[i].point);
			}
			cv::rectangle(bgr,cv::Point(sp[i].point.x-rx,sp[i].point.y-ry),cv::Point(sp[i].point.x+rx,sp[i].point.y+ry),cv::Scalar(0,0,255));
			//cv::rectangle(bgr,cv::Point(sp[i].point.x-rx-1,sp[i].point.y-ry-1),cv::Point(sp[i].point.x+rx+1,sp[i].point.y+ry+1),cv::Scalar(0,0,255));
		}
	  }
	  if (opt_trace) {
		cv::imwrite(TRACEPREFIX "localmaxpoints.png", bgr);
	  }

	  // draw maximum symmetry point in original image
	  if (maxpoint.x == 0 && maxpoint.y == 0) {
		fputs("no symmetry point found\n", stderr);
		return {-1, 3, 0, 0};;
	  }
	  ry = rx = (int)result_rx.at<double>(maxpoint);
	  if (opt_rect) {
		ry = (int)result_ry.at<double>(maxpoint);
	  }

	  std::array<int, 4> symmetry_summary = {maxpoint.x, maxpoint.y, rx, ry};

	  if (opt_outfile != "") {
		cv::rectangle(img,cv::Point(maxpoint.x-rx,maxpoint.y-ry),cv::Point(maxpoint.x+rx,maxpoint.y+ry),cv::Scalar(0,0,255));
		cv::imwrite(opt_outfile, img);
	  }

	  return symmetry_summary;
	}
}