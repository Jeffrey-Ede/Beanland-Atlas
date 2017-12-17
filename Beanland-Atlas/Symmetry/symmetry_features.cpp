//
// Author:  Christoph Dalitz
// Version: 2.0a from 2014/05/14
//
// Computes the features for selected points of the symmetry transform
//

#include <make_defs.h>
#include <stdio.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include <math.h>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "symmetry_transform.h"

#define TRACEPREFIX "trace-"

namespace sym
{
	// command line arguments
	char* opt_infile = NULL;
	char* opt_pointfile = NULL;
	int opt_r = 0;
	double opt_alpha = 0.5;
	double opt_percentage = 0.5;
	int opt_s = 3;
	int opt_topn = 10;
	int opt_maxwin = 0;
	bool opt_rect = true;
	bool opt_log = false;
	bool opt_norm = false;
	bool opt_trace = false;
	char* opt_outfile = NULL;
	const char* usage = "Usage:\n\tsymmetry_features -r <r> [options] <infile>\nOptions:\n"
	  "\t-r <r>\n"
	  "\t     maximum radius of symmetry transform\n"
	  "\t-alpha <alpha>\n"
	  "\t     exponent for normalisation of symmetry score with r^alpha [0.5]\n"
	  "\t-p <p>\n"
	  "\t     percentage in (0,1) for feature skeleton_size [0.5]\n"
	  "\t-s <s>\n"
	  "\t     window size (radius) for computation of cov_eig and edge_dir [3]\n"
	  "\t-top <n>\n"
	  "\t     print features for top <n> highest local maxima [10]\n"
	  "\t-f <pointfile>\n"
	  "\t     file listing the points to examine in the format \"(x,y)\"\n"
	  "\t     with one point per line. White space and everything after\n"
	  "\t     the second parenthesis is ignored.\n"
	  "\t-maxwin <k>\n"
	  "\t     replace points in <pointfile> with position of a maximum of the\n"
	  "\t     symmetry transform in a window of size 2<k>+1 around each point\n"
	  "\t-square\n"
	  "\t     do not try rectangular regions, but only squares (rx=ry)\n"
	  "\t-o <outpng>\n"
	  "\t     draw detected symmetry as rectangle in file <outpng>\n"
	  "\t-gradlog\n"
	  "\t     transform gradient strength as log(1+|G|)\n"
	  "\t-trace\n"
	  "\t     write trace of interim images with prefix '" TRACEPREFIX "'\n"
	  "\t-v   print version and exit\n";


	//-------------------------------------------------------------
	// helper functions
	//-------------------------------------------------------------

	// read points from a file
	// the value property of point vectors is set to zero
	// rc = 0 (ok) or 1 (error)
	int read_points_from_file(const char* pointfile, SymmetryPointVector *sp)
	{
	  char buffer[1024];
	  char c;
	  char *p, *q;
	  int x,y,n;
	  std::string tok;
	  bool line_end;

	  sp->clear();
	  FILE* f = fopen(pointfile, "r");
	  if (!f) return 1;
	  n = 0;
	  while (fgets(buffer, 1024, f)) {
		n++;
		line_end = (buffer[strlen(buffer)-1] == '\n');
		q = NULL;
		x = y = -1;
		for (p=buffer; *p; p++) {
		  if (*p == '(') {
			q = p+1;
		  }
		  else if (q && *p == ',') {
			*p = 0;
			x = atoi(q);
			q = p+1;
		  }
		  else if (q && *p == ')') {
			*p = 0;
			y = atoi(q);
			break;
		  }
		}
		if (y<0 || x<0) {
		  fprintf(stderr, "parse error in line %i of file '%s'\n", n, pointfile);
		  return 1;
		}
		sp->push_back(SymmetryPoint(cv::Point(x,y), 0.0));
		if (!line_end && !feof(f)) {
		  do {c = fgetc(f);} while (c!='\n');
		}
	  }
	  return 0;
	}

	// replaces given SymmetryPoint with maximum in S inside window of size 2*k+1
	// returns 1, when replaced, 0 when not
	int replace_with_maximum(SymmetryPoint* sp, const cv::Mat &S, int k)
	{
	  int x,y,minx,maxx,miny,maxy;
	  cv::Point maxpoint = sp->point;
	  double maxval = S.at<double>(maxpoint);
	  minx = std::max(0, sp->point.x-k);
	  miny = std::max(0, sp->point.y-k);
	  maxx = std::min(S.cols-1, sp->point.x+k);
	  maxy = std::min(S.rows-1, sp->point.y+k);
	  for (y=miny; y<=maxy; y++) {
		for (x=minx; x<=maxx; x++) {
		  if (S.at<double>(y,x) > maxval) {
			maxval = S.at<double>(y,x);
			maxpoint = cv::Point(x,y);
		  }
		}
	  }
	  if (maxpoint != sp->point) {
		sp->point = maxpoint;
		sp->value = maxval;
		return 1;
	  }
	  return 0;
  
	}


	//-------------------------------------------------------------
	// main program
	//-------------------------------------------------------------
	int main(int argc, char** argv)
	{
	  cv::Mat img, grey, bgr, result, result_rx, result_ry, gradx, grady;
	  int rx,ry,i;
	  cv::Point maxpoint(0,0);

	  // parse command line
	  for (i=1; i<argc; i++) {
		if (0==strcmp("-r", argv[i])) {
		  i++; if (i<argc) opt_r = atoi(argv[i]);
		}
		else if (0==strcmp("-p", argv[i])) {
		  i++; if (i<argc) opt_percentage = atof(argv[i]);
		}
		else if (0==strcmp("-s", argv[i])) {
		  i++; if (i<argc) opt_s = atoi(argv[i]);
		}
		else if (0==strcmp("-alpha", argv[i])) {
		  i++; if (i<argc) opt_alpha = atof(argv[i]);
		}
		else if (0==strcmp("-top", argv[i])) {
		  i++; if (i<argc) opt_topn = atoi(argv[i]);
		}
		else if (0==strcmp("-f", argv[i])) {
		  i++; if (i<argc) opt_pointfile = argv[i];
		}
		else if (0==strcmp("-maxwin", argv[i])) {
		  i++; if (i<argc) opt_maxwin = atoi(argv[i]);
		}
		else if (0==strcmp("-square", argv[i])) {
		  opt_rect = false;
		}
		else if (0==strcmp("-o", argv[i])) {
		  i++; if (i<argc) opt_outfile = argv[i];
		}
		else if (0==strcmp("-v", argv[i])) {
		  printf("%s\n", VERSION);
		  return 0;
		}
		else if (0==strcmp("-gradlog", argv[i])) {
		  opt_log = true;
		}
		else if (0==strcmp("-trace", argv[i])) {
		  opt_trace = true;
		}
		else if (argv[i][0] == '-') {
		  fputs(usage, stderr);
		  return 1;
		}
		else {
		  opt_infile = (char*)argv[i];
		} 
	  }
	  if (!opt_r || !opt_infile) {
		fputs(usage, stderr);
		return 1;
	  }
	  if (opt_percentage <= 0 || opt_percentage >= 1) {
		fputs("percentage (option -p) must be in ]0,1[\n", stderr);
		return 1;
	  }
	  if (opt_pointfile && (0 != _access(opt_pointfile, R_OK))) {
		fprintf(stderr, "Cannot read file '%s'\n", opt_pointfile);
		fputs(usage, stderr);
		return 1;
	  }

	  // load image and call symmetry transform
	  img = cv::imread(opt_infile);
	  if (!img.data) {
		fprintf(stderr, "cannot open '%s'\n", opt_infile);
		return 1;
	  }
	  cv::cvtColor(img,grey,CV_RGB2GRAY);
	  if (opt_log) {
		GradientNormalization gradnorm;
		if (opt_rect) {
		  symmetry_transform_rect(grey, result, result_rx, result_ry, opt_r, opt_alpha, 0, &gradnorm);
		} else {
		  symmetry_transform(grey, result, result_rx, opt_r, opt_alpha, 0, &gradnorm);
		}
	  } else {
		if (opt_rect) {
		  symmetry_transform_rect(grey, result, result_rx, result_ry, opt_r, opt_alpha);
		} else {
		  symmetry_transform(grey, result, result_rx, opt_r, opt_alpha);
		}
	  }
	  if (opt_trace) {
		float2grey(result,grey);
		cv::imwrite(TRACEPREFIX "symmetry.png", grey);
	  }

	  // determine points to be examined
	  SymmetryPointVector sp;
	  if (opt_pointfile) {
		if (read_points_from_file(opt_pointfile, &sp)) {
		  fputs("error in reading pointfile\n", stderr);
		  return 2;
		}
		for (i=0; i<(int)sp.size(); i++) {
		  //printf("(%i, %i)\n", sp[i].point.x, sp[i].point.y);
		  sp[i].value = result.at<double>(sp[i].point);
		  if (opt_maxwin > 0) {
			if (opt_trace)
			  printf("(%i,%i) -> ", sp[i].point.x, sp[i].point.y);
			replace_with_maximum(&(sp[i]), result, opt_maxwin);
			if (opt_trace)
			  printf("(%i,%i)\n", sp[i].point.x, sp[i].point.y);       
		  }
		}
	  }
	  else {
		// find all local maxima and sort them by symmetry score
		// BEWARE: sorting is in ascending order!
		SymmetryPointVector sp_all;
		if (0 >= points_local_maxima(result, sp_all)) {
		  fputs("no symmetry points found\n", stderr);
		  return -2;
		}
		std::sort(sp_all.begin(), sp_all.end());
		int topn_i = (int)sp_all.size()-opt_topn-1;
		if (topn_i < 0) topn_i = 0;
		for (i=(int)sp_all.size()-1; i>topn_i; i--) {
		  sp.push_back(sp_all[i]);
		}
	  }
	  // compute features for topn maxima
	  cv::Sobel(result, gradx, CV_64F, 1, 0);
	  cv::Sobel(result, grady, CV_64F, 0, 1);
	  if (opt_trace) {
		float2bgr(result,bgr);
	  }
	  double feature;
	  fputs("#(x,y):edge_dir,skel_size,anti_par,cov_ratio,diag_sum\n", stdout);
	  for (i=0; i<(int)sp.size(); i++) {
		// edge directedness
		feature = edge_directedness_cd(gradx, grady, sp[i].point, opt_s, 16);
		printf("(%i,%i):%5.4f", sp[i].point.x, sp[i].point.y, feature);
		// skeleton size = length of skeleton divided by diagonal of symmetry region
		PointVector skeleton;
		get_skeleton(sp[i].point, result, &skeleton, opt_percentage, opt_r);
		if (opt_trace) {
		  bgr.at<cv::Vec3b>(sp[i].point) = cv::Vec3b(0,0,255);
		}
		rx = result_rx.at<double>(sp[i].point);
		ry = rx;
		if (opt_rect) {
		  ry = result_ry.at<double>(sp[i].point);
		}
		feature = skeleton_size(sp[i].point, skeleton, rx, ry);
		printf(",%5.4f", feature);
		// antiparallelity
		antiparallelity_rpf(gradx, grady, sp[i].point, (int)std::min(rx, ry), feature);
		printf(",%5.4f", feature);
		// covariance eigenratio
		feature = coveigenratio_cd(result, sp[i].point, opt_s);
		printf(",%5.4f", feature);
		// diagonal gradient product sum
		feature = diagonal_sum(sp[i].point, gradx, grady, rx, ry);
		printf(",%5.4f", feature);
		fputs("\n", stdout);
	  }
	  if (opt_trace) {
		cv::imwrite(TRACEPREFIX "points.png", bgr);
	  }

	  return 0;
	}
}