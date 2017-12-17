#pragma once

#ifdef SYM_API
#define SYM_API __declspec(dllexport)
#else
#define SYM_API __declspec(dllimport)
#endif

#include <array>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

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
	extern std::array<int, 4> symmetry(cv::Mat img, int opt_r, cv::Mat imgmask, double opt_alpha = 0.5, bool opt_rect = true, int opt_axial = 3, int opt_only = 0,
		char* opt_outfile = "sym_out", bool opt_log = false, bool opt_trace = false, int opt_trace_greytrans = 0);
}