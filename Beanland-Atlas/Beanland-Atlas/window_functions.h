#pragma once

#include <defines.h>
#include <includes.h>

namespace ba
{
	/*Calculates values of Hann window function so that they are ready for repeated application
	**Inputs:
	**mat_rows: int, Number of rows in window
	**mat_cols: int, Number of columns in windo
	**NUM_THREADS: const int, Number of threads to use for OpenMP CPU acceleration
	**Returns:
	**cv::Mat, Values of Hann window function at the pixels making up the 
	*/
	cv::Mat create_hann_window(int mat_rows, int mat_cols, const int NUM_THREADS);

	/*Applies Hann window to an image
	**Inputs:
	**mat: cv::Mat &, Image to apply window to
	**win: cv::Mat &, Window to apply
	**NUM_THREADS: const int, Number of threads to use for OpenMP CPU acceleration
	*/
	void apply_win_func(cv::Mat &mat, cv::Mat &win, const int NUM_THREADS);
}