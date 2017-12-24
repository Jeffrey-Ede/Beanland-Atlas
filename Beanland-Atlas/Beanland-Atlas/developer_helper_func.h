#pragma once

#include <defines.h>
#include <includes.h>

namespace ba
{
	/*Display C++ API ArrayFire array
	**Inputs:
	**arr: af::array &, ArrayFire C++ API array to display
	**scale: float, Multiply the array elements by this value before displaying
	**plt_name: char *, Optional name for plot
	**dim0: int, Optional image side size
	**dim1: int, Optional image side size
	*/
	void display_AF(af::array &arr, float scale = 1.0f, char * plt_name = "AF C++ API 2D Plot", int dim0 = 512, int dim1 = 512);

	/*Display C API ArrayFire array
	**Inputs:
	**arr: af_array &, ArrayFire C API array to display
	**scale: float, Multiply the array elements by this value before displaying
	**plt_name: char *, Optional name for plot
	**dim0: int, Optional image side size
	**dim1: int, Optional image side size
	*/
	void display_AF(af_array &arr, float scale = 1.0f, char * plt_name = "AF C API 2D Plot", int dim0 = 512, int dim1 = 512);

	/*Display OpenCV mat
	**Inputs:
	**mat: cv::Mat &, OpenCV mat to display
	**scale: float, Multiply the mat elements by this value before displaying
	**norm: bool, If true, min-max normalise the mat before displaying it with values in the range 0-255
	**plt_name: char *, Optional name for plot
	*/
	void display_CV(cv::Mat &mat, float scale = 1.0f, bool norm = true, char * plt_name = "OpenCV 2D Plot");

	/*Print the size of the first 2 dimensions of a C++ API ArrayFire array
	**Input:
	**arr: af::array &, Arrayfire C++ API array
	*/
	void print_AF_dims(af::array &arr);

	/*Print the size of the first 2 dimensions of a C API ArrayFire array
	**Input:
	**arr: af_array &, Arrayfire C API array
	*/
	void print_AF_dims(af_array &arr);

	/*Combines individual spots' surveys of k space into a single atlas. Surveys are positioned proportionally to their spot's position in 
	**the aligned average px values pattern
	**Inputs:
	**surveys: std::vector<cv::Mat> &, Surveys of k space made by each spot
	**spot_pos: std::vector<cv::Point> &, Positions of spots in aligned average image
	**radius: int, Radius of the spots being used
	**cols_diff: int, Difference between the minimum and maximum spot columns
	**rows_diff: int, Difference between the minimum and maximum spot rows
	**Return:
	cv::Mat, Atlas of the k space surveyed by the diffraction spots
	*/
	cv::Mat create_raw_atlas(std::vector<cv::Mat> &surveys, std::vector<cv::Point> &spot_pos, int radius, int cols_diff, int rows_diff);

	/*Print contents of vector, then wait for user input to continue. Defaults to printing the entire vector if no print size is specified
	**Inputs:
	**vect: std::vector<T> &, Vector to print
	**Num: const int &, Number of elements to print
	*/
	template <class T> void print_vect(std::vector<T> &vect, const int num = 0);
    
	//Include template function definitions below their prototypes
#include "developer_utility.hpp"
}