#pragma once

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
	**plt_name: char *, Optional name for plot
	*/
	void display_CV(cv::Mat &mat, float scale = 1.0f, char * plt_name = "OpenCV 2D Plot");

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
}