#pragma once

#include <includes.h>

namespace ba
{
	/*Utility function that builds a named kernel from source code. Will print errors if there are problems compiling it
	**Inputs:
	**kernel_sourceFile: const char*, File containing kernel source code
	**kernel_name: const char*, Name of kernel to be built
	**af_context: cl_context, Context to create kernel in
	**af_device_id: cl_device_id, Device to run kernel on
	**Returns:
	**cl_kernel, Build kernel ready for arguments to be passed to it
	*/
	cl_kernel create_kernel(const char* kernel_sourceFile, const char* kernel_name, cl_context af_context, cl_device_id af_device_id);

	/*Create extended Gaussian to blur images with to remove high frequency components.
	**Inputs:
	**rows: int, Number of columnss in input matrices. ArrayFire matrices are transposed so this is the number of rows of the ArrayFire
	**array
	**cols: int, Number of rows in input matrices. ArrayFire matrices are transposed so this is the number of columns of the ArrayFire
	**array
	**sigma: float, standard deviation of Gaussian used to blur image
	**kernel: cl_kernel, OpenCL kernel that creates extended Gaussian
	**af_queue: cl_command_queue, ArrayFire command queue
	**Returns:
	**af::array, ArrayFire array containing extended Guassian distribution
	*/
	af::array extended_gauss(int cols, int rows, float sigma, cl_kernel kernel, cl_command_queue af_queue);

	/*Multiplies 2 C API ArrayFire arrays containing r2c Fourier transforms and returns a histogram containing the frequency spectrum
	**of the result. Frequency spectrum bins all have the same width in the frequency domain
	**Inputs:
	**input_af: af_array, Amplitudes of r2c 2d Fourier transform
	**length: size_t, Number of bins for frequency spectrum
	**height: int, Height of original image
	**width: int, Hidth of original image and of fft
	**reduced_height: int, Height of fft, which is half the original hight + 1
	**inv_height2: float, 1 divided by the height squared
	**inv_width2: float, 1 divided by the width squared
	**af_kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
	**af_queue: cl_command_queue, ArrayFire command queue
	**Returns:
	**af::array, ArrayFire array containing the frequency spectrum of the elementwise multiplication of the Fourier transforms
	*/
	af::array freq_spectrum1D(af::array input_af, size_t length, int height, int width, int reduced_height, float inv_height2, 
		float inv_width2, cl_kernel kernel, cl_command_queue af_queue);

	/*Create padded unblurred annulus with a given radius and thickness. Its inner radius is radius - thickness/2 and the outer radius
	** is radius + thickness/2
	**Inputs:
	**length: size_t, Number of pixels making up annulus
	**width: int, Width of padded annulus
	**half_width: int, Half the width of the padded annulus
	**height: int, Height of padded annulus
	**half_height: int, Half the height of the padded annulus
	**radius: int, radius of annulus, approximately halfway between its inner and outer radii
	**thickness: int, thickness of annulus. If even, thickness will be rounded up to the next odd integer
	**kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
	**af_queue: cl_command_queue, ArrayFire command queue
	**Returns:
	**af::array, ArrayFire array containing unblurred annulus
	*/
	af::array create_annulus(size_t length, int width, int half_width, int height, int half_height, int radius, int thickness, 
		cl_kernel kernel, cl_command_queue af_queue);

	/*Create padded unblurred circle with a specified radius
	**Inputs:
	**length: size_t, Number of pixels making up annulus
	**width: int, Width of padded annulus
	**half_width: int, Half the width of the padded annulus
	**height: int, Height of padded annulus
	**half_height: int, Half the height of the padded annulus
	**radius: int, radius of annulus, approximately halfway between its inner and outer radii
	**kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
	**af_queue: cl_command_queue, ArrayFire command queue
	**Returns:
	**af::array, ArrayFire array containing unblurred circle
	*/
	af::array create_circle(size_t length, int width, int half_width, int height, int half_height, int radius,
		cl_kernel kernel, cl_command_queue af_queue);
}