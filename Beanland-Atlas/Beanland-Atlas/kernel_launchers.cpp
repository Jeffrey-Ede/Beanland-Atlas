#include <beanland_atlas.h>

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
af::array extended_gauss(int cols, int rows, float sigma, cl_kernel kernel, cl_command_queue af_queue)
{
	int rows_half_width = rows/2;
	int cols_half_width = cols/2;
	float inv_sigma = 1.0f/sigma;
	float minus_half_inv_sigma2 = -0.5*inv_sigma*inv_sigma;

	//Create ArrayFire memory and transfer it to OpenCL
	size_t length = rows*cols;
	af::array output_af = af::constant(0, length, f32);
	cl_mem * output_cl = output_af.device<cl_mem>();

	//Pass arguments to kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), output_cl);
	clSetKernelArg(kernel, 1, sizeof(float), &inv_sigma);
	clSetKernelArg(kernel, 2, sizeof(float), &minus_half_inv_sigma2);
	clSetKernelArg(kernel, 3, sizeof(int), &rows_half_width);
	clSetKernelArg(kernel, 4, sizeof(int), &cols_half_width);
	clSetKernelArg(kernel, 5, sizeof(int), &rows);

	//Execute kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 1, NULL, &length, NULL, 0, NULL, NULL);

	//Convert OpenCL output to ArrayFire
	output_af.unlock();

	return af::moddims(output_af, rows, cols);
}

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
**kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
**af_queue: cl_command_queue, ArrayFire command queue
**Returns:
**af::array, ArrayFire array containing the frequency spectrum of the elementwise multiplication of the Fourier transforms
*/
af::array freq_spectrum1D(af::array input_af, size_t length, int height, int width, int reduced_height, float inv_height2, 
	float inv_width2, cl_kernel kernel, cl_command_queue af_queue)
{
	size_t num_data = reduced_height * width;
	cl_mem * input_cl = af::moddims(input_af, num_data).device<cl_mem>();

	//Create ArrayFire memory to hold spectrum and transfer it to OpenCL
	af::array output_af = af::constant(0, length, f32);
	cl_mem * output_cl = output_af.device<cl_mem>();

	//Prepare additional arguments for kernel
	int half_width = width/2;
	float inv_max_freq = SQRT_OF_2; //Half of 1 divided by sqrt(2) is sqrt(2)

	//Pass arguments to kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), input_cl);
	clSetKernelArg(kernel, 1, sizeof(cl_mem), output_cl);
	clSetKernelArg(kernel, 2, sizeof(int), &width);
	clSetKernelArg(kernel, 3, sizeof(int), &half_width);
	clSetKernelArg(kernel, 4, sizeof(float), &inv_width2);
	clSetKernelArg(kernel, 5, sizeof(float), &inv_height2);
	clSetKernelArg(kernel, 6, sizeof(float), &inv_max_freq);
	clSetKernelArg(kernel, 7, sizeof(size_t), &length);

	//Execute kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 1, NULL, &num_data, NULL, 0, NULL, NULL);

	//Transfer OpenCL memory back to ArrayFire
	output_af.unlock();

	//Free OpenCL resources
	clReleaseMemObject(*input_cl);

	return output_af;
}

/*Create padded unblurred annulus with a given radius and thickness. Its inner radius is radius - thickness/2 and the outer radius
** is radius + thickness/2 + (thickness%2 ? 0 : 1)
**Inputs:
**length: size_t, Number of pixels making up annulus
**width: int, Width of padded annulus
**half_width: int, Half the width of the padded annulus
**height: int, Height of padded annulus
**half_height: int, Half the height of the padded annulus
**radius: int, radius of annulus, approximately halfway between its inner and outer radii
**thickness: int, thickness of annulus. If even, the outer radius will be increased by 1
**kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
**af_queue: cl_command_queue, ArrayFire command queue
**Returns:
**af::array, ArrayFire array containing unblurred annulus
*/
af::array create_annulus(size_t length, int width, int half_width, int height, int half_height, int radius, int thickness, 
	cl_kernel kernel, cl_command_queue af_queue)
{
	//Create ArrayFire memory to hold spectrum and transfer it to OpenCL
	af::array output_af = af::constant(0, length, f32);
	cl_mem * output_cl = output_af.device<cl_mem>();

	//Prepare additional arguments for kernel
	int inner_rad2 = (radius - thickness/2)*(radius - thickness/2);
	int outer_rad2 = (radius + thickness/2 + (thickness%2 ? 0 : 1))*(radius + thickness/2 + (thickness%2 ? 0 : 1));

	//Pass arguments to kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), output_cl);
	clSetKernelArg(kernel, 1, sizeof(int), &inner_rad2);
	clSetKernelArg(kernel, 2, sizeof(int), &outer_rad2);
	clSetKernelArg(kernel, 3, sizeof(int), &half_width);
	clSetKernelArg(kernel, 4, sizeof(int), &half_height);
	clSetKernelArg(kernel, 5, sizeof(int), &width);

	//Execute kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 1, NULL, &length, NULL, 0, NULL, NULL);

	//Transfer OpenCL memory back to ArrayFire
	output_af.unlock();

	return af::moddims(output_af, width, height);
}

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
	cl_kernel kernel, cl_command_queue af_queue)
{
	//Create ArrayFire memory to hold spectrum and transfer it to OpenCL
	af::array output_af = af::constant(0, length, f32);
	cl_mem * output_cl = output_af.device<cl_mem>();

	//Prepare additional arguments for kernel
	int radius_squared = radius*radius;

	//Pass arguments to kernel
	clSetKernelArg(kernel, 0, sizeof(cl_mem), output_cl);
	clSetKernelArg(kernel, 1, sizeof(int), &radius_squared);
	clSetKernelArg(kernel, 2, sizeof(int), &half_width);
	clSetKernelArg(kernel, 3, sizeof(int), &half_height);
	clSetKernelArg(kernel, 4, sizeof(int), &width);

	//Execute kernel
	clEnqueueNDRangeKernel(af_queue, kernel, 1, NULL, &length, NULL, 0, NULL, NULL);

	//Transfer OpenCL memory back to ArrayFire
	output_af.unlock();

	return af::moddims(output_af, width, height);
}

