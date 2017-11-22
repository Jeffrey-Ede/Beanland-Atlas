#include <beanland_atlas.h>

/*Calculate upper bound for the size of circles in an image. This is done by convolving images with a gaussian low pass filter in the
**Fourier domain. The amplitudes of freq radii are then caclulated for the processed image. These are rebinned to get a histogram with
**equally spaced histogram bins. This process is repeated until the improvement in the autocorrelation of the bins no longer significantly
**increases as additional images are processed. The maximum amplitude above a minimum threshold is then used to set an upper limit on
**the size of the circles
**Inputs:
**&mats: std::vector<cv::Mat>, Vector of input images
**min_circ_size: int, Minimum dimeter of circles, in px
**autocorr_req_conv: float, Ratio of increased autocorrelation to previous must be at least this
**max_num_imgs: int, If autocorrelations don't converge after this number of images, method has failed
**sigma: float, standard deviation of Gaussian used to blur image
**af_context: cl_context, ArrayFire context
**af_device_id: cl_device_id, ArrayFire device
**af_queue: cl_command_queue, ArrayFire command queue
**NUM_THREADS: const int, Number of threads supported for OpenMP
**Returns:
**int, Upper bound for circles size, 0 if upper bound can't be determined
*/
int circ_size_ubound(std::vector<cv::Mat> &mats, int mats_rows_af, int mats_cols_af, int min_circ_size, float autocorr_req_conv,
	int max_num_imgs, float sigma, cl_context af_context, cl_device_id af_device_id, cl_command_queue af_queue, const int NUM_THREADS)
{
	//Ratio of autocorrelation to previous autocorrelation
	float convergence = 0.0f;

	//Load first image onto GPU
	cv::Mat image32F;
	mats[0].convertTo(image32F, CV_32FC1, 1);
	af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

	//Fourier transform the image
	af_array inputC = inputImage_af.get();
	af_array fft2_af;
	af_fft2_r2c(&fft2_af, inputC, 1.0f, mats_rows_af, mats_cols_af);

	//Create extended Gaussian creating kernel
	cl_kernel gauss_kernel = create_kernel(gauss_kernel_ext_source, gauss_kernel_ext_kernel, af_context, af_device_id);

	//Create the extended Gaussian
	af::array ext_gauss = extended_gauss(mats[0].rows, mats[0].cols, sigma, gauss_kernel, af_queue);
	af_array ext_gaussC = ext_gauss.get();

	//Fourier transform the Gaussian
	af_array gauss_fft2_af;
	af_fft2_r2c(&gauss_fft2_af, ext_gaussC, 1.0f, mats_rows_af, mats_cols_af);

	//Create kernel
	cl_kernel freq_spectrum_kernel = create_kernel(freq_spectrum1D_source, freq_spectrum1D_kernel, af_context, af_device_id);

	//Multiply the Fourier transformed image with the Fourier transform of the Gaussian to remove high frequency noise,
	//then create the frequency spectrum
	int reduced_height = mats_rows_af/2 + 1;
	int spectrum_size = std::max(mats_rows_af, mats_cols_af)/2;
	float inv_height2 = 1.0f/(mats_rows_af*mats_rows_af);
	float inv_width2 = 1.0f/(mats_cols_af*mats_cols_af);
	af::array spectrum = freq_spectrum1D(af::abs(af::array(fft2_af)*af::array(gauss_fft2_af)), spectrum_size, 
		mats_rows_af, mats_cols_af, reduced_height, inv_height2, inv_width2, freq_spectrum_kernel, af_queue);

	//Transfer 1D Fourier spectrum back to the host while calculating spectrum for the next micrograph
	std::vector<float> spectrum_host(spectrum_size);

	//Additionally, calculate expected number of contributions to each element of frequency spectrum on CPU to avoid
	//having to compile a 2nd kernel on the GPU as that kernel would only have 1 use
	std::vector<int> spectrum_contrib(spectrum_size, 0);

	int num_sections = NUM_THREADS >= 3 ? 3 : (NUM_THREADS == 2 ? 2 : 1);
	#pragma omp parallel sections num_threads(num_sections)  
    {  
		//Transfer data to host
        #pragma omp section
		{
			spectrum.host(&spectrum_host[0]);
		}

		//Calculate number of contributions to each frequency bin
        #pragma omp section
		{
			int half_width = mats_cols_af/2;
			for (int i = 0; i < reduced_height; i++) {
				for (int j = 0; j < mats_cols_af; j++) {

					//Get distances from center of shifted 2D fft
					int y = i%(half_width+1);
					int x = i/(half_width+1);
					if (x > half_width)
					{
						x = mats_cols_af - x;
					}
					x += 1;

					//Increment contribution count each time a bin is indexed
					int idx = (int)(std::sqrt(x*x*inv_width2 + y*y*inv_height2)*INV_ROOTOF2*spectrum_size);
					if (idx < spectrum_size){
						spectrum_contrib[idx] += 1;
					}
					else
					{
						spectrum_contrib[spectrum_size-1] += 1;
					}
				}
			}
		}

		//Calculate spectrum of next image
        #pragma omp section
		{
			printf_s("Hello from thread %d\n", omp_get_thread_num());
		}
    }

	////Calculate Durbin-Watson autocorrelation statistic
	//float sum_diff_sqrd = 0.0f;
	//float sum_sqrs = spectrum_host[0]*spectrum_host[0];
	//for (int i = 1; i < spectrum_size; i++) {

	//	float diffs = spectrum_host[i] - spectrum_host[i-1]
	//		sum_diff_sqrd += diffs*diffs;
	//	sum_sqrs += spectrum_host[i]*spectrum_host[i];
	//}

	af::print("sfsd", spectrum);

	system("pause");

	//af::print("sdfsd", af::abs(af::array(gauss_fft2_af)));

	/*if(TEST){
		af::Window window(512, 512, "Plot");
		do{
			window.image(af::abs(af::array(gauss_fft2_af)).as(f32));
		} while( !window.close() );

		return 0;
	}*/


	////Get frequency spectrum for each image, loading next image whole doing calculations
	//for (int i = 1; i <= max_num_imgs; i++) {



	//	//Upper bound can be found
	//	if (convergence < autocorr_req_conv) {
	//		
	//		//Return upper bound
	//		return the_result;
	//	}
	//}

	//Upper bound could not be determined
	return 0;
}

/*Create extended Gaussian to blur images with to remove high frequency components. It
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
**af_kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
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
	float inv_max_freq = INV_ROOTOF2;

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
	input_af.unlock();
	output_af.unlock();
	

	return output_af;
}