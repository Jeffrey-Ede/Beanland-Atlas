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

	//Create extended Gaussian creating kernel
	cl_kernel gauss_kernel = create_kernel(gauss_kernel_ext_source, gauss_kernel_ext_kernel, af_context, af_device_id);

	//Create the extended Gaussian
	af::array ext_gauss = extended_gauss(mats[0].rows, mats[0].cols, sigma, gauss_kernel, af_queue);
	af_array ext_gaussC = ext_gauss.get();

	//Fourier transform the Gaussian
	af_array gauss_fft2_af;
	af_fft2_r2c(&gauss_fft2_af, ext_gaussC, 1.0f, mats_rows_af, mats_cols_af);
	af::array gauss = af::array(gauss_fft2_af);

	//Create kernel
	cl_kernel freq_spectrum_kernel = create_kernel(freq_spectrum1D_source, freq_spectrum1D_kernel, af_context, af_device_id);

	//Multiply the Fourier transformed image with the Fourier transform of the Gaussian to remove high frequency noise,
	//then create the frequency spectrum
	int reduced_height = mats_rows_af/2 + 1;
	int spectrum_size = std::max(mats_rows_af, mats_cols_af)/2;
	float inv_height2 = 1.0f/(mats_rows_af*mats_rows_af);
	float inv_width2 = 1.0f/(mats_cols_af*mats_cols_af);

	//Assign memory to store spectrums on the host
	std::vector<float> total_spectrum_host(spectrum_size); //Sum of all spectrums
	std::vector<float> spectrum_host(spectrum_size); //Individual spectrum

	//Assign memory to store errors of the spectrums on the host
	std::vector<float> spectrum_err(spectrum_size, 0);
	std::vector<float> total_spectrum_err(spectrum_size);

	//Calculate expected number of contributions to each element of the 1D frequency spectrum on CPU to avoid
	//having to compile a 2nd kernel on the GPU as that kernel would only have 1 use
	int half_width = mats_cols_af/2;
    #pragma omp parallel for num_threads(NUM_THREADS)
	for (int i = 0; i < reduced_height * mats_cols_af; i++) 
	{
		//Get distances from center of shifted 2D fft
		int y = i%(half_width+1);
		int x = i/(half_width+1);
		if (x > half_width)
		{
			x = mats_cols_af - x;
		}
		x += 1;

		//Increment contribution count each time a bin is indexed
		int idx = (int)(std::sqrt(x*x*inv_width2 + y*y*inv_height2)*SQRT_OF_2*spectrum_size);
		if (idx < spectrum_size)
		{
			spectrum_err[idx] += 1.0f;
		}
		else
		{
			spectrum_err[spectrum_size-1] += 1.0f;
		}
	}

	//Load first image onto GPU
	cv::Mat image32F;
	mats[0].convertTo(image32F, CV_32FC1, 1);
	af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

	//Fourier transform the image
	af_array fft2_af;
	af_fft2_r2c(&fft2_af, inputImage_af.get(), 1.0f, mats_rows_af, mats_cols_af);

	//Get the 1D frequency spectrum of that FFT
	freq_spectrum1D(af::abs(af::array(fft2_af)*gauss), spectrum_size, mats_rows_af, mats_cols_af,
		reduced_height, inv_height2, inv_width2, freq_spectrum_kernel, af_queue).host(&spectrum_host[0]);

	//Load second image and take it's fft
	mats[1].convertTo(image32F, CV_32FC1, 1);
	inputImage_af = af::array(mats_rows_af, mats_cols_af, (float*)(image32F.data));
	af_fft2_r2c(&fft2_af, inputImage_af.get(), 1.0f, mats_rows_af, mats_cols_af);

	//Get the 1D frequency spectrum of the second FFT
	freq_spectrum1D(af::abs(af::array(fft2_af)*gauss), spectrum_size, mats_rows_af, mats_cols_af,
		reduced_height, inv_height2, inv_width2, freq_spectrum_kernel, af_queue).host(&total_spectrum_host[0]);


    #pragma omp parallel sections num_threads(NUM_THREADS > = 2 ? 2 : 1)
	{
        #pragma omp section 
		{
			//Add the first 2 spectrums together
			std::transform(total_spectrum_host.begin(), total_spectrum_host.end(), spectrum_host.begin(),
				spectrum_host.end(), std::plus<float>());
		}

        #pragma omp section
		{
			//Get relative size of errors squared for each of the bins in the total spectrum of the first 2 images and for the first 
			//image by itself. These are proportional to the sqrt of the number of contributions in each bin
			for (int i = 0; i < spectrum_size; i++) {

				spectrum_err[i] = spectrum_err[i];
				total_spectrum_err[i] = 2 * spectrum_err[i];
			}
		}
	}

	//Use a lagged, weighted Pearson normalised product moment correlation coefficient as a proxy for the Durbin-Watson
	//autocorrelation statistic, which is approx 2*(1-r_p)
	float spectrum_autocorr_prev, spectrum_autocorr;
	#pragma omp parallel sections num_threads(NUM_THREADS > = 2 ? 2 : 1)
	{
	    //Get the autocorrelation of the first spectrum
        #pragma omp section 
		{
			spectrum_autocorr_prev = wighted_pearson_autocorr(spectrum_host, spectrum_err);
		}

		//Get the autocorrelation of the total spectrum
        #pragma omp section 
		{
			spectrum_autocorr = wighted_pearson_autocorr(total_spectrum_host, total_spectrum_err);
		}
	}

	//Keep calculating the autocorrelations of accumulated 1D Fourier spectra until the autocorrelation of the autocorrelation 
	//becomes negative
	for(int i = 2; spectrum_autocorr > spectrum_autocorr_prev; i++)
	{
		//Load image
		mats[i].convertTo(image32F, CV_32FC1, 1);
		af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

		//Take 2D FFT of the image
		af_fft2_r2c(&fft2_af, inputImage_af.get(), 1.0f, mats_rows_af, mats_cols_af);

		//Get 1D frequency spectrum from 2D FFT
		freq_spectrum1D(af::abs(af::array(fft2_af)*gauss), spectrum_size, mats_rows_af, mats_cols_af,
			reduced_height, inv_height2, inv_width2, freq_spectrum_kernel, af_queue).host(&spectrum_host[0]);

        #pragma omp parallel sections num_threads(NUM_THREADS >= 2 ? 2 : 1)
		{
            #pragma omp section
			{
				//Add it to the total spectrum
				std::transform(total_spectrum_host.begin(), total_spectrum_host.end(), spectrum_host.begin(),
					spectrum_host.end(), std::plus<float>());
			}

            #pragma omp section
			{
				//Update the spectral errors

				for (int j = 0; j < spectrum_size; j++) 
				{
					total_spectrum_err[j] = i * spectrum_err[j];
				}
			}
		}

		//Get the total spectrum's autocorrelation
		spectrum_autocorr_prev = spectrum_autocorr;
		spectrum_autocorr = wighted_pearson_autocorr(total_spectrum_host, total_spectrum_err);
	}

	//Find the maxima in Fourier space 
	int max_loc = std::distance(total_spectrum_host.begin(), std::max_element(total_spectrum_host.begin(), total_spectrum_host.end()));

	for (int i = 0; i < total_spectrum_host.size(); i++)
	{
		std::cout << total_spectrum_host[i] << std::endl;
	}

	//Refine maxima location by finding the centroid of indices of values above half the maxima's size

	//Get upper index where 2nd derivative changes sign
	//int u;
	//for (u = max_loc+1; total_spectrum_host[u-1] - 2*total_spectrum_host[u] + total_spectrum_host[u] < 0; u++) {}
	//
	////Get lower index where 2nd derivative changes sign
	//int d;
	//if (max_loc > 2)
	//{
	//	for (d = max_loc-1; total_spectrum_host[d-1] - 2*total_spectrum_host[d] + total_spectrum_host[d] < 0; d--)
	//	{
	//		if (d == 2)
	//		{
	//			d = 1;
	//			break;
	//		}
	//	}
	//}
	//else
	//{
	//	d = 1;
	//}

	////Find the centroid of the values between the indices where the 2nd derivatives change sign
	//float sum_weight_idx = total_spectrum_host[u]*u, sum_weight = total_spectrum_host[u];
	//for (int i = d; i < u; i++)
	//{
	//	sum_weight_idx += total_spectrum_host[i]*i;
	//	sum_weight += total_spectrum_host[i];
	//}

	//Use the weighted centroid of the 1D Fourier spectrum to estimate the spot separation
	float weighted_sum = 0.0f, sum_err = total_spectrum_host[0] /total_spectrum_err[0];
	for (int i = 1; i < spectrum_size; i++)
	{
		weighted_sum += total_spectrum_host[i]*i / total_spectrum_err[i];
		sum_err += total_spectrum_host[i] / total_spectrum_err[i];
	}

	std::cout << weighted_sum << ", " << sum_err << ", " << weighted_sum / sum_err << ", " << spectrum_size << std::endl;

	return spectrum_size * sum_err / ( SQRT_OF_2 * weighted_sum );
}

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
**af_kernel: cl_kernel, OpenCL kernel that creates the 1D frequency spectrum
**af_queue: cl_command_queue, ArrayFire command queue
**Returns:
**af::array, ArrayFire array containing the frequency spectrum of the elementwise multiplication of the Fourier transforms
*/
af::array freq_spectrum1D(af::array input_af, size_t length, int height, int width, int reduced_height, float inv_height2, 
	float inv_width2, cl_kernel kernel, cl_command_queue af_queue)
{
	size_t num_data = reduced_height * width;
	int dims0 = input_af.dims(0);
	int dims1 = input_af.dims(1);

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
	input_af.unlock();
	af::moddims(input_af, dims0, dims1);
	output_af.unlock();

	return output_af;
}