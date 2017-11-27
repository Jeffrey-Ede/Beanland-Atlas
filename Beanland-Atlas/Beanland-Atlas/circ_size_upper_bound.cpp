#include <beanland_atlas.h>

/*Calculate upper bound for the size of circles in an image. This is done by convolving images with a gaussian low pass filter in the
**Fourier domain. The amplitudes of freq radii are then caclulated for the processed image. These are rebinned to get a histogram with
**equally spaced histogram bins. This process is repeated until the improvement in the autocorrelation of the bins no longer significantly
**increases as additional images are processed. The error-weighted centroid of the 1D spectrum is then used to generate an upper bound for
**the separation of the circles
**Inputs:
**&mats: std::vector<cv::Mat>, Vector of input images
**mats_rows_af: int, Rows in ArrayFire array containing an input image. ArrayFire arrays are transpositional to OpenCV mats
**mats_cols_af: int, Rows in ArrayFire array containing an input image. ArrayFire arrays are transpositional to OpenCV mats
**gauss: af::array, ArrayFire array containing Fourier transform of a gaussian blurring filter to reduce high frequency components of
**the Fourier transforms of images with
**min_circ_size: int, Minimum dimeter of circles, in px
**max_num_imgs: int, If autocorrelations don't converge after this number of images, uses the current total to estimate the size
**af_context: cl_context, ArrayFire context
**af_device_id: cl_device_id, ArrayFire device
**af_queue: cl_command_queue, ArrayFire command queue
**NUM_THREADS: const int, Number of threads supported for OpenMP
**Returns:
**int, Upper bound for circles size
*/
int circ_size_ubound(std::vector<cv::Mat> &mats, int mats_rows_af, int mats_cols_af, af::array &gauss, int min_circ_size,
	int max_num_imgs, cl_context af_context, cl_device_id af_device_id, cl_command_queue af_queue, const int NUM_THREADS)
{
	//Create kernel
	cl_kernel freq_spectrum_kernel = create_kernel(freq_spectrum1D_source, freq_spectrum1D_kernel, af_context, af_device_id);

	//Multiply the Fourier transformed image with the Fourier transform of the Gaussian to remove high frequency noise,
	//then create the frequency spectrum
	int reduced_height = mats_rows_af/2 + 1;
	int spectrum_size = std::max(mats_rows_af, mats_cols_af)/2;
	float inv_height2 = 1.0f/(mats_rows_af*mats_rows_af);
	float inv_width2 = 1.0f/(mats_cols_af*mats_cols_af);

	//Assign memory to store spectrums on the host
	std::vector<float> spectrum_host(spectrum_size); //Individual spectrum

	//Assign memory to store errors of the spectrums on the host
	std::vector<float> spectrum_err(spectrum_size, 0);

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


	//Use a lagged, weighted Pearson normalised product moment correlation coefficient as a proxy for the Durbin-Watson
	//autocorrelation statistic, which is approx 2*(1-r_p)
	float spectrum_autocorr = wighted_pearson_autocorr(spectrum_host, spectrum_err, NUM_THREADS);

	//Use the weighted centroid of the 1D Fourier spectrum to estimate the spot separation
	float weighted_sum = 0.0f, sum_err = spectrum_host[0] / spectrum_err[0];
	for (int i = 1; i < spectrum_size; i++)
	{
		weighted_sum += spectrum_host[i]*i / spectrum_err[i];
		sum_err += spectrum_host[i] / spectrum_err[i];
	}

	//Free OpenCL resources
	clReleaseKernel(freq_spectrum_kernel);

	return (spectrum_size * sum_err / ( SQRT_OF_2 * weighted_sum ));
}