#include <beanland_atlas.h>

/*Calculate the relative positions between images needed to align them.
**Inputs:
**mats: cv::Mat &, Images
**hann_LUT: cv::Mat &, Precalculated look up table to apply Hann window function with
**annulus: af::array &, Annulus to convolve gradiated image with
**gauss_fft: af::array &, Fourier transform of Gaussian used to blur the annulus to remove high frequency components
**order: int, Number of times to recursively convolve the blurred annulus with itself
**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
**NUM_THREADS: const int, Number of threads to use for OpenMP CPU acceleration
**Return:
**std::vector<cv::Vec3f>, Positions of each image relative to the first. The third element of the cv::Vec3f holds the value
**of the maximum phase correlation between successive images
*/
std::vector<cv::Vec3f> img_rel_pos(std::vector<cv::Mat> &mats, cv::Mat &hann_LUT, af::array &annulus, af::array &gauss_fft, int order,
	int mats_rows_af, int mats_cols_af, const int NUM_THREADS)
{
	//Assign memory to store relative image positions and their phase correlation weightings
	std::vector<cv::Vec3f> positions(mats.size() - 1);

	af::array xcorr_primer;
	af::array fft;
    #pragma omp parallel num_threads(NUM_THREADS)
	{
        #pragma omp single nowait
		{
			//Create Fourier domain matrix to multiply the Fourier transformed images with before phase correlating
			xcorr_primer = create_xcorr_primer(annulus, order, gauss_fft, mats_rows_af, mats_cols_af);
		}

        #pragma omp schedule(dynamic) nowait
		{
			//Hanning window the first image
			cv::Mat hanned_img = apply_win_func(mats[0], hann_LUT, NUM_THREADS);

			//Load the image
			cv::Mat image32F;
			hanned_img.convertTo(image32F, CV_32FC1, 1);
			af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

			//Fourier transform the image's Sobel filtrate
			af_array fft_af;
			af_fft2_r2c(&fft_af, af::sobel(inputImage_af, SOBEL_SIZE, false).get(), 1.0f, mats_rows_af, mats_cols_af);
			af::array fft = af::array(fft_af);
		}
	}

	//Prepare first image to be phase correlated
	af::array primed_fft_prev = xcorr_primer*fft;

	for (int i = 1; i < mats.size(); i++)
	{
		//Hanning window the image
		cv::Mat hanned_img = apply_win_func(mats[i], hann_LUT, NUM_THREADS - 1);

		//Load the image
		cv::Mat image32F;
		hanned_img.convertTo(image32F, CV_32FC1, 1);
		af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

		//Fourier transform the image
		af_array fft_af;
		af_fft2_r2c(&fft_af, af::sobel(inputImage_af, SOBEL_SIZE, false).get(), 1.0f, mats_rows_af, mats_cols_af);
		af::array fft = af::array(fft_af);

		//Prepare the image to be phase correlated
		af::array primed_fft = xcorr_primer*fft;

		//Find the position of the maximum phase correlation and its unnormalised value
		positions[i-1] = max_phase_corr(primed_fft, primed_fft_prev);
	}

	return positions;
}

/*Create Fourier domain matrix to multiply the Fourier transformed images with before phase correlating
**Inputs:
**annulus: af::array &, Annulus to perform cross correlation with
**order: int, Number of times to recursively cross correlate the annulus with itself
**gauss_fft: af::array &, Fourier transform of Gaussian used to blur the annuluses
**Return:
**af::array, Fourier domain matrix to multiply FFTs of images with to prime them for phase correlation
*/
af::array create_xcorr_primer(af::array &annulus, int order, af::array &gauss_fft, int mats_rows_af, int mats_cols_af)
{
	//Fourier transform annulus to prime it for convolution with the Gaussian
	af_array annulus_fft_af;
	af_fft2_r2c(&annulus_fft_af, annulus.get(), 1.0f, mats_rows_af, mats_cols_af);

	//Recursively convolute the Gaussian blurred annulus with itself in the Fourier domain
	return recur_conv(af::array(annulus_fft_af)*gauss_fft, order);
}

/*Use the convolution theorem to create a filter that performs the recursive convolution of a convolution filter with itself
**Inputs:
**filter: af::array &, Filter to recursively convolved with its own convolution
**n: int, The number of times the filter is recursively convolved with itself
**Return:
**af::array, Fourier domain matrix that can be elemtwise with another fft to recursively convolute it with a filter n times
*/
af::array recur_conv(af::array &filter, int n)
{
	return (n == 1 || n == 0) ? filter : recur_conv(filter*filter, n-1);
}

/*Finds the position of max phase correlation of 2 images from their Fourier transforms
**fft1: af::array &, One of the 2 Fourier transforms
**fft2: af::array &, One of the 2 Fourier transforms
**Return:
**cv::Vec3f, The relative position of the 2 images. The 3rd index is the value of the phase correlation
*/
cv::Vec3f max_phase_corr(af::array &fft1, af::array &fft2)
{
	//Create vector to store relative position and value of the phase correlation
	cv::Vec3f position;

	//Fourier transform the element-wise normalised cross-power spectrum
	af_array phase_corr;
	af_fft2_c2r(&phase_corr, (fft1*af::conjg(fft2)/af::abs(fft1*af::conjg(fft2))).get(), 1.0f, false);

	//Get position of maximum correlation
	af::array max1, idx1;
	af::array max, idx;
	af::max(max1, idx1, af::abs(af::array(phase_corr)), 1);
	af::max(max, idx, max1, 0);

	//Transfer results back to the host
	idx.host(&position[0]);
	idx1.host(&position[1]);
	max.host(&position[2]);

	return position;
}