#include <beanland_atlas.h>

/*Calculate the relative positions between images needed to align them.
**Inputs:
**mats: cv::Mat &, Images
**hann_LUT: cv::Mat &, Precalculated look up table to apply Hann window function with
**annulus: af::array &, Annulus to convolve gradiated image with
**circle: af::array &, Circle to convolve gradiated image with. Will remove annular cross correlation halo
**gauss_fft: af::array &, Fourier transform of Gaussian used to blur the annulus to remove high frequency components
**impulsed_xcorr_blurer: af::array &, Fourier transform of a Gaussian designed to blur the impulsed annular cross correlation
**order: int, Number of times to recursively convolve the blurred annulus with itself
**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
**Return:
**std::vector<std::array<float, 5>>, Positions of each image relative to the first. The third element of the cv::Vec3f holds the value
**of the maximum phase correlation between successive images
*/
std::vector<std::array<float, 5>> img_rel_pos(std::vector<cv::Mat> &mats, cv::Mat &hann_LUT, af::array &annulus, af::array &circle, 
	af::array &gauss_fft, int order, int mats_rows_af, int mats_cols_af)
{
	//Assign memory to store relative image positions and their phase correlation weightings
	std::vector<std::array<float, 5>> positions(mats.size());

	//In this preliminary version of the function, image positions are measured relative to the first image. Set first index parameters
	//A more advanced method will be implemented later
	positions[0] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

	//Create Fourier domain matrix to multiply the Fourier transformed images with before phase correlating
	af::array xcorr_primer = create_xcorr_primer(annulus, order, gauss_fft, mats_rows_af, mats_cols_af);

	//Prepare Fourier transform of circle to impulse annular cross correlations
	af_array circle_c;
	af_fft2_r2c(&circle_c, annulus.get(), 1.0f, mats_rows_af, mats_cols_af);
	af::array impulser = gauss_fft*af::array(circle_c);

	//Load the Hanning window onto the GPU
	af::array hann_af(mats_rows_af, mats_cols_af, (float*)(hann_LUT.data));

	//Prepare first image to be aligned
	af::array primed_fft_prev = prime_img(mats[0], hann_af, xcorr_primer, impulser,	mats_rows_af, mats_cols_af);

	//Use the phase correlation to find the relative positions of images
	for (int i = 1; i < mats.size(); i++)
	{
		//Prepare the image to be phase correlated
		af::array primed_fft = prime_img(mats[i], hann_af, xcorr_primer, impulser, mats_rows_af, mats_cols_af);

		//Find the position of the maximum phase correlation and its unnormalised value
		positions[i] = max_phase_corr(primed_fft, primed_fft_prev, i, 0);

		if (positions[i-1][0] >= 128)
		{
			positions[i-1][0] -= 255;
		}

		if (positions[i-1][1] >= 128)
		{
			positions[i-1][1] -= 255;
		}
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
af::array create_xcorr_primer(af::array &annulus, int order, af::array &gauss_fft,
	int mats_rows_af, int mats_cols_af)
{
	//Fourier transform annulus to prime it for convolution with the Gaussian
	af_array annulus_fft_af;
	af_fft2_r2c(&annulus_fft_af, annulus.get(), 1.0f, mats_rows_af, mats_cols_af);

	//Recursively convolve the Gaussian blurred annulus with itself in the Fourier domain
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
**fft2: af::array &, Second of the 2 Fourier transforms
**img_idx1: int, index of the image used to create the first of the 2 Fourier transforms
**img_idx2: int, index of the image used to create the second of the 2 Fourier transforms
**Return:
**std::array<float, 5>, The 0th and 1st indices are the relative positions of images, the 2nd index is the value of the phase correlation
**and the 3rd and 4th indices hold the indices of the images being compared in the OpenCV mats container 
*/
std::array<float, 5> max_phase_corr(af::array &fft1, af::array &fft2, int img_idx1, int img_idx2)
{
	//Create array to store relative position, value of the phase correlation and the identities of the images being compared
	std::array<float, 5> position;

	//Fourier transform the element-wise normalised cross-power spectrum
	af_array phase_corr;
	af_fft2_c2r(&phase_corr, (fft1*af::conjg(fft2)/(af::abs(fft1*af::conjg(fft2)) + 1)).get(), 1.0f, false); // +1 to avoid divide by 0 errors

	//Get position of maximum correlation
	af::array max1, idx1;
	af::array max, idx;
	af::max(max1, idx1, af::abs(af::array(phase_corr)), 1);
	af::max(max, idx, max1, 0);

	//Transfer results back to the host
	idx.as(f32).host(&position[0]);
	idx1(idx).as(f32).host(&position[1]);
	max.host(&position[2]);

	//Record identities of images being compared
	position[3] = img_idx1;
	position[4] = img_idx2;

	return position;
}

/*Primes images for alignment. The primed images are the Gaussian blurred cross correlation of their Hann windowed Sobel filtrate
**with an annulus after it has been scaled by the cross correlation of the Hann windowed image with a circle
**img: cv::Mat &, Image to prime for alignment
**hann_af: af::array &, Hanning window function look up table
**xcorr_primer: af::array &, Fourier transform of the convolution of a Gaussian and an annulus
**impulser: af::array &, Fourier transform of the convolution of a Gaussian and a circle
**impulsed_xcorr_blurer: af::array &, Fourier transform of a Gaussian designed to blur the impulsed annular cross correlation
**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
**Return:
**af::array, Image primed for alignment
*/
af::array prime_img(cv::Mat &img, af::array &hann_af, af::array &xcorr_primer, af::array &impulser, int mats_rows_af, int mats_cols_af)
{
	//Convert image to known data type
	cv::Mat image32F;
	img.convertTo(image32F, CV_32FC1, 1);

	//Load the image onto the GPU
	af::array img_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

	//Fourier transform the Hanning windowed image's Sobel filtrate
	af_array sobel_filtrate;
	af_fft2_r2c(&sobel_filtrate, (/*hann_af**/af::sobel(img_af, SOBEL_SIZE, false)).get(), 1.0f, mats_rows_af, mats_cols_af);

	//Cross correlate the the Sobel filtrate with the Gaussian blurred annulus in the Fourier domain
	af_array annular_xcorr;
	af_fft2_c2r(&annular_xcorr, (1e-10 * xcorr_primer*af::array(sobel_filtrate)).get(), 1.0f, false);

	//Cross correlate the image with the Gaussian blurred circle in the Fourier domain
	af_array img_fft;
	af_fft2_r2c(&img_fft, (/*hann_af**/img_af).get(), 1.0f, mats_rows_af, mats_cols_af);
	af_array circle_xcorr;
	af_fft2_c2r(&circle_xcorr, (1e-10 * impulser*af::array(img_fft)).get(), 1.0f, false);

	//Scale the recursive annular cross correlation by the circle cross correlation
	af_array primed_img;
	af_fft2_r2c(&primed_img, (/*hann_af**/af::array(annular_xcorr)*af::array(circle_xcorr)).get(), 1.0f, mats_rows_af, mats_cols_af);

	return af::array(primed_img);
}