#include <beanland_atlas.h>

/*Calculate the relative positions between images needed to align them.
**Inputs:
**mats: cv::Mat &, Images
**hann_LUT: cv::Mat &, Precalculated look up table to apply Hann window function with
**annulus: af::array &, Annulus to convolve gradiated image with
**circle: af::array &, Circle to convolve gradiated image with. Will remove annular cross correlation halo
**gauss_fft: af::array &, Fourier transform of Gaussian used to blur the annulus to remove high frequency components
**order: int, Number of times to recursively convolve the blurred annulus with itself
**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
**NUM_THREADS: const int, Number of threads to use for OpenMP CPU acceleration
**Return:
**std::vector<cv::Vec3f>, Positions of each image relative to the first. The third element of the cv::Vec3f holds the value
**of the maximum phase correlation between successive images
*/
std::vector<cv::Vec3f> img_rel_pos(std::vector<cv::Mat> &mats, cv::Mat &hann_LUT, af::array &annulus, af::array &circle, 
	af::array &gauss_fft, af::array impulsed_xcorr_blurer, int order, int mats_rows_af, int mats_cols_af, const int NUM_THREADS)
{
	//Assign memory to store relative image positions and their phase correlation weightings
	std::vector<cv::Vec3f> positions(mats.size() - 1);

	//Create Fourier domain matrix to multiply the Fourier transformed images with before phase correlating
	af::array xcorr_primer = create_xcorr_primer(annulus, order, gauss_fft, mats_rows_af, mats_cols_af);

	//Prepare Fourier transform of circle to impulse annular cross correlations
	af_array circle_c;
	af_fft2_r2c(&circle_c, annulus.get(), 1.0f, mats_rows_af, mats_cols_af);
	af::array impulser = gauss_fft*af::array(circle_c);

	//Load the Hanning window onto the GPU
	af::array hann_af(mats_rows_af, mats_cols_af, (float*)(hann_LUT.data));

	//Prepare first image to be aligned
	af::array primed_fft_prev = prime_img(mats[0], hann_af, xcorr_primer, impulser, impulsed_xcorr_blurer, 
		mats_rows_af, mats_cols_af);

	//Convert image to known data type
	cv::Mat image32F;
	mats[0].convertTo(image32F, CV_32FC1, 1);

	////Hanning window the first image
	//apply_win_func(image32F, hann_LUT, NUM_THREADS);

	//Load the image
	af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

	//Fourier transform the image's Sobel filtrate
	af_array fft_af;
	af_fft2_r2c(&fft_af, af::sobel(inputImage_af, SOBEL_SIZE, false).get(), 1.0f, mats_rows_af, mats_cols_af);
	af::array fft = af::array(fft_af);

	af_array ifft;
	//	af_fft2_c2r(&ifft, phase_corr, 1.0f, false);

	af::array primed_fft_prev = xcorr_primer*fft;

	//af::print("af", primed_fft_prev);

	//if(TEST){
	//	af_array ifft;
	//	af_fft2_c2r(&ifft, primed_fft_prev.get(), 1.0f, false);

	//	//const static int width = size, height = size;
	//	af::Window window(512, 512, "2D plot");
	//	do{
	//		window.image(af::abs(af::array(ifft)).as(f32)/(1e11));
	//	} while( !window.close() );
	//}

	//Use the phase correlation to find the relative positions of images
	for (int i = 1; i < mats.size(); i++)
	{
		//Convert image to known data type
		cv::Mat image32F;
		mats[i].convertTo(image32F, CV_32FC1, 1);

		//Hanning window the first image
		apply_win_func(image32F, hann_LUT, NUM_THREADS);

		//Load the image
		af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

		//Fourier transform the image
		af_array fft_af;
		af_fft2_r2c(&fft_af, af::sobel(inputImage_af, SOBEL_SIZE, false).get(), 1.0f, mats_rows_af, mats_cols_af);
		af::array fft = af::array(fft_af);

		//Prepare the image to be phase correlated
		af::array primed_fft = xcorr_primer*fft;

		//Find the position of the maximum phase correlation and its unnormalised value
		positions[i-1] = max_phase_corr(primed_fft, primed_fft_prev);

		std::cout << positions[i-1][0]/* << ", " << positions[i-1][1] << ", " << positions[i-1][2]*/ << std::endl;

		//primed_fft_prev = primed_fft;
	}

	//std::cout << positions[0][0] << ", " << positions[0][1] << ", " << positions[0][2] << std::endl;
	system("pause");

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
**fft2: af::array &, Second of the 2 Fourier transforms
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
	//af_fft2_c2r(&phase_corr, (fft1*fft2).get(), 1.0f, false);

	//if(TEST)
	//{
	//	af_array ifft;
	//	af_fft2_c2r(&ifft, phase_corr, 1.0f, false);

	//	//const static int width = size, height = size;
	//	af::Window window(512, 512, "2D plot");
	//	do{
	//		window.image(af::abs(af::array(phase_corr)).as(f32)/(1e21));
	//	} while( !window.close() );
	//}

	if(TEST){

		//const static int width = size, height = size;
		af::Window window(512, 512, "2D plot");
		do{
			window.image(af::abs(af::array(phase_corr)).as(f32)/(5e3));
		} while( !window.close() );
	}

	//Get position of maximum correlation
	af::array max1, idx1;
	af::array max, idx;
	af::max(max1, idx1, af::abs(af::array(phase_corr)), 1);
	af::max(max, idx, max1, 0);

	//Transfer results back to the host
	idx.as(f32).host(&position[0]);
	idx1(idx).as(f32).host(&position[1]);
	max.host(&position[2]);

	return position;
}

//Temporary: for convolution primer prototype
cv::Vec3f ssd(af::array &W0, af::array &I0, af::array &S0, af::array &W1, af::array &I1, af::array &S1)
{
	//Create vector to store relative position and value of the phase correlation
	cv::Vec3f position;

	//Fourier transform the element-wise normalised cross-power spectrum
	af_array ssd;
	af_fft2_c2r(&ssd, (W0*af::conjg(S1) + S0*af::conjg(W1) - 2*I0*af::conjg(I1)).get(), 1.0f, false);

	//if(TEST){

	//	//const static int width = size, height = size;
	//	af::Window window(512, 512, "2D plot");
	//	do{
	//		window.image(af::abs(af::array(ssd)).as(f32)/(2e14));
	//	} while( !window.close() );
	//}

	//Get position of maximum correlation
	af::array max1, idx1;
	af::array max, idx;
	af::min(max1, idx1, af::abs(af::array(ssd)), 1);
	af::min(max, idx, max1, 0);

	//Transfer results back to the host
	idx.as(f32).host(&position[0]);
	idx1(idx).as(f32).host(&position[1]);
	max.host(&position[2]);

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
af::array prime_img(cv::Mat &img, af::array &hann_af, af::array &xcorr_primer, af::array &impulser, af::array &impulsed_xcorr_blurer,
	int mats_rows_af, int mats_cols_af)
{
	//Convert image to known data type
	cv::Mat image32F;
	img.convertTo(image32F, CV_32FC1, 1);

	//Load the image onto the GPU
	af::array img_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

	//Fourier transform the Hanning windowed image's Sobel filtrate
	af_array sobel_filtrate;
	af_fft2_r2c(&sobel_filtrate, af::sobel(hann_af*img_af, SOBEL_SIZE, false).get(), 1.0f, mats_rows_af, mats_cols_af);

	//Cross correlate the the Sobel filtrate with the Gaussian blurred annulus in the Fourier domain
	af_array annular_xcorr;
	af_fft2_c2r(&annular_xcorr, (xcorr_primer*af::array(sobel_filtrate)).get(), 1.0f, false);

	//Cross correlate the image with the Gaussian blurred circle in the Fourier domain
	af_array img_fft;
	af_fft2_r2c(&img_fft, (hann_af*img_af).get(), 1.0f, mats_rows_af, mats_cols_af);
	af_array circle_xcorr;
	af_fft2_c2r(&circle_xcorr, (impulser*af::array(img_fft)).get(), 1.0f, false);

	//Gaussian blur the recursive annular cross correlation scaled by the circle cross correlation
	af_array impulsed_xcorr;
	af_fft2_r2c(&impulsed_xcorr, (af::array(annular_xcorr)*af::array(circle_xcorr)).get(), 1.0f, mats_rows_af, mats_cols_af);
	af_array primed_img;
	af_fft2_c2r(&primed_img, (impulsed_xcorr_blurer*af::array(impulsed_xcorr)).get(), 1.0f, false);

	return af::array(primed_img);
}