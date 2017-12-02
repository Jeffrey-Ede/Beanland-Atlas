#include <beanland_atlas.h>

namespace ba
{
	/*Convolve images with annulus with a range of radii and until the autocorrelation of product moment correlation of the
	**spectra decreases when adding additional images' contribution to the spectrum. The annulus radius that produces the best
	**fit is then refined from this spectrum, including the thickness of the annulus
	**Inputs:
	**mats: std::vector<cv::Mat> &, reference to a sequence of input images to use to refine the annulus parameters
	**min_rad: int, minimum radius of radius to create
	**max_rad: int, maximum radius of radius to create
	**init_thickness: int, Thickness to use when getting initial radius of annulus. The radii tested are separated by this value
	**max_contrib: int, Maximum number of images to use. If the autocorrelation of the autocorrelation of the cross correlation
	**still hasn't decreased after this many images, the parameters will be calculated from the images processed so far
	**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**gauss_fft_af: Fourier transform of Gaussian to blur annuluses with
	**af_context: cl_context, ArrayFire context
	**af_device_id: cl_device_id, ArrayFire device
	**af_queue: cl_command_queue, ArrayFire command queue
	**NUM_THREADS: const int, Number of threads supported for OpenMP
	**Returns:
	**std::vector<int>, Refined radius and thickness of annulus, in that order
	*/
	std::vector<int> get_annulus_param(cv::Mat &mat, int min_rad, int max_rad, int init_thickness, int max_contrib,
		int mats_rows_af, int mats_cols_af, af::array &gauss_fft_af, cl_kernel create_annulus_kernel, cl_command_queue af_queue, 
		int NUM_THREADS)
	{
		//Assign memory to store spectra of cross correlation maxima
		int spectrum_size = (max_rad-min_rad)/init_thickness + ((max_rad-min_rad)%init_thickness ? 1 : 0);
		std::vector<float> spectrum(spectrum_size);

		//Prepare arguments for repeated use by annulus creating kernel
		size_t length = mats_rows_af*mats_cols_af;
		int half_width = mats_cols_af/2;
		int half_height = mats_rows_af/2;

		/* Perform analysis on first image separately to grow ArrayFire abstract syntax tree */

		//Load image
		cv::Mat image32F;
		mat.convertTo(image32F, CV_32FC1, 1);
		af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

		//Fourier transform the Sobel filtrate
		af_array fft_af;
		af_fft2_r2c(&fft_af, af::sobel(inputImage_af, SOBEL_SIZE, false).get(), 1.0f, mats_rows_af, mats_cols_af);
		af::array fft = af::array(fft_af);

		//Create Fourier transform of the first annulus
		af_array annulus_fft_af;
		af_fft2_r2c(&annulus_fft_af, create_annulus(length, mats_cols_af, half_width, mats_rows_af, half_height, 
			min_rad, init_thickness, create_annulus_kernel, af_queue).get(), 1.0f, mats_rows_af, mats_cols_af);
		af::array annulus = af::array(annulus_fft_af);

		//Convolve the Fourier transforms of the annulus, Gaussian (to blur the annulus) and the image then inverse Fourier transform
		//to create the accumulator space
		af_array ifft;
		af_fft2_c2r(&ifft, (gauss_fft_af*annulus*fft).get(), 1.0f, false);

		//Get the maximum value of the cross-correlation
		af::array max1, idx1;
		af::array max, idx;
		af::max(max1, idx1, af::abs(af::array(ifft)), 1);
		af::max(max, idx, max1, 0);

		//Transfer the maximum back to the host
		max.host(&spectrum[0]);

		//Divide by radius of annulus to normalise results
		spectrum[0] /= sum_annulus_px(min_rad, init_thickness);

		//Increment spot radius
		for (int r = min_rad+init_thickness, i = 1; r < max_rad; r += init_thickness, i++)
		{
			//Create Fourier transform of the annulus
			af_array annulus_fft_af;
			af_fft2_r2c(&annulus_fft_af, create_annulus(length, mats_cols_af, half_width, mats_rows_af, half_height, 
				r, init_thickness, create_annulus_kernel, af_queue).get(), 1.0f, mats_rows_af, mats_cols_af);

			//Convolve the Fourier transforms of the annulus, Gaussian (to blur the annulus) and the image then inverse Fourier transform
			//to create the cross-correlation space space
			af_fft2_c2r(&ifft, (gauss_fft_af*af::array(annulus_fft_af)*fft).get(), 1.0f, false);

			//Get the maximum value of the cross-correlation
			af::max(max1, idx1, af::abs(af::array(ifft)), 1);
			af::max(max, idx, max1, 0);

			//Transfer the maximum back to the host
			max.host(&spectrum[i]);

			//Divide by radius of annulus to normalise results
			spectrum[i] /= sum_annulus_px(r, init_thickness);
		}

		//Use peak in spectrum to estimate the spot radius
		int rad = min_rad + std::distance(spectrum.begin(), std::max_element(spectrum.begin(), spectrum.end()))*init_thickness;

		return refine_annulus_param(rad, init_thickness, length, mats_cols_af, mats_rows_af, half_width, half_height, gauss_fft_af,
			fft, create_annulus_kernel, af_queue);
	}

	/*Calculates relative area of annulus to divide cross-correlations by so that they can be compared
	**Inputs:
	**rad: int, Average of annulus inner and outer radii
	**thickness: int, Thickness of annulus. 
	**Returns
	**float, Sum of values of pixels maxing up blurred annulus
	*/
	float sum_annulus_px(int rad, int thickness)
	{
		return rad*thickness;
	}

	/*Refines annulus radius and thickness estimate based on a radius estimate and p/m the accuracy that radius is known to
	**Inputs:
	**rad: int, Estimated spot radius
	**range: int, The refined radius estimate will be within this distance from the initial radius estimate.
	**length: size_t, Number of px making up padded annulus
	**mats_rows_af: int, Number of rows of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**mats_cols_af: int, Number of cols of ArrayFire array containing the images. This is transpositional to the OpenCV mat
	**half_width: int, Half the number of ArrayFire columns
	**half_height: int, Half the number of ArrayFire rows
	**gauss_fft_af: af::array, r2c ArrayFire FFT of Gaussian blurring kernel
	**fft: af::array, r2c ArrayFire FFT of input image
	**create_annulus_kernel: cl_kernel, OpenCL kernel that creates the padded annulus
	**af_queue: cl_command_queue, OpenCL command queue
	**Returns
	**std::vector<int>, Refined radius and thickness of annulus, in that order
	*/
	std::vector<int> refine_annulus_param(int rad, int range, size_t length, int mats_cols_af, int mats_rows_af, int half_width, 
		int half_height, af::array gauss_fft_af, af::array fft, cl_kernel create_annulus_kernel, cl_command_queue af_queue)
	{
		std::vector<int> ref_param(2, 0); //Highest cross correlation
		float max_xcorr = 0.0f; //Value of highest cross correlation

		//Interate accross radii in the range
		for (int r = rad-range; r < rad+range; r++)
		{
			//Innrement thickness from its minimum value (range) until it stops increasing
			float local_xcorr = 0.0f;
			float local_xcorr_prev = 0.0f;
			int t;
			{
				for (t = range; local_xcorr_prev <= local_xcorr; t++)
				{
					local_xcorr_prev = local_xcorr;

					//Create padded annular cross correlation filter
					af_array annulus_fft_af;
					af_fft2_r2c(&annulus_fft_af, create_annulus(length, mats_cols_af, half_width, mats_rows_af, half_height, 
						r, t, create_annulus_kernel, af_queue).get(), 1.0f, mats_rows_af, mats_cols_af);

					//Convolve the Fourier transforms of the annulus, Gaussian (to blur the annulus) and the image then inverse Fourier transform
					//to create the cross-correlation space space
					af_array ifft;
					af_fft2_c2r(&ifft, (gauss_fft_af*af::array(annulus_fft_af)*fft).get(), 1.0f, false);

					//Get the maximum value of the cross-correlation
					af::array max1, idx1;
					af::array max, idx;
					af::max(max1, idx1, af::abs(af::array(ifft)), 1);
					af::max(max, idx, max1, 0);

					max.host(&local_xcorr);

					local_xcorr /= sum_annulus_px(r, t);
				}
			}

			//Check if this spot is larger than all the others
			if (local_xcorr_prev > max_xcorr)
			{
				max_xcorr = local_xcorr_prev;

				//Update highest cross correlation parameters
				ref_param[0] = r;
				ref_param[1] = t;
			}
		}

		return ref_param;
	}
}