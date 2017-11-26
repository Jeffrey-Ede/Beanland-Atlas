#include <beanland_atlas.h>

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
std::vector<int> get_annulus_param(std::vector<cv::Mat> &mats, int min_rad, int max_rad, int init_thickness, int max_contrib,
	int mats_rows_af, int mats_cols_af, af::array gauss_fft_af, cl_kernel create_annulus_kernel, cl_command_queue af_queue, 
	int NUM_THREADS)
{
	//Assign memory to store spectra of cross correlation maxima
	int spectrum_size = (max_rad-min_rad)/init_thickness + ((max_rad-min_rad)%init_thickness ? 1 : 0);
	std::vector<float> crosscorr_spectrum(spectrum_size);
	std::vector<float> total_crosscorr_spectrum(spectrum_size);

	//Prepare arguments for repeated use by annulus creating kernel
	size_t length = mats_rows_af*mats_cols_af;
	int half_width = mats_cols_af/2;
	int half_height = mats_rows_af/2;

	/* Perform analysis on first image separately to grow ArrayFire abstract syntax tree */

	//Load image
	cv::Mat image32F;
	mats[0].convertTo(image32F, CV_32FC1, 1);
	af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(mats[0].data));

	//Fourier transform the Sobel filtrate
	af_array fft_af;
	af_fft2_r2c(&fft_af, af::sobel(inputImage_af, 3, false).get(), 1.0f, mats_rows_af, mats_cols_af);

	//Create Fourier transform of the first annulus
	af_array annulus_fft_af;
	af_fft2_r2c(&annulus_fft_af, create_annulus(length, mats_cols_af, half_width, mats_rows_af, half_height, 
		min_rad, init_thickness, create_annulus_kernel, af_queue).get(), 1.0f, mats_rows_af, mats_cols_af);

	//Convolve the Fourier transforms of the annulus, Gaussian (to blur the annulus) and the image then inverse Fourier transform
	//to create the accumulator space
	af_array ifft;
	af_fft2_c2r(&ifft, (gauss_fft_af*af::array(annulus_fft_af)*af::array(fft_af)).get(), 1.0f, false);

	//Get the maximum value of the cross-correlation
	af::array max1, idx1;
	af::array max, idx;
	af::max(max1, idx1, af::abs(af::array(ifft)), 1);
	af::max(max, idx, max1, 0);

	//Transfer the maximum back to the host
	max.host(&total_crosscorr_spectrum[0]);
	
	int img_idx_incr = mats.size()/max_contrib;

	////Load image
	//cv::Mat image32F;
	//std::cout << i*img_idx_incr << std::endl; 
	//mats[i*img_idx_incr].convertTo(image32F, CV_32FC1, 1);

	////Sweep through radii
	//for (int r = min_rad+init_thickness, i = 1; r < max_rad; r += init_thickness, i++)
	//{
	//	af::array inputImage_af(mats_rows_af, mats_cols_af, (float*)(image32F.data));

	//	//Fourier transform the Sobel filtrate
	//	af_fft2_r2c(&fft_af, af::sobel(inputImage_af, 3, false).get(), 1.0f, mats_rows_af, mats_cols_af);

	//	//Create Fourier transform of the first annulus
	//	af_fft2_r2c(&annulus_fft_af, create_annulus(length, mats_cols_af, half_width, mats_rows_af, half_height, 
	//		r, init_thickness, create_annulus_kernel, af_queue).get(), 1.0f, mats_rows_af, mats_cols_af);

	//	//Convolve the Fourier transforms of the annulus, Gaussian (to blur the annulus) and the image then inverse Fourier transform
	//	//to create the accumulator space
	//	af_fft2_c2r(&ifft, (gauss_fft_af*af::array(annulus_fft_af)*af::array(fft_af)).get(), 1.0f, false);

	//	//Get the maximum value of the cross-correlation
	//	af::max(max1, idx1, af::abs(af::array(ifft)), 1);
	//	af::max(max, idx, max1, 0);

	//	//Transfer the maximum back to the host
	//	max.host(&total_crosscorr_spectrum[i]);
	//}

	//for (int i = 0; i < spectrum_size; i++) {
	//	std::cout << total_crosscorr_spectrum[i] << std::endl;
	//}

	system("pause");

	//af::array a = af::abs(af::array(ifft));

	////af::print("af", a);

	//if(TEST){
	//	//const static int width = size, height = size;
	//	af::Window window(512, 512, "2D plot");
	//	do{
	//		window.image(a.as(f32)/50000000000.0);
	//	} while( !window.close() );
	//}

	std::vector<int> something;
	return something;
}